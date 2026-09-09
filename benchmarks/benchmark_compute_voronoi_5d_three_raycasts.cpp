#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/storage/hvdatabase.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 5;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::detail::EmptyLock,
    DatabaseParameters,
    Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;

struct Options {
    std::vector<std::size_t> node_counts{1000, 2000};
    std::size_t repeats = 3;
    std::uint64_t seed = 0x485642454e434835ULL; // "HVBENCH5"
    std::string domain = "both"; // bounded | open | both
    std::string search = "copy"; // copy | provider
    std::string raycast = "all"; // classic | inrange | combined | combined-fast | all
    std::string combined_fallback = "auto"; // auto | inrange
    bool verbose = false;
    std::string points_in;
    std::string points_out;
    std::string csv_out;
};

struct TimingResult {
    std::size_t nodes = 0;
    std::string domain;
    std::size_t repeat = 0;
    std::string search;
    std::string raycast;
    double mesh_setup_seconds = 0.0;
    double search_setup_seconds = 0.0;
    double algorithm_setup_seconds = 0.0;
    double compute_seconds = 0.0;
    double total_seconds = 0.0;
    std::size_t finite_vertices = 0;
    std::size_t infinite_edges = 0;
};

[[nodiscard]] double seconds_between(
    Clock::time_point begin,
    Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

[[nodiscard]] std::size_t next_power_of_two(std::size_t value) {
    if (value <= 1) {
        return 1;
    }
    --value;
    for (std::size_t shift = 1; shift < sizeof(value) * 8; shift <<= 1U) {
        value |= value >> shift;
    }
    return value + 1;
}

class SplitMix64 final {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    [[nodiscard]] std::uint64_t next_u64() noexcept {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }

    [[nodiscard]] double uniform01() noexcept {
        // Exact binary conversion using the high 53 random bits.
        return static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
    }

private:
    std::uint64_t state_;
};

[[nodiscard]] std::vector<Point> generate_points(
    std::size_t count,
    std::uint64_t seed) {
    if (count > static_cast<std::size_t>(
                    (std::numeric_limits<Index>::max)())) {
        throw std::overflow_error("node count does not fit Index");
    }

    SplitMix64 random(seed);
    std::vector<Point> result;
    result.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        Point point;
        for (int d = 0; d < Dimension; ++d) {
            point[d] = random.uniform01();
        }
        result.push_back(point);
    }
    return result;
}

void write_points_csv(
    const std::string& path,
    const std::vector<Point>& points) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open points output: " + path);
    }

    output << std::setprecision(17);
    for (const Point& point : points) {
        for (int d = 0; d < Dimension; ++d) {
            if (d != 0) {
                output << ',';
            }
            output << point[d];
        }
        output << '\n';
    }
}

[[nodiscard]] std::vector<Point> read_points_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open points input: " + path);
    }

    std::vector<Point> result;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        std::stringstream row(line);
        Point point;
        std::string field;
        for (int d = 0; d < Dimension; ++d) {
            if (!std::getline(row, field, ',')) {
                throw std::runtime_error(
                    "points CSV has fewer than 5 columns on line " +
                    std::to_string(line_number));
            }
            point[d] = std::stod(field);
        }
        if (std::getline(row, field, ',')) {
            throw std::runtime_error(
                "points CSV has more than 5 columns on line " +
                std::to_string(line_number));
        }
        result.push_back(point);
    }

    return result;
}

[[nodiscard]] Boundary unit_cube_boundary() {
    Point dimensions = Point::Ones();
    Point offset = Point::Zero();
    return Boundary::cuboid(
        dimensions,
        offset,
        std::vector<Index>{});
}

[[nodiscard]] std::size_t persistent_hash_capacity(std::size_t node_count) {
    return next_power_of_two(
        std::max<std::size_t>(std::size_t{1} << 18U, node_count * 512U));
}

[[nodiscard]] std::shared_ptr<Database> make_database(
    std::size_t node_count) {
    // 5D iid meshes have a large number of finite vertices. Give the persistent
    // signature table enough initial room that rehash timing does not dominate
    // one language/run by accident. The chosen capacity is printed below.
    const std::size_t hash_capacity = persistent_hash_capacity(node_count);

    constexpr std::size_t storage_block_units = std::size_t{1} << 20U;
    return std::make_shared<Database>(
        storage_block_units,
        DatabaseParameters{highvoronoi::DirectHash{hash_capacity}});
}

[[nodiscard]] Mesh make_mesh(
    const std::vector<Point>& all_points,
    std::size_t node_count,
    bool bounded) {
    Nodes nodes(static_cast<Index>(node_count));
    for (Index i = 0; i < static_cast<Index>(node_count); ++i) {
        nodes.set(i, all_points[static_cast<std::size_t>(i)]);
    }

    auto database = make_database(node_count);
    if (bounded) {
        return Mesh(std::move(nodes), unit_cube_boundary(), std::move(database));
    }
    return Mesh(std::move(nodes), std::move(database));
}

[[nodiscard]] std::size_t count_infinite_edges(const Mesh& mesh) {
    std::size_t count = 0;
    for ([[maybe_unused]] const auto& edge : mesh.infinite_edges()) {
        ++count;
    }
    return count;
}

template <class RaycastMethod, class SearchKeyword>
[[nodiscard]] TimingResult run_once(
    const std::vector<Point>& points,
    std::size_t node_count,
    bool bounded,
    std::size_t repeat,
    bool verbose,
    const SearchKeyword& search_keyword,
    std::string_view search_name,
    std::string_view raycast_name,
    const highvoronoi::CombinedRaycastOptions& combined_options = {}) {
    const auto total_begin = Clock::now();

    const auto mesh_begin = total_begin;
    Mesh mesh = make_mesh(points, node_count, bounded);
    const auto mesh_end = Clock::now();

    const auto search_begin = mesh_end;
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        search_keyword);
    const auto search_end = Clock::now();

    const auto algorithm_begin = search_end;
    highvoronoi::RaycastParameters<RaycastMethod, Scalar> ray_parameters;
    auto raycaster = [&]() {
        if constexpr (std::is_same_v<RaycastMethod, highvoronoi::CombinedRaycast>) {
            return highvoronoi::make_raycaster(
                tree, ray_parameters, combined_options);
        } else {
            return highvoronoi::make_raycaster(tree, ray_parameters);
        }
    }();

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        DatabaseParameters,
        EdgeParameters>;

    // Cell-local queues/edge tables are recycled between cells. The values are
    // deliberately fixed across every benchmark case.
    constexpr std::size_t queue_capacity = 16384;
    constexpr std::size_t edge_capacity = 32768;

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        DatabaseParameters{highvoronoi::DirectHash{queue_capacity}},
        EdgeParameters{highvoronoi::DirectHash{edge_capacity}});
    const auto algorithm_end = Clock::now();

    const auto compute_begin = algorithm_end;
    compute.compute(verbose);
    const auto compute_end = Clock::now();

    TimingResult result;
    result.nodes = node_count;
    result.domain = bounded ? "bounded" : "open";
    result.repeat = repeat;
    result.search = std::string(search_name);
    result.raycast = std::string(raycast_name);
    result.mesh_setup_seconds = seconds_between(mesh_begin, mesh_end);
    result.search_setup_seconds = seconds_between(search_begin, search_end);
    result.algorithm_setup_seconds = seconds_between(algorithm_begin, algorithm_end);
    result.compute_seconds = seconds_between(compute_begin, compute_end);
    result.total_seconds = seconds_between(total_begin, compute_end);
    result.finite_vertices = compute.new_vertex_count();
    result.infinite_edges = count_infinite_edges(mesh);
    return result;
}

[[nodiscard]] double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0U) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

[[nodiscard]] double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

[[nodiscard]] double standard_deviation(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double m = mean(values);
    double sum = 0.0;
    for (const double value : values) {
        const double delta = value - m;
        sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

void print_result(const TimingResult& result) {
    std::cout << std::fixed << std::setprecision(6)
              << "RESULT"
              << " nodes=" << result.nodes
              << " domain=" << result.domain
              << " repeat=" << result.repeat
              << " search=" << result.search
              << " raycast=" << result.raycast
              << " mesh_setup_s=" << result.mesh_setup_seconds
              << " search_setup_s=" << result.search_setup_seconds
              << " algorithm_setup_s=" << result.algorithm_setup_seconds
              << " compute_s=" << result.compute_seconds
              << " total_s=" << result.total_seconds
              << " finite_vertices=" << result.finite_vertices
              << " infinite_edges=" << result.infinite_edges
              << '\n';
}

void print_summary(
    const std::vector<TimingResult>& results,
    std::size_t node_count,
    std::string_view domain,
    std::string_view raycast) {
    std::vector<double> times;
    std::vector<double> total_times;
    for (const auto& result : results) {
        if (result.nodes == node_count &&
            result.domain == domain &&
            result.raycast == raycast) {
            times.push_back(result.compute_seconds);
            total_times.push_back(result.total_seconds);
        }
    }
    if (times.empty()) {
        return;
    }

    const auto [minimum, maximum] = std::minmax_element(times.begin(), times.end());
    std::cout << std::fixed << std::setprecision(6)
              << "SUMMARY"
              << " nodes=" << node_count
              << " domain=" << domain
              << " raycast=" << raycast
              << " repeats=" << times.size()
              << " compute_min_s=" << *minimum
              << " compute_median_s=" << median(times)
              << " compute_mean_s=" << mean(times)
              << " compute_stddev_s=" << standard_deviation(times)
              << " compute_max_s=" << *maximum
              << " total_median_s=" << median(total_times)
              << '\n';
}

void write_results_csv(
    const std::string& path,
    const std::vector<TimingResult>& results,
    std::uint64_t seed,
    std::string_view combined_fallback) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open results output: " + path);
    }

    output << "dimension,nodes,domain,repeat,search,raycast,combined_fallback,seed,mesh_setup_s,search_setup_s,"
              "algorithm_setup_s,compute_s,total_s,finite_vertices,infinite_edges\n";
    output << std::setprecision(17);
    for (const auto& result : results) {
        output << Dimension << ','
               << result.nodes << ','
               << result.domain << ','
               << result.repeat << ','
               << result.search << ','
               << result.raycast << ','
               << combined_fallback << ','
               << seed << ','
               << result.mesh_setup_seconds << ','
               << result.search_setup_seconds << ','
               << result.algorithm_setup_seconds << ','
               << result.compute_seconds << ','
               << result.total_seconds << ','
               << result.finite_vertices << ','
               << result.infinite_edges << '\n';
    }
}

[[nodiscard]] std::vector<std::size_t> parse_node_counts(const std::string& text) {
    std::vector<std::size_t> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const std::size_t value = static_cast<std::size_t>(std::stoull(item));
        if (value <= static_cast<std::size_t>(Dimension)) {
            throw std::invalid_argument("node count must exceed dimension");
        }
        result.push_back(value);
    }
    if (result.empty()) {
        throw std::invalid_argument("--nodes requires at least one count");
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

void usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "Options:\n"
        << "  --nodes N[,N...]     Node counts (default: 1000,2000)\n"
        << "  --repeats N          Fresh runs per case (default: 3)\n"
        << "  --domain MODE        bounded | open | both (default: both)\n"
        << "  --search MODE        copy | provider (default: copy)\n"
        << "                       copy = contiguous search-local KD snapshot\n"
        << "                       provider = original zero-copy node provider\n"
        << "  --raycast MODE       classic | inrange | combined | combined-fast | all\n"
        << "                       combined = default Robust, combined-fast = explicit Fast/Lossy\n"
        << "  --combined-fallback MODE  auto | inrange (default: auto)\n"
        << "                       auto = Classic for d-generator WalkRay edges, otherwise InRange\n"
        << "  --seed U64           SplitMix64 seed (default fixed)\n"
        << "  --points-in FILE     Read exact 5D points from CSV\n"
        << "  --points-out FILE    Write generated/read point set to CSV\n"
        << "  --csv FILE           Write raw timing rows to CSV\n"
        << "  --verbose             Enable ComputeVoronoi ProgressMeter\n"
        << "  --help                Show this message\n";
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value after " + arg);
            }
            return argv[++i];
        };

        if (arg == "--nodes") {
            options.node_counts = parse_node_counts(require_value());
        } else if (arg == "--repeats") {
            options.repeats = static_cast<std::size_t>(std::stoull(require_value()));
            if (options.repeats == 0) {
                throw std::invalid_argument("--repeats must be positive");
            }
        } else if (arg == "--domain") {
            options.domain = require_value();
            if (options.domain != "bounded" &&
                options.domain != "open" &&
                options.domain != "both") {
                throw std::invalid_argument(
                    "--domain must be bounded, open, or both");
            }
        } else if (arg == "--search") {
            options.search = require_value();
            if (options.search != "copy" &&
                options.search != "provider") {
                throw std::invalid_argument(
                    "--search must be copy or provider");
            }
        } else if (arg == "--raycast") {
            options.raycast = require_value();
            if (options.raycast != "classic" &&
                options.raycast != "inrange" &&
                options.raycast != "combined" &&
                options.raycast != "combined-fast" &&
                options.raycast != "all") {
                throw std::invalid_argument(
                    "--raycast must be classic, inrange, combined, combined-fast, or all");
            }
        } else if (arg == "--combined-fallback") {
            options.combined_fallback = require_value();
            if (options.combined_fallback != "auto" &&
                options.combined_fallback != "inrange") {
                throw std::invalid_argument(
                    "--combined-fallback must be auto or inrange");
            }
        } else if (arg == "--seed") {
            options.seed = std::stoull(require_value(), nullptr, 0);
        } else if (arg == "--points-in") {
            options.points_in = require_value();
        } else if (arg == "--points-out") {
            options.points_out = require_value();
        } else if (arg == "--csv") {
            options.csv_out = require_value();
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return options;
}

void print_configuration(
    const Options& options,
    std::size_t available_points,
    std::size_t maximum_nodes) {
    std::cout << "HighVoronoi ComputeVoronoi 5D benchmark\n"
              << "dimension:          " << Dimension << '\n'
              << "scalar:             double\n"
              << "index:              uint32_t\n"
              << "raycast:            " << options.raycast << '\n'
              << "combined fallback:  " << options.combined_fallback << '\n'
              << "search:             "
              << (options.search == "copy"
                      ? "CopyKDSearch{8,1}"
                      : "KDSearch{8,1} (provider-backed)")
              << '\n' 
              << "threading:          SingleThread / SingleThread\n"
              << "domain:             " << options.domain << '\n'
              << "repeats:            " << options.repeats << '\n'
              << "seed:               0x" << std::hex << options.seed << std::dec << '\n'
              << "available points:   " << available_points << '\n'
              << "persistent hash:    " << persistent_hash_capacity(maximum_nodes)
              << " (for largest N)\n"
              << "queue capacity:     16384\n"
              << "edge capacity:      32768\n"
              << "progress output:    " << (options.verbose ? "on" : "off") << '\n';
#ifdef NDEBUG
    std::cout << "assertions:          disabled (NDEBUG)\n";
#else
    std::cout << "assertions:          ENABLED -- do not use for final timings\n";
#endif
#if defined(__clang__)
    std::cout << "compiler:            clang " << __clang_version__ << '\n';
#elif defined(__GNUC__)
    std::cout << "compiler:            gcc " << __VERSION__ << '\n';
#else
    std::cout << "compiler:            unknown\n";
#endif
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::size_t maximum_nodes =
            *std::max_element(options.node_counts.begin(), options.node_counts.end());

        std::vector<Point> points = options.points_in.empty()
            ? generate_points(maximum_nodes, options.seed)
            : read_points_csv(options.points_in);

        if (points.size() < maximum_nodes) {
            throw std::runtime_error(
                "point set contains fewer rows than the largest --nodes case");
        }
        if (!options.points_out.empty()) {
            write_points_csv(options.points_out, points);
        }

        print_configuration(options, points.size(), maximum_nodes);

        std::vector<TimingResult> results;
        results.reserve(
            options.node_counts.size() * options.repeats *
            (options.domain == "both" ? 2U : 1U) *
            (options.raycast == "all" ? 4U : 1U));

        enum class RaycastKind { Classic, InRange, Combined, CombinedFast };

        const auto raycast_name = [](RaycastKind kind) -> std::string_view {
            switch (kind) {
            case RaycastKind::Classic:        return "classic";
            case RaycastKind::InRange:        return "inrange";
            case RaycastKind::Combined:       return "combined";
            case RaycastKind::CombinedFast:   return "combined-fast";
            }
            return "unknown";
        };

        const auto selected_raycasts = [&]() {
            std::vector<RaycastKind> variants;
            if (options.raycast == "all" || options.raycast == "classic") {
                variants.push_back(RaycastKind::Classic);
            }
            if (options.raycast == "all" || options.raycast == "inrange") {
                variants.push_back(RaycastKind::InRange);
            }
            if (options.raycast == "all" || options.raycast == "combined") {
                variants.push_back(RaycastKind::Combined);
            }
            if (options.raycast == "all" || options.raycast == "combined-fast") {
                variants.push_back(RaycastKind::CombinedFast);
            }
            return variants;
        }();

        const auto run_one = [&](
            std::size_t node_count,
            bool bounded,
            std::size_t repeat,
            RaycastKind raycast_kind) {
            const std::string_view raycast = raycast_name(raycast_kind);
            std::cout << "RUN nodes=" << node_count
                      << " domain=" << (bounded ? "bounded" : "open")
                      << " repeat=" << repeat << '/' << options.repeats
                      << " raycast=" << raycast
                      << '\n';

            TimingResult result;
            const auto dispatch_search = [&](const auto& search_keyword,
                                             std::string_view search_name) {
                switch (raycast_kind) {
                case RaycastKind::Classic:
                    return run_once<highvoronoi::ClassicRaycast>(
                        points, node_count, bounded, repeat, options.verbose,
                        search_keyword, search_name, raycast);
                case RaycastKind::InRange:
                    return run_once<highvoronoi::InRangeRaycast>(
                        points, node_count, bounded, repeat, options.verbose,
                        search_keyword, search_name, raycast);
                case RaycastKind::Combined:
                case RaycastKind::CombinedFast: {
                    highvoronoi::CombinedRaycastOptions combined_options;
                    if (raycast_kind == RaycastKind::CombinedFast) {
                        combined_options.precision_policy =
                            highvoronoi::CombinedPrecisionPolicy::FastLossy;
                    }
                    combined_options.fallback_method =
                        options.combined_fallback == "inrange"
                            ? highvoronoi::CombinedFallbackMethod::InRangeOnly
                            : highvoronoi::CombinedFallbackMethod::ClassicGeneralInRangeDegenerate;
                    return run_once<highvoronoi::CombinedRaycast>(
                        points, node_count, bounded, repeat, options.verbose,
                        search_keyword, search_name, raycast, combined_options);
                }
                }
                throw std::logic_error("invalid raycast kind");
            };

            if (options.search == "copy") {
                result = dispatch_search(
                    highvoronoi::geometry::CopyKDSearch{8, 1}, "copy");
            } else {
                result = dispatch_search(
                    highvoronoi::geometry::KDSearch{8, 1}, "provider");
            }

            print_result(result);
            results.push_back(std::move(result));
        };

        // Interleave raycast variants inside every (N, domain, repeat) case.
        // For --raycast all, rotate the starting variant per repetition so
        // thermal/frequency drift does not systematically favor one method.
        for (const std::size_t node_count : options.node_counts) {
            for (std::size_t repeat = 1; repeat <= options.repeats; ++repeat) {
                std::vector<RaycastKind> order = selected_raycasts;
                if (options.raycast == "all" && order.size() > 1) {
                    const std::size_t shift = (repeat - 1U) % order.size();
                    std::rotate(
                        order.begin(),
                        order.begin() + static_cast<std::ptrdiff_t>(shift),
                        order.end());
                }

                const auto run_domain = [&](bool bounded) {
                    const std::size_t result_begin = results.size();
                    for (const RaycastKind kind : order) {
                        run_one(node_count, bounded, repeat, kind);
                    }

                    // Different raycast variants are expected to be very close.
                    // Report topology/count differences without failing the timing run.
                    if (options.raycast == "all") {
                        const TimingResult* classic = nullptr;
                        const TimingResult* inrange = nullptr;
                        const TimingResult* combined = nullptr;
                        const TimingResult* combined_fast = nullptr;

                        for (std::size_t i = result_begin; i < results.size(); ++i) {
                            if (results[i].raycast == "classic") {
                                classic = &results[i];
                            } else if (results[i].raycast == "inrange") {
                                inrange = &results[i];
                            } else if (results[i].raycast == "combined") {
                                combined = &results[i];
                            } else if (results[i].raycast == "combined-fast") {
                                combined_fast = &results[i];
                            }
                        }

                        if (!classic || !inrange || !combined || !combined_fast) {
                            throw std::logic_error(
                                "--raycast all did not produce all four variants");
                        }

                        const auto differs_from_classic = [&](const TimingResult& other) {
                            return classic->finite_vertices != other.finite_vertices ||
                                   classic->infinite_edges != other.infinite_edges;
                        };

                        if (differs_from_classic(*inrange) ||
                            differs_from_classic(*combined) ||
                            differs_from_classic(*combined_fast)) {
                            std::cout
                                << "MISMATCH"
                                << " seed=0x" << std::hex << options.seed << std::dec
                                << " nodes=" << node_count
                                << " domain=" << (bounded ? "bounded" : "open")
                                << " repeat=" << repeat
                                << " search=" << options.search
                                << " classic_finite=" << classic->finite_vertices
                                << " classic_infinite=" << classic->infinite_edges
                                << " inrange_finite=" << inrange->finite_vertices
                                << " inrange_infinite=" << inrange->infinite_edges
                                << " combined_finite=" << combined->finite_vertices
                                << " combined_infinite=" << combined->infinite_edges
                                << " fast_finite=" << combined_fast->finite_vertices
                                << " fast_infinite=" << combined_fast->infinite_edges
                                << '\n';

                            std::cout
                                << "REPRO --nodes " << node_count
                                << " --repeats 1"
                                << " --domain " << (bounded ? "bounded" : "open")
                                << " --search " << options.search
                                << " --raycast all"
                                << " --combined-fallback " << options.combined_fallback
                                << " --seed 0x" << std::hex << options.seed << std::dec
                                << '\n';
                        }
                    }
                };

                if (options.domain == "bounded" || options.domain == "both") {
                    run_domain(true);
                }
                if (options.domain == "open" || options.domain == "both") {
                    run_domain(false);
                }
            }
        }

        std::cout << "\n";
        for (const std::size_t node_count : options.node_counts) {
            for (const RaycastKind kind : selected_raycasts) {
                const std::string_view raycast = raycast_name(kind);
                if (options.domain == "bounded" || options.domain == "both") {
                    print_summary(results, node_count, "bounded", raycast);
                }
                if (options.domain == "open" || options.domain == "both") {
                    print_summary(results, node_count, "open", raycast);
                }
            }
        }

        if (!options.csv_out.empty()) {
            write_results_csv(
                options.csv_out, results, options.seed, options.combined_fallback);
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "benchmark error: " << exception.what() << '\n';
        return 1;
    }
}
