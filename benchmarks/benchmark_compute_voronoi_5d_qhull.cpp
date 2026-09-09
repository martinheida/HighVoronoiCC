#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/storage/hvdatabase.hpp>

#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullFacetList.h>

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
inline constexpr std::string_view QhullOptions = "v Qbb Qc Qz";

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::detail::EmptyLock,
    DatabaseParameters,
    Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;

struct Options {
    std::vector<std::size_t> node_counts{1000, 2000};
    std::size_t repeats = 5;
    std::uint64_t seed = 0x485642454e434835ULL; // "HVBENCH5"
    std::string search = "copy"; // copy | provider; HighVoronoi only
    std::string method = "all"; // classic | inrange | combined | combined-fast | qhull | all
    std::string combined_fallback = "auto"; // auto | inrange
    bool verbose = false;
    std::string points_in;
    std::string points_out;
    std::string csv_out;
};

struct TimingResult {
    std::size_t nodes = 0;
    std::size_t repeat = 0;
    std::string method;
    std::string search;
    double setup_seconds = 0.0;
    double compute_seconds = 0.0;
    double total_seconds = 0.0;
    std::size_t finite_vertices = 0;
    std::optional<std::size_t> infinite_edges;
    std::optional<std::size_t> qhull_upper_facets;
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

[[nodiscard]] std::size_t persistent_hash_capacity(std::size_t node_count) {
    return next_power_of_two(
        std::max<std::size_t>(std::size_t{1} << 18U, node_count * 512U));
}

[[nodiscard]] std::shared_ptr<Database> make_database(
    std::size_t node_count) {
    const std::size_t hash_capacity = persistent_hash_capacity(node_count);
    constexpr std::size_t storage_block_units = std::size_t{1} << 20U;
    return std::make_shared<Database>(
        storage_block_units,
        DatabaseParameters{highvoronoi::DirectHash{hash_capacity}});
}

[[nodiscard]] Mesh make_open_mesh(
    const std::vector<Point>& all_points,
    std::size_t node_count) {
    Nodes nodes(static_cast<Index>(node_count));
    for (Index i = 0; i < static_cast<Index>(node_count); ++i) {
        nodes.set(i, all_points[static_cast<std::size_t>(i)]);
    }
    return Mesh(std::move(nodes), make_database(node_count));
}

[[nodiscard]] std::size_t count_infinite_edges(const Mesh& mesh) {
    std::size_t count = 0;
    for ([[maybe_unused]] const auto& edge : mesh.infinite_edges()) {
        ++count;
    }
    return count;
}

template <class RaycastMethod, class SearchKeyword>
[[nodiscard]] TimingResult run_highvoronoi_once(
    const std::vector<Point>& points,
    std::size_t node_count,
    std::size_t repeat,
    bool verbose,
    const SearchKeyword& search_keyword,
    std::string_view search_name,
    std::string_view method_name,
    const highvoronoi::CombinedRaycastOptions& combined_options = {}) {
    const auto total_begin = Clock::now();

    Mesh mesh = make_open_mesh(points, node_count);
    auto tree = highvoronoi::geometry::make_search_tree(mesh, search_keyword);

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

    const auto setup_end = Clock::now();
    const auto compute_begin = setup_end;
    compute.compute(verbose);
    const auto compute_end = Clock::now();

    TimingResult result;
    result.nodes = node_count;
    result.repeat = repeat;
    result.method = std::string(method_name);
    result.search = std::string(search_name);
    result.setup_seconds = seconds_between(total_begin, setup_end);
    result.compute_seconds = seconds_between(compute_begin, compute_end);
    result.total_seconds = seconds_between(total_begin, compute_end);
    result.finite_vertices = compute.new_vertex_count();
    result.infinite_edges = count_infinite_edges(mesh);
    return result;
}

[[nodiscard]] TimingResult run_qhull_once(
    const std::vector<Point>& points,
    std::size_t node_count,
    std::size_t repeat) {
    if (node_count > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
        throw std::overflow_error("Qhull point count does not fit int");
    }

    const auto total_begin = Clock::now();

    // Qhull expects one contiguous row-major coordinate array. This adapter
    // cost is reported as setup_s and deliberately excluded from compute_s.
    std::vector<double> coordinates;
    coordinates.reserve(node_count * static_cast<std::size_t>(Dimension));
    for (std::size_t i = 0; i < node_count; ++i) {
        for (int d = 0; d < Dimension; ++d) {
            coordinates.push_back(points[i][d]);
        }
    }

    orgQhull::Qhull qhull;
    const auto setup_end = Clock::now();

    const auto compute_begin = setup_end;
    qhull.runQhull(
        "HighVoronoiCC benchmark",
        Dimension,
        static_cast<int>(node_count),
        coordinates.data(),
        QhullOptions.data());
    const auto compute_end = Clock::now();

    if (!qhull.isDelaunay()) {
        throw std::runtime_error(
            "Qhull did not enter Delaunay/Voronoi mode");
    }

    std::size_t finite_vertices = 0;
    std::size_t upper_facets = 0;
    const orgQhull::QhullFacetList facets = qhull.facetList();
    for (auto facet = facets.begin(); facet != facets.end(); ++facet) {
        if (facet->isUpperDelaunay()) {
            ++upper_facets;
        } else {
            ++finite_vertices;
        }
    }

    TimingResult result;
    result.nodes = node_count;
    result.repeat = repeat;
    result.method = "qhull";
    result.search = "n/a";
    result.setup_seconds = seconds_between(total_begin, setup_end);
    result.compute_seconds = seconds_between(compute_begin, compute_end);
    result.total_seconds = seconds_between(total_begin, compute_end);
    result.finite_vertices = finite_vertices;
    result.qhull_upper_facets = upper_facets;
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
              << " domain=open"
              << " repeat=" << result.repeat
              << " method=" << result.method
              << " search=" << result.search
              << " setup_s=" << result.setup_seconds
              << " compute_s=" << result.compute_seconds
              << " total_s=" << result.total_seconds
              << " finite_vertices=" << result.finite_vertices;

    if (result.infinite_edges) {
        std::cout << " infinite_edges=" << *result.infinite_edges;
    } else {
        std::cout << " infinite_edges=NA";
    }

    if (result.qhull_upper_facets) {
        std::cout << " qhull_upper_facets=" << *result.qhull_upper_facets;
    }
    std::cout << '\n';
}

void print_summary(
    const std::vector<TimingResult>& results,
    std::size_t node_count,
    std::string_view method) {
    std::vector<double> compute_times;
    std::vector<double> total_times;
    for (const auto& result : results) {
        if (result.nodes == node_count && result.method == method) {
            compute_times.push_back(result.compute_seconds);
            total_times.push_back(result.total_seconds);
        }
    }
    if (compute_times.empty()) {
        return;
    }

    const auto [minimum, maximum] =
        std::minmax_element(compute_times.begin(), compute_times.end());
    std::cout << std::fixed << std::setprecision(6)
              << "SUMMARY"
              << " nodes=" << node_count
              << " domain=open"
              << " method=" << method
              << " repeats=" << compute_times.size()
              << " compute_min_s=" << *minimum
              << " compute_median_s=" << median(compute_times)
              << " compute_mean_s=" << mean(compute_times)
              << " compute_stddev_s=" << standard_deviation(compute_times)
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

    output
        << "dimension,nodes,domain,repeat,method,search,combined_fallback,seed,"
           "setup_s,compute_s,total_s,finite_vertices,infinite_edges,qhull_upper_facets\n";
    output << std::setprecision(17);

    for (const auto& result : results) {
        output << Dimension << ','
               << result.nodes << ','
               << "open" << ','
               << result.repeat << ','
               << result.method << ','
               << result.search << ','
               << combined_fallback << ','
               << seed << ','
               << result.setup_seconds << ','
               << result.compute_seconds << ','
               << result.total_seconds << ','
               << result.finite_vertices << ',';

        if (result.infinite_edges) {
            output << *result.infinite_edges;
        }
        output << ',';
        if (result.qhull_upper_facets) {
            output << *result.qhull_upper_facets;
        }
        output << '\n';
    }
}

[[nodiscard]] std::vector<std::size_t> parse_node_counts(
    const std::string& text) {
    std::vector<std::size_t> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const std::size_t value =
            static_cast<std::size_t>(std::stoull(item));
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
        << "Open-domain 5D HighVoronoi vs Qhull benchmark.\n\n"
        << "Options:\n"
        << "  --nodes N[,N...]     Node counts (default: 1000,2000)\n"
        << "  --repeats N          Fresh runs per case (default: 5)\n"
        << "  --search MODE        copy | provider (default: copy; HighVoronoi only)\n"
        << "  --method MODE        classic | inrange | combined | combined-fast | qhull | all\n"
        << "                       combined = default Robust, combined-fast = Fast/Lossy\n"
        << "  --combined-fallback MODE  auto | inrange (default: auto)\n"
        << "  --seed U64           SplitMix64 seed (default fixed)\n"
        << "  --points-in FILE     Read exact 5D points from CSV\n"
        << "  --points-out FILE    Write generated/read point set to CSV\n"
        << "  --csv FILE           Write raw timing rows to CSV\n"
        << "  --verbose            Enable HighVoronoi ProgressMeter\n"
        << "  --help               Show this message\n";
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
            options.repeats =
                static_cast<std::size_t>(std::stoull(require_value()));
            if (options.repeats == 0) {
                throw std::invalid_argument("--repeats must be positive");
            }
        } else if (arg == "--search") {
            options.search = require_value();
            if (options.search != "copy" && options.search != "provider") {
                throw std::invalid_argument("--search must be copy or provider");
            }
        } else if (arg == "--method") {
            options.method = require_value();
            if (options.method != "classic" &&
                options.method != "inrange" &&
                options.method != "combined" &&
                options.method != "combined-fast" &&
                options.method != "qhull" &&
                options.method != "all") {
                throw std::invalid_argument(
                    "--method must be classic, inrange, combined, "
                    "combined-fast, qhull, or all");
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
    std::cout << "HighVoronoi vs Qhull open-domain 5D benchmark\n"
              << "dimension:          " << Dimension << '\n'
              << "scalar:             double\n"
              << "method:             " << options.method << '\n'
              << "HighVoronoi search: "
              << (options.search == "copy"
                      ? "CopyKDSearch{8,1}"
                      : "KDSearch{8,1} (provider-backed)")
              << '\n'
              << "combined fallback:  " << options.combined_fallback << '\n'
              << "Qhull options:      " << QhullOptions << '\n'
              << "threading:          HighVoronoi SingleThread / SingleThread\n"
              << "domain:             open only\n"
              << "repeats:            " << options.repeats << '\n'
              << "seed:               0x" << std::hex << options.seed
              << std::dec << '\n'
              << "available points:   " << available_points << '\n'
              << "persistent hash:    "
              << persistent_hash_capacity(maximum_nodes)
              << " (for largest HighVoronoi N)\n"
              << "queue capacity:     16384\n"
              << "edge capacity:      32768\n"
              << "progress output:    "
              << (options.verbose ? "on" : "off") << '\n';
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

enum class MethodKind {
    Classic,
    InRange,
    Combined,
    CombinedFast,
    Qhull
};

[[nodiscard]] std::string_view method_name(MethodKind kind) {
    switch (kind) {
    case MethodKind::Classic:      return "classic";
    case MethodKind::InRange:      return "inrange";
    case MethodKind::Combined:     return "combined";
    case MethodKind::CombinedFast: return "combined-fast";
    case MethodKind::Qhull:        return "qhull";
    }
    return "unknown";
}

[[nodiscard]] std::vector<MethodKind> selected_methods(
    const std::string& method) {
    std::vector<MethodKind> result;
    if (method == "all" || method == "classic") {
        result.push_back(MethodKind::Classic);
    }
    if (method == "all" || method == "inrange") {
        result.push_back(MethodKind::InRange);
    }
    if (method == "all" || method == "combined") {
        result.push_back(MethodKind::Combined);
    }
    if (method == "all" || method == "combined-fast") {
        result.push_back(MethodKind::CombinedFast);
    }
    if (method == "all" || method == "qhull") {
        result.push_back(MethodKind::Qhull);
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::size_t maximum_nodes =
            *std::max_element(
                options.node_counts.begin(), options.node_counts.end());

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

        const std::vector<MethodKind> methods = selected_methods(options.method);
        std::vector<TimingResult> results;
        results.reserve(
            options.node_counts.size() * options.repeats * methods.size());

        const auto run_one = [&](std::size_t node_count,
                                 std::size_t repeat,
                                 MethodKind kind) {
            const std::string_view name = method_name(kind);
            std::cout << "RUN nodes=" << node_count
                      << " domain=open"
                      << " repeat=" << repeat << '/' << options.repeats
                      << " method=" << name << '\n';

            TimingResult result;
            if (kind == MethodKind::Qhull) {
                result = run_qhull_once(points, node_count, repeat);
            } else {
                const auto dispatch_search = [&](const auto& search_keyword,
                                                 std::string_view search_name) {
                    switch (kind) {
                    case MethodKind::Classic:
                        return run_highvoronoi_once<highvoronoi::ClassicRaycast>(
                            points, node_count, repeat, options.verbose,
                            search_keyword, search_name, name);
                    case MethodKind::InRange:
                        return run_highvoronoi_once<highvoronoi::InRangeRaycast>(
                            points, node_count, repeat, options.verbose,
                            search_keyword, search_name, name);
                    case MethodKind::Combined:
                    case MethodKind::CombinedFast: {
                        highvoronoi::CombinedRaycastOptions combined_options;
                        if (kind == MethodKind::CombinedFast) {
                            combined_options.precision_policy =
                                highvoronoi::CombinedPrecisionPolicy::FastLossy;
                        }
                        combined_options.fallback_method =
                            options.combined_fallback == "inrange"
                                ? highvoronoi::CombinedFallbackMethod::InRangeOnly
                                : highvoronoi::CombinedFallbackMethod::ClassicGeneralInRangeDegenerate;
                        return run_highvoronoi_once<highvoronoi::CombinedRaycast>(
                            points, node_count, repeat, options.verbose,
                            search_keyword, search_name, name, combined_options);
                    }
                    case MethodKind::Qhull:
                        break;
                    }
                    throw std::logic_error("invalid method kind");
                };

                if (options.search == "copy") {
                    result = dispatch_search(
                        highvoronoi::geometry::CopyKDSearch{8, 1}, "copy");
                } else {
                    result = dispatch_search(
                        highvoronoi::geometry::KDSearch{8, 1}, "provider");
                }
            }

            print_result(result);
            results.push_back(std::move(result));
        };

        // Interleave all implementations and rotate the first one on each
        // repeat to reduce systematic thermal/frequency bias.
        for (const std::size_t node_count : options.node_counts) {
            for (std::size_t repeat = 1; repeat <= options.repeats; ++repeat) {
                std::vector<MethodKind> order = methods;
                if (order.size() > 1) {
                    const std::size_t shift = (repeat - 1U) % order.size();
                    std::rotate(
                        order.begin(),
                        order.begin() + static_cast<std::ptrdiff_t>(shift),
                        order.end());
                }

                const std::size_t result_begin = results.size();
                for (const MethodKind kind : order) {
                    run_one(node_count, repeat, kind);
                }

                if (options.method == "all") {
                    const TimingResult* classic = nullptr;
                    const TimingResult* inrange = nullptr;
                    const TimingResult* combined = nullptr;
                    const TimingResult* combined_fast = nullptr;
                    const TimingResult* qhull = nullptr;

                    for (std::size_t i = result_begin; i < results.size(); ++i) {
                        const TimingResult& r = results[i];
                        if (r.method == "classic") {
                            classic = &r;
                        } else if (r.method == "inrange") {
                            inrange = &r;
                        } else if (r.method == "combined") {
                            combined = &r;
                        } else if (r.method == "combined-fast") {
                            combined_fast = &r;
                        } else if (r.method == "qhull") {
                            qhull = &r;
                        }
                    }

                    if (!classic || !inrange || !combined ||
                        !combined_fast || !qhull) {
                        throw std::logic_error(
                            "--method all did not produce all five variants");
                    }

                    if (classic->finite_vertices != qhull->finite_vertices ||
                        inrange->finite_vertices != qhull->finite_vertices ||
                        combined->finite_vertices != qhull->finite_vertices) {
                        std::cout
                            << "REFERENCE_MISMATCH"
                            << " seed=0x" << std::hex << options.seed << std::dec
                            << " nodes=" << node_count
                            << " repeat=" << repeat
                            << " classic_finite=" << classic->finite_vertices
                            << " inrange_finite=" << inrange->finite_vertices
                            << " combined_finite=" << combined->finite_vertices
                            << " qhull_finite=" << qhull->finite_vertices
                            << '\n';
                    }

                    if (combined_fast->finite_vertices != combined->finite_vertices) {
                        std::cout
                            << "FAST_DIFFERENCE"
                            << " seed=0x" << std::hex << options.seed << std::dec
                            << " nodes=" << node_count
                            << " repeat=" << repeat
                            << " combined_finite=" << combined->finite_vertices
                            << " combined_fast_finite="
                            << combined_fast->finite_vertices
                            << '\n';
                    }
                }
            }
        }

        std::cout << '\n';
        for (const std::size_t node_count : options.node_counts) {
            for (const MethodKind kind : methods) {
                print_summary(results, node_count, method_name(kind));
            }
        }

        if (!options.csv_out.empty()) {
            write_results_csv(
                options.csv_out,
                results,
                options.seed,
                options.combined_fallback);
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "benchmark error: " << exception.what() << '\n';
        return 1;
    }
}
