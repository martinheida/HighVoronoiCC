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
inline constexpr std::size_t HashShards = 16;

using HashGenerator = highvoronoi::FNV64_128HashGenerator;

using DatabaseParameters = highvoronoi::DataBaseParams<
    Scalar,
    Index,
    HashGenerator,
    highvoronoi::StaticHash<HashShards>>;

using QueueParameters = highvoronoi::DataBaseParams<
    Scalar,
    Index,
    HashGenerator,
    highvoronoi::DirectHash>;

using EdgeParameters = highvoronoi::EdgeBufferParams<
    HashGenerator,
    highvoronoi::StaticHash<HashShards>>;

// Keep the same thread-safe mesh/storage configuration for 1, 2, and 4 worker
// runs. This makes the speedup comparison isolate CastThreading instead of also
// changing database locking/hash layout.
using Database = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters,
    Dimension>;

using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;

struct Options {
    std::vector<std::size_t> node_counts{1000, 2000};
    std::vector<std::size_t> thread_counts{1, 2, 4};
    std::size_t repeats = 5;
    std::uint64_t seed = 0x485642454e434835ULL; // "HVBENCH5"
    std::string domain = "both";                 // bounded | open | both
    std::string search = "copy";                 // copy | provider
    std::string raycast = "combined";            // combined | inrange | both
    std::string parallel_axis = "both";           // cast | mesh | both
    std::string combined_mode = "robust";        // robust | fast
    std::string combined_fallback = "auto";      // auto | inrange
    bool verbose = false;
    std::string csv_out;
};

struct TimingResult {
    std::size_t nodes = 0;
    std::string domain;
    std::size_t repeat = 0;
    std::string search;
    std::string raycast;
    std::string parallel_mode; // serial | cast | mesh
    std::size_t mesh_threads = 1;
    std::size_t cast_threads = 1;
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

[[nodiscard]] std::size_t per_shard_capacity(std::size_t total_capacity) {
    const std::size_t rounded =
        (total_capacity + HashShards - 1) / HashShards;
    return next_power_of_two(std::max<std::size_t>(8, rounded));
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

[[nodiscard]] Boundary unit_cube_boundary() {
    Point dimensions = Point::Ones();
    Point offset = Point::Zero();
    return Boundary::cuboid(dimensions, offset, std::vector<Index>{});
}

[[nodiscard]] std::size_t persistent_hash_capacity(std::size_t node_count) {
    return next_power_of_two(
        std::max<std::size_t>(std::size_t{1} << 18U, node_count * 512U));
}

[[nodiscard]] std::shared_ptr<Database> make_database(
    std::size_t node_count) {

    const std::size_t total_capacity =
        persistent_hash_capacity(node_count);
    const std::size_t shard_capacity =
        per_shard_capacity(total_capacity);

    constexpr std::size_t storage_block_units = std::size_t{1} << 20U;

    return std::make_shared<Database>(
        storage_block_units,
        DatabaseParameters{
            highvoronoi::StaticHash<HashShards>{shard_capacity}});
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
        return Mesh(
            std::move(nodes),
            unit_cube_boundary(),
            std::move(database));
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

template <
    class RaycastMethod,
    class MeshThreading,
    class CastThreading,
    class SearchKeyword>
[[nodiscard]] TimingResult run_once(
    const std::vector<Point>& points,
    std::size_t node_count,
    bool bounded,
    std::size_t repeat,
    std::string_view parallel_mode,
    std::size_t mesh_threads,
    std::size_t cast_threads,
    bool verbose,
    const SearchKeyword& search_keyword,
    std::string_view search_name,
    std::string_view raycast_name,
    MeshThreading mesh_threading,
    CastThreading cast_threading,
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
        if constexpr (
            std::is_same_v<RaycastMethod, highvoronoi::CombinedRaycast>) {
            return highvoronoi::make_raycaster(
                tree,
                ray_parameters,
                combined_options);
        } else {
            return highvoronoi::make_raycaster(
                tree,
                ray_parameters);
        }
    }();

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        MeshThreading,
        CastThreading,
        QueueParameters,
        EdgeParameters>;

    // Preserve approximately the same total initial hash capacities as the
    // serial 5D benchmark. EdgeHash is sharded because its lock is hit by all
    // workers; the vertex queue remains DirectHash because the queue itself has
    // an outer synchronization layer.
    constexpr std::size_t queue_capacity = 16384;
    constexpr std::size_t edge_total_capacity = 32768;
    constexpr std::size_t edge_shard_capacity =
        edge_total_capacity / HashShards;

    Compute compute(
        mesh,
        raycaster,
        std::move(mesh_threading),
        std::move(cast_threading),
        std::nullopt,
        QueueParameters{
            highvoronoi::DirectHash{queue_capacity}},
        EdgeParameters{
            highvoronoi::StaticHash<HashShards>{
                edge_shard_capacity}});

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
    result.parallel_mode = std::string(parallel_mode);
    result.mesh_threads = mesh_threads;
    result.cast_threads = cast_threads;
    result.mesh_setup_seconds =
        seconds_between(mesh_begin, mesh_end);
    result.search_setup_seconds =
        seconds_between(search_begin, search_end);
    result.algorithm_setup_seconds =
        seconds_between(algorithm_begin, algorithm_end);
    result.compute_seconds =
        seconds_between(compute_begin, compute_end);
    result.total_seconds =
        seconds_between(total_begin, compute_end);
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
    return Scalar{0.5} * (values[middle - 1] + values[middle]);
}

[[nodiscard]] double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

[[nodiscard]] double standard_deviation(
    const std::vector<double>& values) {

    if (values.size() < 2) {
        return 0.0;
    }

    const double m = mean(values);
    double sum = 0.0;
    for (const double value : values) {
        const double delta = value - m;
        sum += delta * delta;
    }

    return std::sqrt(
        sum / static_cast<double>(values.size()));
}

struct ParallelConfig {
    std::string mode; // serial | cast | mesh
    std::size_t mesh_threads = 1;
    std::size_t cast_threads = 1;

    [[nodiscard]] std::size_t active_threads() const noexcept {
        return mesh_threads * cast_threads;
    }
};

[[nodiscard]] bool same_config(
    const TimingResult& result,
    const ParallelConfig& config) {
    return result.parallel_mode == config.mode &&
           result.mesh_threads == config.mesh_threads &&
           result.cast_threads == config.cast_threads;
}

[[nodiscard]] std::vector<double> compute_values(
    const std::vector<TimingResult>& results,
    std::size_t nodes,
    std::string_view domain,
    std::string_view raycast,
    const ParallelConfig& config) {

    std::vector<double> values;
    for (const auto& result : results) {
        if (result.nodes == nodes &&
            result.domain == domain &&
            result.raycast == raycast &&
            same_config(result, config)) {
            values.push_back(result.compute_seconds);
        }
    }
    return values;
}

[[nodiscard]] double serial_median_compute(
    const std::vector<TimingResult>& results,
    std::size_t nodes,
    std::string_view domain,
    std::string_view raycast) {

    return median(compute_values(
        results,
        nodes,
        domain,
        raycast,
        ParallelConfig{"serial", 1, 1}));
}

void print_result(const TimingResult& result) {
    std::cout << std::fixed << std::setprecision(6)
              << "RESULT nodes=" << result.nodes
              << " domain=" << result.domain
              << " repeat=" << result.repeat
              << " search=" << result.search
              << " raycast=" << result.raycast
              << " parallel_mode=" << result.parallel_mode
              << " mesh_threads=" << result.mesh_threads
              << " cast_threads=" << result.cast_threads
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
    std::size_t nodes,
    std::string_view domain,
    std::string_view raycast,
    const ParallelConfig& config) {

    const std::vector<double> values = compute_values(
        results, nodes, domain, raycast, config);

    if (values.empty()) {
        return;
    }

    const double med = median(values);
    const double serial_median =
        serial_median_compute(results, nodes, domain, raycast);
    const double speedup =
        serial_median > 0.0 ? serial_median / med : 0.0;
    const double efficiency =
        config.active_threads() > 0
            ? speedup / static_cast<double>(config.active_threads())
            : 0.0;

    std::vector<double> total_values;
    for (const auto& result : results) {
        if (result.nodes == nodes &&
            result.domain == domain &&
            result.raycast == raycast &&
            same_config(result, config)) {
            total_values.push_back(result.total_seconds);
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "SUMMARY nodes=" << nodes
              << " domain=" << domain
              << " raycast=" << raycast
              << " parallel_mode=" << config.mode
              << " mesh_threads=" << config.mesh_threads
              << " cast_threads=" << config.cast_threads
              << " repeats=" << values.size()
              << " compute_min_s="
              << *std::min_element(values.begin(), values.end())
              << " compute_median_s=" << med
              << " compute_mean_s=" << mean(values)
              << " compute_stddev_s=" << standard_deviation(values)
              << " compute_max_s="
              << *std::max_element(values.begin(), values.end())
              << " total_median_s=" << median(total_values)
              << " speedup_vs_serial=" << speedup
              << " efficiency=" << efficiency
              << '\n';
}

void write_csv(
    const std::string& path,
    const std::vector<TimingResult>& results,
    std::uint64_t seed,
    std::string_view combined_mode,
    std::string_view combined_fallback) {

    if (path.empty()) {
        return;
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open CSV output: " + path);
    }

    output
        << "dimension,nodes,domain,repeat,search,raycast,parallel_mode,"
        << "mesh_threads,cast_threads,combined_mode,combined_fallback,seed,"
        << "mesh_setup_s,search_setup_s,algorithm_setup_s,compute_s,total_s,"
        << "finite_vertices,infinite_edges\n";

    output << std::setprecision(17);

    for (const auto& result : results) {
        output
            << Dimension << ','
            << result.nodes << ','
            << result.domain << ','
            << result.repeat << ','
            << result.search << ','
            << result.raycast << ','
            << result.parallel_mode << ','
            << result.mesh_threads << ','
            << result.cast_threads << ','
            << combined_mode << ','
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

[[nodiscard]] std::vector<std::size_t> parse_size_list(
    const std::string& text,
    std::string_view option_name) {

    std::vector<std::size_t> values;
    std::stringstream stream(text);
    std::string field;

    while (std::getline(stream, field, ',')) {
        if (field.empty()) {
            throw std::invalid_argument(
                std::string(option_name) + " contains an empty item");
        }
        const std::size_t value =
            static_cast<std::size_t>(std::stoull(field));
        if (value == 0) {
            throw std::invalid_argument(
                std::string(option_name) + " values must be positive");
        }
        values.push_back(value);
    }

    if (values.empty()) {
        throw std::invalid_argument(
            std::string(option_name) + " must not be empty");
    }

    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --nodes LIST              comma-separated node counts (default 1000,2000)\n"
        << "  --threads LIST            thread counts (default 1,2,4; must include 1)\n"
        << "  --parallel-axis MODE      cast | mesh | both (default both)\n"
        << "  --repeats N               repetitions per case (default 5)\n"
        << "  --domain MODE             bounded | open | both\n"
        << "  --search MODE             copy | provider\n"
        << "  --raycast MODE            combined | inrange | both\n"
        << "  --combined-mode MODE      robust | fast\n"
        << "  --combined-fallback MODE  auto | inrange\n"
        << "  --seed VALUE              decimal or 0x... seed\n"
        << "  --csv PATH                write raw results as CSV\n"
        << "  --verbose                 enable ComputeVoronoi progress output\n"
        << "  --help                    show this help\n";
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        const auto require_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    "missing value after " + arg);
            }
            return argv[++i];
        };

        if (arg == "--nodes") {
            options.node_counts =
                parse_size_list(require_value(), "--nodes");
        } else if (arg == "--threads") {
            options.thread_counts =
                parse_size_list(require_value(), "--threads");
        } else if (arg == "--parallel-axis") {
            options.parallel_axis = require_value();
            if (options.parallel_axis != "cast" &&
                options.parallel_axis != "mesh" &&
                options.parallel_axis != "both") {
                throw std::invalid_argument(
                    "--parallel-axis must be cast, mesh, or both");
            }
        } else if (arg == "--repeats") {
            options.repeats =
                static_cast<std::size_t>(
                    std::stoull(require_value()));
            if (options.repeats == 0) {
                throw std::invalid_argument(
                    "--repeats must be positive");
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
            if (options.raycast != "combined" &&
                options.raycast != "inrange" &&
                options.raycast != "both") {
                throw std::invalid_argument(
                    "--raycast must be combined, inrange, or both");
            }
        } else if (arg == "--combined-mode") {
            options.combined_mode = require_value();
            if (options.combined_mode != "robust" &&
                options.combined_mode != "fast") {
                throw std::invalid_argument(
                    "--combined-mode must be robust or fast");
            }
        } else if (arg == "--combined-fallback") {
            options.combined_fallback = require_value();
            if (options.combined_fallback != "auto" &&
                options.combined_fallback != "inrange") {
                throw std::invalid_argument(
                    "--combined-fallback must be auto or inrange");
            }
        } else if (arg == "--seed") {
            const std::string value = require_value();
            options.seed = std::stoull(value, nullptr, 0);
        } else if (arg == "--csv") {
            options.csv_out = require_value();
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(
                "unknown option: " + arg);
        }
    }

    if (std::find(
            options.thread_counts.begin(),
            options.thread_counts.end(),
            std::size_t{1}) == options.thread_counts.end()) {
        throw std::invalid_argument(
            "--threads must include 1 so the serial baseline exists");
    }

    return options;
}

[[nodiscard]] std::vector<ParallelConfig> make_parallel_configs(
    const Options& options) {

    std::vector<ParallelConfig> configs;
    configs.push_back(ParallelConfig{"serial", 1, 1});

    for (const std::size_t threads : options.thread_counts) {
        if (threads == 1) {
            continue;
        }

        if (options.parallel_axis == "cast" ||
            options.parallel_axis == "both") {
            configs.push_back(
                ParallelConfig{"cast", 1, threads});
        }

        if (options.parallel_axis == "mesh" ||
            options.parallel_axis == "both") {
            configs.push_back(
                ParallelConfig{"mesh", threads, 1});
        }
    }

    return configs;
}

[[nodiscard]] bool topology_counts_match_serial(
    const std::vector<TimingResult>& results,
    std::size_t nodes,
    std::string_view domain,
    std::string_view raycast,
    const ParallelConfig& candidate) {

    std::optional<std::pair<std::size_t, std::size_t>> reference;

    for (const auto& result : results) {
        if (result.nodes == nodes &&
            result.domain == domain &&
            result.raycast == raycast &&
            result.parallel_mode == "serial" &&
            result.mesh_threads == 1 &&
            result.cast_threads == 1) {
            reference = {
                result.finite_vertices,
                result.infinite_edges};
            break;
        }
    }

    if (!reference) {
        return false;
    }

    bool found_candidate = false;
    for (const auto& result : results) {
        if (result.nodes == nodes &&
            result.domain == domain &&
            result.raycast == raycast &&
            same_config(result, candidate)) {
            found_candidate = true;
            if (result.finite_vertices != reference->first ||
                result.infinite_edges != reference->second) {
                return false;
            }
        }
    }

    return found_candidate;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        const std::size_t maximum_nodes =
            *std::max_element(
                options.node_counts.begin(),
                options.node_counts.end());

        const std::vector<Point> points =
            generate_points(maximum_nodes, options.seed);

        const std::size_t persistent_total =
            persistent_hash_capacity(maximum_nodes);
        const std::size_t persistent_shard =
            per_shard_capacity(persistent_total);

        const std::vector<ParallelConfig> configs =
            make_parallel_configs(options);

        std::cout
            << "HighVoronoi ComputeVoronoi 5D parallel benchmark\n"
            << "dimension:          " << Dimension << '\n'
            << "scalar:             double\n"
            << "index:              uint32_t\n"
            << "raycast:            " << options.raycast << '\n'
            << "combined mode:      " << options.combined_mode << '\n'
            << "combined fallback:  " << options.combined_fallback << '\n'
            << "search:             "
            << (options.search == "copy"
                    ? "CopyKDSearch{8,1}"
                    : "KDSearch{8,1}")
            << '\n'
            << "parallel axis:      " << options.parallel_axis << '\n'
            << "thread counts:      ";

        for (std::size_t i = 0; i < options.thread_counts.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << options.thread_counts[i];
        }

        std::cout
            << '\n'
            << "cases:              serial(1x1)";
        for (const auto& config : configs) {
            if (config.mode == "serial") {
                continue;
            }
            std::cout << ", "
                      << config.mode
                      << "(mesh=" << config.mesh_threads
                      << ",cast=" << config.cast_threads << ')';
        }

        std::cout
            << '\n'
            << "domain:             " << options.domain << '\n'
            << "repeats:            " << options.repeats << '\n'
            << "seed:               0x"
            << std::hex << options.seed << std::dec << '\n'
            << "available points:   " << points.size() << '\n'
            << "persistent hash:    " << persistent_total
            << " total target, " << HashShards
            << " shards x " << persistent_shard << '\n'
            << "queue capacity:     16384\n"
            << "edge capacity:      32768 total target, "
            << HashShards << " shards x "
            << (32768 / HashShards) << '\n'
            << "progress output:    "
            << (options.verbose ? "on" : "off") << '\n'
#ifdef NDEBUG
            << "assertions:          disabled (NDEBUG)\n"
#else
            << "assertions:          enabled\n"
#endif
#if defined(__clang__)
            << "compiler:            clang " << __clang_version__ << '\n';
#elif defined(__GNUC__)
            << "compiler:            gcc " << __VERSION__ << '\n';
#else
            << "compiler:            unknown\n";
#endif

        std::vector<TimingResult> results;

        const std::vector<std::string> raycasts = [&]() {
            if (options.raycast == "both") {
                return std::vector<std::string>{"combined", "inrange"};
            }
            return std::vector<std::string>{options.raycast};
        }();

        const auto run_configuration = [&](
            std::size_t node_count,
            bool bounded,
            std::size_t repeat,
            std::string_view raycast_name,
            const ParallelConfig& config) {

            std::cout
                << "RUN nodes=" << node_count
                << " domain=" << (bounded ? "bounded" : "open")
                << " repeat=" << repeat << '/' << options.repeats
                << " raycast=" << raycast_name
                << " parallel_mode=" << config.mode
                << " mesh_threads=" << config.mesh_threads
                << " cast_threads=" << config.cast_threads
                << '\n';

            highvoronoi::CombinedRaycastOptions combined_options;
            combined_options.precision_policy =
                options.combined_mode == "robust"
                    ? highvoronoi::CombinedPrecisionPolicy::Robust
                    : highvoronoi::CombinedPrecisionPolicy::FastLossy;
            combined_options.fallback_method =
                options.combined_fallback == "inrange"
                    ? highvoronoi::CombinedFallbackMethod::InRangeOnly
                    : highvoronoi::CombinedFallbackMethod::
                          ClassicGeneralInRangeDegenerate;

            const auto dispatch_search = [&](const auto& search_keyword,
                                             std::string_view search_name) {
                if (raycast_name == "combined") {
                    if (config.mode == "serial") {
                        return run_once<
                            highvoronoi::CombinedRaycast,
                            highvoronoi::SingleThread,
                            highvoronoi::SingleThread>(
                            points, node_count, bounded, repeat,
                            config.mode, 1, 1, options.verbose,
                            search_keyword, search_name, raycast_name,
                            highvoronoi::SingleThread{},
                            highvoronoi::SingleThread{},
                            combined_options);
                    }

                    if (config.mode == "cast") {
                        return run_once<
                            highvoronoi::CombinedRaycast,
                            highvoronoi::SingleThread,
                            highvoronoi::MultiThread>(
                            points, node_count, bounded, repeat,
                            config.mode, 1, config.cast_threads,
                            options.verbose,
                            search_keyword, search_name, raycast_name,
                            highvoronoi::SingleThread{},
                            highvoronoi::MultiThread{config.cast_threads},
                            combined_options);
                    }

                    return run_once<
                        highvoronoi::CombinedRaycast,
                        highvoronoi::MultiThread,
                        highvoronoi::SingleThread>(
                        points, node_count, bounded, repeat,
                        config.mode, config.mesh_threads, 1,
                        options.verbose,
                        search_keyword, search_name, raycast_name,
                        highvoronoi::MultiThread{config.mesh_threads},
                        highvoronoi::SingleThread{},
                        combined_options);
                }

                if (raycast_name == "inrange") {
                    if (config.mode == "serial") {
                        return run_once<
                            highvoronoi::InRangeRaycast,
                            highvoronoi::SingleThread,
                            highvoronoi::SingleThread>(
                            points, node_count, bounded, repeat,
                            config.mode, 1, 1, options.verbose,
                            search_keyword, search_name, raycast_name,
                            highvoronoi::SingleThread{},
                            highvoronoi::SingleThread{});
                    }

                    if (config.mode == "cast") {
                        return run_once<
                            highvoronoi::InRangeRaycast,
                            highvoronoi::SingleThread,
                            highvoronoi::MultiThread>(
                            points, node_count, bounded, repeat,
                            config.mode, 1, config.cast_threads,
                            options.verbose,
                            search_keyword, search_name, raycast_name,
                            highvoronoi::SingleThread{},
                            highvoronoi::MultiThread{config.cast_threads});
                    }

                    return run_once<
                        highvoronoi::InRangeRaycast,
                        highvoronoi::MultiThread,
                        highvoronoi::SingleThread>(
                        points, node_count, bounded, repeat,
                        config.mode, config.mesh_threads, 1,
                        options.verbose,
                        search_keyword, search_name, raycast_name,
                        highvoronoi::MultiThread{config.mesh_threads},
                        highvoronoi::SingleThread{});
                }

                throw std::logic_error("invalid raycast dispatch");
            };

            TimingResult result;
            if (options.search == "copy") {
                result = dispatch_search(
                    highvoronoi::geometry::CopyKDSearch{8, 1},
                    "copy");
            } else {
                result = dispatch_search(
                    highvoronoi::geometry::KDSearch{8, 1},
                    "provider");
            }

            print_result(result);
            results.push_back(std::move(result));
        };

        for (const std::size_t node_count : options.node_counts) {
            for (std::size_t repeat = 1;
                 repeat <= options.repeats;
                 ++repeat) {

                std::vector<ParallelConfig> order = configs;

                if (order.size() > 1) {
                    const std::size_t shift =
                        (repeat - 1U) % order.size();
                    std::rotate(
                        order.begin(),
                        order.begin() +
                            static_cast<std::ptrdiff_t>(shift),
                        order.end());
                }

                for (const std::string& raycast_name : raycasts) {
                    if (options.domain == "bounded" ||
                        options.domain == "both") {
                        for (const auto& config : order) {
                            run_configuration(
                                node_count, true, repeat,
                                raycast_name, config);
                        }
                    }

                    if (options.domain == "open" ||
                        options.domain == "both") {
                        for (const auto& config : order) {
                            run_configuration(
                                node_count, false, repeat,
                                raycast_name, config);
                        }
                    }
                }
            }
        }

        std::cout << '\n';

        for (const std::size_t node_count : options.node_counts) {
            for (const std::string& raycast_name : raycasts) {
                for (const auto& config : configs) {
                    if (options.domain == "bounded" ||
                        options.domain == "both") {
                        print_summary(
                            results, node_count, "bounded",
                            raycast_name, config);
                    }
                    if (options.domain == "open" ||
                        options.domain == "both") {
                        print_summary(
                            results, node_count, "open",
                            raycast_name, config);
                    }
                }
            }
        }

        bool topology_counts_identical = true;

        for (const std::size_t node_count : options.node_counts) {
            for (const std::string& raycast_name : raycasts) {
                for (const auto& config : configs) {
                    if (config.mode == "serial") {
                        continue;
                    }

                    for (const std::string_view domain :
                         {std::string_view{"bounded"},
                          std::string_view{"open"}}) {

                        const bool selected =
                            options.domain == "both" ||
                            options.domain == domain;
                        if (!selected) {
                            continue;
                        }

                        const bool identical =
                            topology_counts_match_serial(
                                results,
                                node_count,
                                domain,
                                raycast_name,
                                config);

                        if (!identical) {
                            topology_counts_identical = false;
                            std::cerr
                                << "TOPOLOGY_COUNT_DIFFERENCE"
                                << " nodes=" << node_count
                                << " domain=" << domain
                                << " raycast=" << raycast_name
                                << " parallel_mode=" << config.mode
                                << " mesh_threads=" << config.mesh_threads
                                << " cast_threads=" << config.cast_threads
                                << '\n';
                        }
                    }
                }
            }
        }

        write_csv(
            options.csv_out,
            results,
            options.seed,
            options.combined_mode,
            options.combined_fallback);

        if (!topology_counts_identical) {
            std::cerr
                << "NOTE: finite/infinite counts differ from the serial baseline "
                   "for at least one parallel configuration. This is reported "
                   "for diagnosis only and does not fail the performance benchmark.\n";
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
