#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/storage/hvdatabase.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
#include <thread>
#include <utility>
#include <vector>

#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;
using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 5;
inline constexpr std::int64_t DefaultRationalScale = 100000000000000LL; // 1e14

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters,
    Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;
using Integral = highvoronoi::VoronoiIntegral<Mesh, double, double>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using RationalVertex = std::array<std::int64_t, Dimension>;

struct Options {
    std::size_t nodes = 1000;
    std::size_t repeats = 5;
    std::uint64_t seed = 0x485642454e434835ULL; // "HVBENCH5"
    std::int64_t rational_scale = DefaultRationalScale;
    std::string lrs_binary = "lrs";
    bool verbose_mesh = false;
    bool lrs_trace = false;
    std::string points_in;
    std::string points_out;
    std::string csv_out;
};

struct GeometryResult {
    std::vector<double> volumes;
    std::vector<std::vector<Index>> neighbours;
    std::vector<std::vector<double>> areas;
    double volume_sum = 0.0;
    double area_entry_sum = 0.0;
    std::size_t area_entries = 0;
};

struct LrsCellFile {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::size_t vertex_count = 0;
    bool valid = false;
};

struct LrsConversionResult {
    std::vector<LrsCellFile> cells;
    std::size_t invalid_cells = 0;
    std::size_t rationalized_vertices = 0;
};

struct ProcessResult {
    double elapsed_s = 0.0;
    int exit_code = -1;
    bool exited = false;
    bool timed_out = false;
};

struct LrsRunResult {
    double elapsed_s = 0.0;
    std::vector<double> volumes;
    std::size_t attempted = 0;
    std::size_t completed = 0;
    std::size_t failed = 0;
    std::size_t timed_out = 0;
    std::size_t invalid_conversion = 0;
    std::size_t unattempted_due_budget = 0;
    double partial_volume_sum = 0.0;
};

struct PartialVolumeError {
    std::size_t compared = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double max_rel = 0.0;
};

struct InterfaceComparison {
    bool layout_equal = true;
    double max_abs_area_error = 0.0;
};

struct RepeatResult {
    std::size_t repeat = 0;

    // Four primary times for this benchmark.
    double polygon_s = 0.0;
    double fast_polygon_s = 0.0;
    double lrslib_conversion_s = 0.0;
    double lrslib_volume_s = 0.0;

    double polygon_volume_sum = 0.0;
    double fast_polygon_volume_sum = 0.0;
    std::size_t polygon_area_entries = 0;
    std::size_t fast_polygon_area_entries = 0;
    double polygon_area_entry_sum = 0.0;
    double fast_polygon_area_entry_sum = 0.0;
    bool polygon_fast_interface_layout_equal = false;
    double polygon_fast_max_abs_area_error = 0.0;

    std::size_t lrs_attempted = 0;
    std::size_t lrs_completed = 0;
    std::size_t lrs_failed = 0;
    std::size_t lrs_timed_out = 0;
    std::size_t lrs_invalid_conversion = 0;
    std::size_t lrs_unattempted_due_budget = 0;
    double lrs_partial_volume_sum = 0.0;
    std::size_t lrs_compared_cells = 0;
    double lrs_max_abs_cell_error_vs_polygon = 0.0;
    double lrs_mean_abs_cell_error_vs_polygon = 0.0;
    double lrs_max_rel_cell_error_vs_polygon = 0.0;
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

class TempDirectory final {
public:
    TempDirectory() {
        std::string pattern = "/tmp/highvoronoi_lrslib_XXXXXX";
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        char* created = ::mkdtemp(buffer.data());
        if (!created) {
            throw std::runtime_error("mkdtemp failed");
        }
        path_ = created;
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
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
    return Boundary::cuboid(
        Point::Ones(),
        Point::Zero(),
        std::vector<Index>{});
}

[[nodiscard]] std::size_t persistent_hash_capacity(std::size_t node_count) {
    return next_power_of_two(
        std::max<std::size_t>(std::size_t{1} << 18U, node_count * 512U));
}

[[nodiscard]] std::shared_ptr<Database> make_database(
    std::size_t node_count) {
    constexpr std::size_t storage_block_units = std::size_t{1} << 20U;
    return std::make_shared<Database>(
        storage_block_units,
        DatabaseParameters{
            highvoronoi::DirectHash{persistent_hash_capacity(node_count)}});
}

[[nodiscard]] Mesh make_bounded_mesh(const std::vector<Point>& points) {
    Nodes nodes(static_cast<Index>(points.size()));
    for (Index i = 0; i < static_cast<Index>(points.size()); ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }
    return Mesh(
        std::move(nodes),
        unit_cube_boundary(),
        make_database(points.size()));
}

void compute_bounded_mesh(Mesh& mesh, bool verbose) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::CopyKDSearch{8, 1});

    highvoronoi::RaycastParameters<highvoronoi::CombinedRaycast, Scalar>
        ray_parameters;
    highvoronoi::CombinedRaycastOptions combined_options;
    combined_options.precision_policy =
        highvoronoi::CombinedPrecisionPolicy::Robust;
    combined_options.fallback_method =
        highvoronoi::CombinedFallbackMethod::ClassicGeneralInRangeDegenerate;

    auto raycaster = highvoronoi::make_raycaster(
        tree,
        ray_parameters,
        combined_options);

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

    compute.compute(verbose);
}

[[nodiscard]] highvoronoi::IntegralDataOptions geometry_options() {
    highvoronoi::IntegralDataOptions options;
    options.volume = true;
    options.area = true;
    options.bulk_integral = false;
    options.interface_integral = false;
    return options;
}

[[nodiscard]] GeometryResult collect_integral_geometry(Integral& integral) {
    Mesh& mesh = integral.mesh();
    GeometryResult result;
    result.volumes.resize(static_cast<std::size_t>(mesh.size()));
    result.neighbours.resize(static_cast<std::size_t>(mesh.size()));
    result.areas.resize(static_cast<std::size_t>(mesh.size()));

    typename Integral::Data::CellData cell_data;
    for (Index public_cell = Index{0}; public_cell < mesh.size(); ++public_cell) {
        const Index stable =
            mesh.index_mapping().public_to_internal(public_cell);
        if (!integral.data().read_cell(stable, cell_data)) {
            throw std::runtime_error(
                "integrator did not publish cell " +
                std::to_string(public_cell));
        }

        const std::size_t position = static_cast<std::size_t>(public_cell);
        result.volumes[position] = cell_data.volume();
        result.volume_sum += cell_data.volume();

        result.neighbours[position] = cell_data.neighbours();
        result.areas[position] = cell_data.area();

        if (result.neighbours[position].size() != result.areas[position].size()) {
            throw std::runtime_error(
                "integral cell neighbour/area layout mismatch at cell " +
                std::to_string(public_cell));
        }

        result.area_entries += result.areas[position].size();
        result.area_entry_sum += std::accumulate(
            result.areas[position].begin(),
            result.areas[position].end(),
            0.0);
    }

    return result;
}

[[nodiscard]] GeometryResult run_polygon(Mesh& mesh) {
    Integral integral(
        mesh,
        std::size_t{0},
        geometry_options());
    highvoronoi::PolygonAlgorithm<Integral> algorithm;
    (void)highvoronoi::integrate(integral, algorithm);
    return collect_integral_geometry(integral);
}

[[nodiscard]] GeometryResult run_fast_polygon(Mesh& mesh) {
    Integral integral(
        mesh,
        std::size_t{0},
        geometry_options());
    auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral);
    (void)highvoronoi::integrate(integral, algorithm);
    return collect_integral_geometry(integral);
}

[[nodiscard]] InterfaceComparison compare_interfaces(
    const GeometryResult& left,
    const GeometryResult& right) {
    InterfaceComparison result;
    if (left.neighbours.size() != right.neighbours.size() ||
        left.areas.size() != right.areas.size()) {
        result.layout_equal = false;
        return result;
    }

    for (std::size_t cell = 0; cell < left.neighbours.size(); ++cell) {
        if (left.neighbours[cell] != right.neighbours[cell] ||
            left.areas[cell].size() != right.areas[cell].size()) {
            result.layout_equal = false;
            continue;
        }
        for (std::size_t i = 0; i < left.areas[cell].size(); ++i) {
            result.max_abs_area_error = std::max(
                result.max_abs_area_error,
                std::abs(left.areas[cell][i] - right.areas[cell][i]));
        }
    }
    return result;
}

[[nodiscard]] std::int64_t rationalize_coordinate(
    double value,
    std::int64_t scale,
    Index cell) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "non-finite Voronoi vertex in cell " + std::to_string(cell));
    }

    const long double scaled =
        static_cast<long double>(value) * static_cast<long double>(scale);
    if (scaled < static_cast<long double>((std::numeric_limits<std::int64_t>::min)()) ||
        scaled > static_cast<long double>((std::numeric_limits<std::int64_t>::max)())) {
        throw std::overflow_error(
            "rationalized coordinate does not fit int64_t in cell " +
            std::to_string(cell));
    }
    return static_cast<std::int64_t>(std::llround(scaled));
}

[[nodiscard]] LrsConversionResult convert_mesh_to_lrs_files(
    const Mesh& mesh,
    const std::filesystem::path& directory,
    std::int64_t scale) {
    LrsConversionResult result;
    result.cells.resize(static_cast<std::size_t>(mesh.size()));

    std::vector<RationalVertex> unique_vertices;

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        unique_vertices.clear();

        for (const auto& vertex : mesh.vertices(cell)) {
            RationalVertex rationalized{};
            for (int d = 0; d < Dimension; ++d) {
                rationalized[static_cast<std::size_t>(d)] =
                    rationalize_coordinate(
                        static_cast<double>(vertex.position[d]),
                        scale,
                        cell);
            }
            unique_vertices.push_back(rationalized);
        }

        // Deduplicate after rationalization. Two floating-point vertices that map
        // to the same 1e-14 grid point are intentionally one lrslib vertex.
        std::sort(unique_vertices.begin(), unique_vertices.end());
        unique_vertices.erase(
            std::unique(unique_vertices.begin(), unique_vertices.end()),
            unique_vertices.end());

        LrsCellFile& target = result.cells[static_cast<std::size_t>(cell)];
        target.vertex_count = unique_vertices.size();
        target.input_path =
            directory / ("cell_" + std::to_string(cell) + ".ext");
        target.output_path =
            directory / ("cell_" + std::to_string(cell) + ".out");

        if (unique_vertices.size() < static_cast<std::size_t>(Dimension + 1)) {
            ++result.invalid_cells;
            continue;
        }

        std::ofstream output(target.input_path);
        if (!output) {
            throw std::runtime_error(
                "cannot create lrslib input: " + target.input_path.string());
        }

        // We encode x -> round(scale*x) as INTEGER coordinates. This is exactly
        // equivalent to the rationalized geometry round(scale*x)/scale, while
        // avoiding a common denominator in every input coefficient. The returned
        // d-volume is divided by scale^d after lrslib finishes.
        output << "HighVoronoi_cell_" << cell << '\n'
               << "V-representation\n"
               << "begin\n"
               << unique_vertices.size() << ' ' << (Dimension + 1)
               << " integer\n";
        for (const RationalVertex& vertex : unique_vertices) {
            output << '1';
            for (const std::int64_t coordinate : vertex) {
                output << ' ' << coordinate;
            }
            output << '\n';
        }
        output << "end\n"
               << "volume\n";
        if (!output) {
            throw std::runtime_error(
                "failed while writing lrslib input: " +
                target.input_path.string());
        }

        target.valid = true;
        result.rationalized_vertices += unique_vertices.size();
    }

    return result;
}

void terminate_child(pid_t pid) noexcept {
    if (pid <= 0) {
        return;
    }

    ::kill(pid, SIGTERM);
    for (int attempt = 0; attempt < 20; ++attempt) {
        int status = 0;
        const pid_t state = ::waitpid(pid, &status, WNOHANG);
        if (state == pid || state == -1) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ::kill(pid, SIGKILL);
    int status = 0;
    (void)::waitpid(pid, &status, 0);
}

[[nodiscard]] ProcessResult run_lrs_process(
    const std::string& lrs_binary,
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    double time_budget_s) {
    ProcessResult result;
    if (!(time_budget_s > 0.0)) {
        result.timed_out = true;
        return result;
    }

    const auto begin = Clock::now();
    const pid_t pid = ::fork();
    if (pid < 0) {
        throw std::runtime_error("fork() failed for lrslib");
    }

    if (pid == 0) {
        // Keep the comparison serial even with lrslib versions that were built
        // with OpenMP support.
        (void)::setenv("OMP_NUM_THREADS", "1", 1);
        (void)::setenv("OMP_DYNAMIC", "FALSE", 1);
        ::execlp(
            lrs_binary.c_str(),
            lrs_binary.c_str(),
            input_path.c_str(),
            output_path.c_str(),
            static_cast<char*>(nullptr));
        ::_exit(127);
    }

    for (;;) {
        int status = 0;
        const pid_t state = ::waitpid(pid, &status, WNOHANG);
        if (state == pid) {
            const auto end = Clock::now();
            result.elapsed_s = seconds_between(begin, end);
            result.exited = true;
            if (WIFEXITED(status)) {
                result.exit_code = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                result.exit_code = 128 + WTERMSIG(status);
            }
            return result;
        }
        if (state < 0) {
            terminate_child(pid);
            throw std::runtime_error("waitpid() failed for lrslib");
        }

        const auto now = Clock::now();
        const double elapsed = seconds_between(begin, now);
        if (elapsed >= time_budget_s) {
            terminate_child(pid);
            result.elapsed_s = seconds_between(begin, Clock::now());
            result.timed_out = true;
            return result;
        }

        // 100 us keeps timeout overshoot small without adding millisecond-scale
        // quantization to every successful short cell run.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

[[nodiscard]] long double parse_rational_long_double(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        throw std::runtime_error("empty lrslib volume");
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    text = text.substr(first, last - first + 1);

    const std::size_t slash = text.find('/');
    if (slash == std::string::npos) {
        return std::stold(text);
    }

    const long double numerator = std::stold(text.substr(0, slash));
    const long double denominator = std::stold(text.substr(slash + 1));
    if (denominator == 0.0L) {
        throw std::runtime_error("lrslib returned zero volume denominator");
    }
    return numerator / denominator;
}

[[nodiscard]] double read_lrs_volume(
    const std::filesystem::path& output_path,
    std::int64_t scale) {
    std::ifstream input(output_path);
    if (!input) {
        throw std::runtime_error(
            "cannot open lrslib output: " + output_path.string());
    }

    std::string line;
    constexpr std::string_view marker = "*Volume=";
    while (std::getline(input, line)) {
        const std::size_t position = line.find(marker);
        if (position == std::string::npos) {
            continue;
        }

        const long double integer_coordinate_volume =
            parse_rational_long_double(
                line.substr(position + marker.size()));
        long double scale_power = 1.0L;
        for (int d = 0; d < Dimension; ++d) {
            scale_power *= static_cast<long double>(scale);
        }
        const long double physical_volume =
            integer_coordinate_volume / scale_power;
        const double result = static_cast<double>(physical_volume);
        if (!std::isfinite(result) || result < 0.0) {
            throw std::runtime_error("lrslib returned invalid scaled volume");
        }
        return result;
    }

    throw std::runtime_error(
        "lrslib output contains no *Volume= line: " + output_path.string());
}

[[nodiscard]] LrsRunResult run_lrslib_until_polygon_budget(
    const LrsConversionResult& conversion,
    std::size_t cell_count,
    const std::string& lrs_binary,
    std::int64_t scale,
    double polygon_budget_s,
    bool trace) {
    LrsRunResult result;
    result.volumes.assign(
        cell_count,
        (std::numeric_limits<double>::quiet_NaN)());
    result.invalid_conversion = conversion.invalid_cells;

    bool budget_exhausted = false;
    for (std::size_t cell = 0; cell < conversion.cells.size(); ++cell) {
        const LrsCellFile& input = conversion.cells[cell];
        if (!input.valid) {
            continue;
        }

        const double remaining_budget = polygon_budget_s - result.elapsed_s;
        if (!(remaining_budget > 0.0)) {
            budget_exhausted = true;
            break;
        }

        if (trace) {
            std::cout << "LRS_CELL_BEGIN"
                      << " cell=" << cell
                      << " vertices=" << input.vertex_count
                      << " accumulated_s=" << std::fixed << std::setprecision(6)
                      << result.elapsed_s
                      << " remaining_budget_s=" << remaining_budget
                      << '\n';
            std::cout.flush();
        }

        ++result.attempted;
        const ProcessResult process = run_lrs_process(
            lrs_binary,
            input.input_path,
            input.output_path,
            remaining_budget);
        result.elapsed_s += process.elapsed_s;

        if (process.timed_out) {
            ++result.timed_out;
            budget_exhausted = true;
            if (trace) {
                std::cout << "LRS_CELL_TIMEOUT"
                          << " cell=" << cell
                          << " accumulated_s=" << result.elapsed_s
                          << '\n';
                std::cout.flush();
            }
            break;
        }

        if (!process.exited || process.exit_code != 0) {
            ++result.failed;
            if (trace) {
                std::cout << "LRS_CELL_FAILED"
                          << " cell=" << cell
                          << " exit_code=" << process.exit_code
                          << " cell_s=" << process.elapsed_s
                          << " accumulated_s=" << result.elapsed_s
                          << '\n';
                std::cout.flush();
            }
            continue;
        }

        try {
            const double volume = read_lrs_volume(input.output_path, scale);
            result.volumes[cell] = volume;
            result.partial_volume_sum += volume;
            ++result.completed;
            if (trace) {
                std::cout << std::setprecision(17)
                          << "LRS_CELL_DONE"
                          << " cell=" << cell
                          << " vertices=" << input.vertex_count
                          << " cell_s=" << process.elapsed_s
                          << " accumulated_s=" << result.elapsed_s
                          << " volume=" << volume
                          << '\n';
                std::cout.flush();
            }
        } catch (const std::exception& error) {
            ++result.failed;
            if (trace) {
                std::cout << "LRS_CELL_PARSE_FAILED"
                          << " cell=" << cell
                          << " what=" << error.what()
                          << '\n';
                std::cout.flush();
            }
        }

        if (result.elapsed_s >= polygon_budget_s) {
            budget_exhausted = true;
            break;
        }
    }

    if (budget_exhausted) {
        const std::size_t accounted =
            result.completed + result.failed + result.timed_out +
            result.invalid_conversion;
        result.unattempted_due_budget =
            accounted < cell_count ? cell_count - accounted : 0;
    }

    return result;
}

[[nodiscard]] PartialVolumeError compare_partial_volumes(
    const std::vector<double>& reference,
    const std::vector<double>& candidate) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("partial volume vectors have different sizes");
    }

    PartialVolumeError result;
    double abs_sum = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(candidate[i])) {
            continue;
        }

        const double abs_error = std::abs(candidate[i] - reference[i]);
        const double denominator = std::max(
            std::abs(reference[i]),
            std::numeric_limits<double>::min());
        const double rel_error = abs_error / denominator;
        result.max_abs = std::max(result.max_abs, abs_error);
        result.max_rel = std::max(result.max_rel, rel_error);
        abs_sum += abs_error;
        ++result.compared;
    }

    if (result.compared != 0) {
        result.mean_abs = abs_sum / static_cast<double>(result.compared);
    }
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

void print_time_summary(
    std::string_view name,
    const std::vector<double>& values) {
    if (values.empty()) {
        return;
    }
    const auto [minimum, maximum] =
        std::minmax_element(values.begin(), values.end());
    std::cout << std::fixed << std::setprecision(6)
              << "SUMMARY_TIME"
              << " name=" << name
              << " repeats=" << values.size()
              << " min_s=" << *minimum
              << " median_s=" << median(values)
              << " mean_s=" << mean(values)
              << " stddev_s=" << standard_deviation(values)
              << " max_s=" << *maximum
              << '\n';
}

void print_repeat_result(
    const RepeatResult& result,
    const Options& options) {
    const std::size_t lrs_not_completed =
        options.nodes > result.lrs_completed
            ? options.nodes - result.lrs_completed
            : 0;

    std::cout << std::fixed << std::setprecision(6)
              << "RESULT"
              << " nodes=" << options.nodes
              << " repeat=" << result.repeat
              << " polygon_s=" << result.polygon_s
              << " fast_polygon_s=" << result.fast_polygon_s
              << " lrslib_conversion_s=" << result.lrslib_conversion_s
              << " lrslib_volume_s=" << result.lrslib_volume_s
              << " lrslib_total_s="
              << (result.lrslib_conversion_s + result.lrslib_volume_s)
              << '\n';

    std::cout << "LRS_STATUS"
              << " repeat=" << result.repeat
              << " budget_s=" << result.polygon_s
              << " attempted=" << result.lrs_attempted
              << " completed=" << result.lrs_completed
              << " failed=" << result.lrs_failed
              << " timed_out=" << result.lrs_timed_out
              << " invalid_conversion=" << result.lrs_invalid_conversion
              << " unattempted_due_budget="
              << result.lrs_unattempted_due_budget
              << " not_completed=" << lrs_not_completed
              << " completed_fraction="
              << static_cast<double>(result.lrs_completed) /
                     static_cast<double>(options.nodes)
              << '\n';

    std::cout << std::setprecision(17)
              << "CHECK"
              << " repeat=" << result.repeat
              << " polygon_volume_sum=" << result.polygon_volume_sum
              << " fast_polygon_volume_sum=" << result.fast_polygon_volume_sum
              << " polygon_abs_sum_error="
              << std::abs(result.polygon_volume_sum - 1.0)
              << " fast_polygon_abs_sum_error="
              << std::abs(result.fast_polygon_volume_sum - 1.0)
              << " polygon_area_entries=" << result.polygon_area_entries
              << " fast_area_entries=" << result.fast_polygon_area_entries
              << " polygon_area_entry_sum=" << result.polygon_area_entry_sum
              << " fast_area_entry_sum=" << result.fast_polygon_area_entry_sum
              << " polygon_fast_interface_layout_equal="
              << (result.polygon_fast_interface_layout_equal ? "YES" : "NO")
              << " polygon_fast_max_abs_area_error="
              << result.polygon_fast_max_abs_area_error
              << " lrs_partial_volume_sum=" << result.lrs_partial_volume_sum
              << " lrs_compared_cells=" << result.lrs_compared_cells
              << " lrs_max_abs_cell_error_vs_polygon="
              << result.lrs_max_abs_cell_error_vs_polygon
              << " lrs_mean_abs_cell_error_vs_polygon="
              << result.lrs_mean_abs_cell_error_vs_polygon
              << " lrs_max_rel_cell_error_vs_polygon="
              << result.lrs_max_rel_cell_error_vs_polygon
              << '\n';
}

void write_results_csv(
    const std::string& path,
    const Options& options,
    const std::vector<RepeatResult>& results) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open results output: " + path);
    }

    output
        << "dimension,nodes,repeat,seed,rational_scale,"
        << "polygon_s,fast_polygon_s,lrslib_conversion_s,lrslib_volume_s,lrslib_total_s,"
        << "polygon_volume_sum,fast_polygon_volume_sum,"
        << "polygon_area_entries,fast_polygon_area_entries,"
        << "polygon_area_entry_sum,fast_polygon_area_entry_sum,"
        << "polygon_fast_interface_layout_equal,polygon_fast_max_abs_area_error,"
        << "lrs_attempted,lrs_completed,lrs_failed,lrs_timed_out,lrs_invalid_conversion,"
        << "lrs_unattempted_due_budget,lrs_partial_volume_sum,lrs_compared_cells,"
        << "lrs_max_abs_cell_error_vs_polygon,lrs_mean_abs_cell_error_vs_polygon,"
        << "lrs_max_rel_cell_error_vs_polygon\n";

    output << std::setprecision(17);
    for (const RepeatResult& result : results) {
        output
            << Dimension << ','
            << options.nodes << ','
            << result.repeat << ','
            << options.seed << ','
            << options.rational_scale << ','
            << result.polygon_s << ','
            << result.fast_polygon_s << ','
            << result.lrslib_conversion_s << ','
            << result.lrslib_volume_s << ','
            << (result.lrslib_conversion_s + result.lrslib_volume_s) << ','
            << result.polygon_volume_sum << ','
            << result.fast_polygon_volume_sum << ','
            << result.polygon_area_entries << ','
            << result.fast_polygon_area_entries << ','
            << result.polygon_area_entry_sum << ','
            << result.fast_polygon_area_entry_sum << ','
            << (result.polygon_fast_interface_layout_equal ? 1 : 0) << ','
            << result.polygon_fast_max_abs_area_error << ','
            << result.lrs_attempted << ','
            << result.lrs_completed << ','
            << result.lrs_failed << ','
            << result.lrs_timed_out << ','
            << result.lrs_invalid_conversion << ','
            << result.lrs_unattempted_due_budget << ','
            << result.lrs_partial_volume_sum << ','
            << result.lrs_compared_cells << ','
            << result.lrs_max_abs_cell_error_vs_polygon << ','
            << result.lrs_mean_abs_cell_error_vs_polygon << ','
            << result.lrs_max_rel_cell_error_vs_polygon
            << '\n';
    }
}

void usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "5D bounded [0,1]^5 volume/interface benchmark against lrslib.\n"
        << "The Voronoi mesh is computed once before timing.\n\n"
        << "Per repeat:\n"
        << "  1. Polygon:     all cell volumes + interfaces + readback\n"
        << "  2. FastPolygon: all cell volumes + interfaces + readback\n"
        << "  3. lrslib conversion: cell vertices -> rationalized V-representation files\n"
        << "  4. lrslib volume: cells in public-index order until cumulative time >= Polygon time\n\n"
        << "Options:\n"
        << "  --nodes N             Node count (default: 1000)\n"
        << "  --repeats N           Fresh integration/conversion runs (default: 5)\n"
        << "  --seed U64            SplitMix64 seed (default fixed)\n"
        << "  --rational-scale N    round(x*N)/N grid (default: 100000000000000 = 1e14)\n"
        << "  --lrs-binary FILE     lrslib executable (default: lrs)\n"
        << "  --points-in FILE      Read exact 5D generator points from CSV\n"
        << "  --points-out FILE     Write generated/read generator points to CSV\n"
        << "  --csv FILE            Write raw timing/correctness rows to CSV\n"
        << "  --verbose-mesh        Enable progress output for untimed mesh build\n"
        << "  --lrs-trace           Print one line before/after every attempted lrslib cell\n"
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
            options.nodes = static_cast<std::size_t>(
                std::stoull(require_value()));
            if (options.nodes <= static_cast<std::size_t>(Dimension)) {
                throw std::invalid_argument("--nodes must exceed dimension");
            }
        } else if (arg == "--repeats") {
            options.repeats = static_cast<std::size_t>(
                std::stoull(require_value()));
            if (options.repeats == 0) {
                throw std::invalid_argument("--repeats must be positive");
            }
        } else if (arg == "--seed") {
            options.seed = std::stoull(require_value(), nullptr, 0);
        } else if (arg == "--rational-scale") {
            options.rational_scale = std::stoll(require_value());
            if (options.rational_scale <= 0) {
                throw std::invalid_argument("--rational-scale must be positive");
            }
        } else if (arg == "--lrs-binary") {
            options.lrs_binary = require_value();
        } else if (arg == "--points-in") {
            options.points_in = require_value();
        } else if (arg == "--points-out") {
            options.points_out = require_value();
        } else if (arg == "--csv") {
            options.csv_out = require_value();
        } else if (arg == "--verbose-mesh") {
            options.verbose_mesh = true;
        } else if (arg == "--lrs-trace") {
            options.lrs_trace = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return options;
}

void print_configuration(const Options& options) {
    std::cout
        << "HighVoronoi 5D bounded integration benchmark against lrslib\n"
        << "dimension:          " << Dimension << '\n'
        << "domain:             [0,1]^5\n"
        << "nodes:              " << options.nodes << '\n'
        << "repeats:            " << options.repeats << '\n'
        << "seed:               0x" << std::hex << options.seed << std::dec << '\n'
        << "mesh raycast:       Combined Robust / auto fallback\n"
        << "mesh search:        CopyKDSearch{8,1}\n"
        << "rationalization:    round(x * " << options.rational_scale
        << ") / " << options.rational_scale << '\n'
        << "lrslib executable:  " << options.lrs_binary << '\n'
        << "lrslib threads:     1 (OMP_NUM_THREADS=1)\n"
        << "lrslib cutoff:      cumulative lrslib volume time >= Polygon time\n"
        << "timing contract:    polygon / fast_polygon / lrslib_conversion / lrslib_volume\n"
        << "Polygon output:     cell volumes + cell-local interface areas\n"
        << "FastPolygon output: cell volumes + cell-local interface areas\n";
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
        print_configuration(options);

        std::vector<Point> points = options.points_in.empty()
            ? generate_points(options.nodes, options.seed)
            : read_points_csv(options.points_in);

        if (points.size() < options.nodes) {
            throw std::runtime_error(
                "point set contains fewer rows than --nodes");
        }
        if (points.size() > options.nodes) {
            points.resize(options.nodes);
        }
        if (!options.points_out.empty()) {
            write_points_csv(options.points_out, points);
        }

        std::cout << "BUILD mesh nodes=" << options.nodes
                  << " domain=bounded method=combined-robust\n";
        Mesh mesh = make_bounded_mesh(points);
        compute_bounded_mesh(mesh, options.verbose_mesh);
        std::cout << "MESH_READY cells=" << mesh.size() << "\n\n";

        std::vector<RepeatResult> results;
        results.reserve(options.repeats);

        for (std::size_t repeat = 1; repeat <= options.repeats; ++repeat) {
            RepeatResult row;
            row.repeat = repeat;

            std::cout << "RUN repeat=" << repeat << '/' << options.repeats
                      << " method=polygon\n";
            const auto polygon_begin = Clock::now();
            GeometryResult polygon = run_polygon(mesh);
            const auto polygon_end = Clock::now();
            row.polygon_s = seconds_between(polygon_begin, polygon_end);
            row.polygon_volume_sum = polygon.volume_sum;
            row.polygon_area_entries = polygon.area_entries;
            row.polygon_area_entry_sum = polygon.area_entry_sum;

            std::cout << "RUN repeat=" << repeat << '/' << options.repeats
                      << " method=fast-polygon\n";
            const auto fast_begin = Clock::now();
            GeometryResult fast_polygon = run_fast_polygon(mesh);
            const auto fast_end = Clock::now();
            row.fast_polygon_s = seconds_between(fast_begin, fast_end);
            row.fast_polygon_volume_sum = fast_polygon.volume_sum;
            row.fast_polygon_area_entries = fast_polygon.area_entries;
            row.fast_polygon_area_entry_sum = fast_polygon.area_entry_sum;

            const InterfaceComparison interface_comparison =
                compare_interfaces(polygon, fast_polygon);
            row.polygon_fast_interface_layout_equal =
                interface_comparison.layout_equal;
            row.polygon_fast_max_abs_area_error =
                interface_comparison.max_abs_area_error;

            TempDirectory temp_directory;
            std::cout << "RUN repeat=" << repeat << '/' << options.repeats
                      << " method=lrslib-conversion\n";
            const auto conversion_begin = Clock::now();
            LrsConversionResult conversion = convert_mesh_to_lrs_files(
                mesh,
                temp_directory.path(),
                options.rational_scale);
            const auto conversion_end = Clock::now();
            row.lrslib_conversion_s =
                seconds_between(conversion_begin, conversion_end);
            std::cout << "LRS_CONVERSION"
                      << " repeat=" << repeat
                      << " rationalized_vertices=" << conversion.rationalized_vertices
                      << " invalid_cells=" << conversion.invalid_cells
                      << '\n';

            std::cout << "RUN repeat=" << repeat << '/' << options.repeats
                      << " method=lrslib-volume"
                      << " budget_s=" << std::fixed << std::setprecision(6)
                      << row.polygon_s << '\n';
            LrsRunResult lrs = run_lrslib_until_polygon_budget(
                conversion,
                options.nodes,
                options.lrs_binary,
                options.rational_scale,
                row.polygon_s,
                options.lrs_trace);

            row.lrslib_volume_s = lrs.elapsed_s;
            row.lrs_attempted = lrs.attempted;
            row.lrs_completed = lrs.completed;
            row.lrs_failed = lrs.failed;
            row.lrs_timed_out = lrs.timed_out;
            row.lrs_invalid_conversion = lrs.invalid_conversion;
            row.lrs_unattempted_due_budget = lrs.unattempted_due_budget;
            row.lrs_partial_volume_sum = lrs.partial_volume_sum;

            const PartialVolumeError lrs_error =
                compare_partial_volumes(polygon.volumes, lrs.volumes);
            row.lrs_compared_cells = lrs_error.compared;
            row.lrs_max_abs_cell_error_vs_polygon = lrs_error.max_abs;
            row.lrs_mean_abs_cell_error_vs_polygon = lrs_error.mean_abs;
            row.lrs_max_rel_cell_error_vs_polygon = lrs_error.max_rel;

            print_repeat_result(row, options);
            results.push_back(std::move(row));
            std::cout << '\n';
        }

        std::vector<double> polygon_times;
        std::vector<double> fast_polygon_times;
        std::vector<double> lrslib_conversion_times;
        std::vector<double> lrslib_volume_times;
        polygon_times.reserve(results.size());
        fast_polygon_times.reserve(results.size());
        lrslib_conversion_times.reserve(results.size());
        lrslib_volume_times.reserve(results.size());

        std::vector<double> lrs_completed_counts;
        lrs_completed_counts.reserve(results.size());

        for (const RepeatResult& result : results) {
            polygon_times.push_back(result.polygon_s);
            fast_polygon_times.push_back(result.fast_polygon_s);
            lrslib_conversion_times.push_back(result.lrslib_conversion_s);
            lrslib_volume_times.push_back(result.lrslib_volume_s);
            lrs_completed_counts.push_back(
                static_cast<double>(result.lrs_completed));
        }

        print_time_summary("polygon", polygon_times);
        print_time_summary("fast_polygon", fast_polygon_times);
        print_time_summary("lrslib_conversion", lrslib_conversion_times);
        print_time_summary("lrslib_volume_until_polygon_cutoff", lrslib_volume_times);

        if (!lrs_completed_counts.empty()) {
            const auto [minimum, maximum] = std::minmax_element(
                lrs_completed_counts.begin(), lrs_completed_counts.end());
            std::cout << std::fixed << std::setprecision(1)
                      << "SUMMARY_LRS_PROGRESS"
                      << " repeats=" << lrs_completed_counts.size()
                      << " completed_min=" << *minimum
                      << " completed_median=" << median(lrs_completed_counts)
                      << " completed_mean=" << mean(lrs_completed_counts)
                      << " completed_max=" << *maximum
                      << " of=" << options.nodes
                      << '\n';
        }

        if (!options.csv_out.empty()) {
            write_results_csv(options.csv_out, options, results);
        }

        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "benchmark error: " << exception.what() << '\n';
        return 1;
    }
}
