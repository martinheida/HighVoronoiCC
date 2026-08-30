#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters>;

inline constexpr std::array<unsigned, 3> Percentages{5u, 10u, 20u};
inline constexpr std::array<Index, 3> NodeCounts{
    Index{30}, Index{100}, Index{300}};
inline constexpr std::size_t RandomRuns = 3;

inline constexpr Scalar CoordinateMinimum = Scalar{-2.5};
inline constexpr Scalar CoordinateMaximum = Scalar{2.5};
inline constexpr Scalar BoundarySize = Scalar{12};
inline constexpr Scalar MinimumNodeDistance = Scalar{0.04};
inline constexpr Scalar PositionTolerance = Scalar{1e-10};
inline constexpr Scalar VerificationVariance = Scalar{1e-18};

inline constexpr std::size_t DatabaseBlockUnits = 262144;
inline constexpr std::size_t DatabaseHashCapacity = 524288;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;

    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

struct VertexRecord {
    std::vector<Index> sigma;
    std::vector<Scalar> position;
};

struct CellDifference {
    std::vector<VertexRecord> missing_from_partial;
    std::vector<VertexRecord> extra_in_partial;

    [[nodiscard]] bool equal() const noexcept {
        return missing_from_partial.empty() &&
               extra_in_partial.empty();
    }
};

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        DatabaseBlockUnits,
        DatabaseParameters{
            highvoronoi::DirectHash{DatabaseHashCapacity}});
}

template <int Dimension>
struct CaseTypes {
    using Mesh =
        highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
    using Nodes = typename Mesh::InternalNodes;
    using Point = typename Mesh::NodePoint;
    using Boundary = typename Mesh::BoundaryType;
};

template <int Dimension>
typename CaseTypes<Dimension>::Boundary make_boundary() {
    using Point = typename CaseTypes<Dimension>::Point;
    using Boundary = typename CaseTypes<Dimension>::Boundary;

    return Boundary::cuboid(
        Point::Constant(BoundarySize),
        Point::Constant(-BoundarySize / Scalar{2}),
        std::vector<Index>{});
}

template <int Dimension>
std::vector<typename CaseTypes<Dimension>::Point>
make_random_points(
    Index node_count,
    std::uint64_t seed) {

    using Point = typename CaseTypes<Dimension>::Point;

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(node_count));

    std::mt19937_64 random(seed);
    std::uniform_real_distribution<Scalar> coordinate(
        CoordinateMinimum,
        CoordinateMaximum);

    while (points.size() < static_cast<std::size_t>(node_count)) {
        Point candidate;
        for (int d = 0; d < Dimension; ++d) {
            candidate[d] = coordinate(random);
        }

        bool accepted = true;
        for (const Point& existing : points) {
            if ((candidate - existing).norm() < MinimumNodeDistance) {
                accepted = false;
                break;
            }
        }

        if (accepted) {
            points.push_back(candidate);
        }
    }

    return points;
}

template <int Dimension>
typename CaseTypes<Dimension>::Nodes make_nodes(
    const std::vector<typename CaseTypes<Dimension>::Point>& points) {

    using Nodes = typename CaseTypes<Dimension>::Nodes;

    Nodes nodes(static_cast<Index>(points.size()));
    for (Index i = Index{0};
         i < static_cast<Index>(points.size());
         ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }
    return nodes;
}

template <int Dimension>
typename CaseTypes<Dimension>::Mesh make_mesh(
    const std::vector<typename CaseTypes<Dimension>::Point>& points) {

    using Mesh = typename CaseTypes<Dimension>::Mesh;

    return Mesh(
        make_nodes<Dimension>(points),
        make_boundary<Dimension>(),
        make_database());
}

template <class Mesh>
void compute_mesh(
    Mesh& mesh,
    std::optional<typename Mesh::Index> range_end) {

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    highvoronoi::RaycastParameters<
        highvoronoi::InRangeRaycast,
        Scalar> ray_parameters;
    ray_parameters.variance_tolerance = Scalar{9e-14};

    auto raycaster =
        highvoronoi::make_raycaster(tree, ray_parameters);

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread>;

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        range_end);

    compute.compute();
}

template <class Mesh>
std::vector<VertexRecord> capture_cell(
    const Mesh& mesh,
    typename Mesh::Index cell) {

    std::vector<VertexRecord> result;

    for (const auto& vertex : mesh.vertices(cell)) {
        VertexRecord record;

        record.sigma.assign(
            vertex.sigma.begin(),
            vertex.sigma.end());
        std::sort(record.sigma.begin(), record.sigma.end());

        record.position.reserve(
            static_cast<std::size_t>(mesh.dimension()));
        for (typename Mesh::Index d = typename Mesh::Index{0};
             d < mesh.dimension();
             ++d) {
            record.position.push_back(
                vertex.position[static_cast<Eigen::Index>(d)]);
        }

        result.push_back(std::move(record));
    }

    return result;
}

bool same_vertex(
    const VertexRecord& left,
    const VertexRecord& right) {

    if (left.sigma != right.sigma ||
        left.position.size() != right.position.size()) {
        return false;
    }

    Scalar squared_distance = Scalar{0};
    for (std::size_t d = 0; d < left.position.size(); ++d) {
        const Scalar difference =
            left.position[d] - right.position[d];
        squared_distance += difference * difference;
    }

    return squared_distance <=
           PositionTolerance * PositionTolerance;
}

CellDifference compare_cell(
    const std::vector<VertexRecord>& reference,
    const std::vector<VertexRecord>& partial) {

    CellDifference difference;
    std::vector<bool> partial_used(partial.size(), false);

    for (const VertexRecord& expected : reference) {
        bool found = false;

        for (std::size_t candidate = 0;
             candidate < partial.size();
             ++candidate) {
            if (partial_used[candidate]) {
                continue;
            }

            if (same_vertex(expected, partial[candidate])) {
                partial_used[candidate] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            difference.missing_from_partial.push_back(expected);
        }
    }

    for (std::size_t candidate = 0;
         candidate < partial.size();
         ++candidate) {
        if (!partial_used[candidate]) {
            difference.extra_in_partial.push_back(partial[candidate]);
        }
    }

    return difference;
}

void print_vertex(const VertexRecord& vertex) {
    std::cerr << "position=(";
    for (std::size_t d = 0; d < vertex.position.size(); ++d) {
        if (d != 0) {
            std::cerr << ',';
        }
        std::cerr << std::setprecision(15) << vertex.position[d];
    }

    std::cerr << ") sigma={";
    for (std::size_t i = 0; i < vertex.sigma.size(); ++i) {
        if (i != 0) {
            std::cerr << ',';
        }
        std::cerr << vertex.sigma[i];
    }
    std::cerr << '}';
}

void print_first_difference(
    Index cell,
    const CellDifference& difference) {

    std::cerr
        << "        first differing cell: " << cell << '\n'
        << "        missing from partial: "
        << difference.missing_from_partial.size() << '\n'
        << "        extra in partial:      "
        << difference.extra_in_partial.size() << '\n';

    constexpr std::size_t MaximumPrintedVertices = 6;

    std::size_t printed = 0;
    for (const VertexRecord& vertex :
         difference.missing_from_partial) {
        if (printed == MaximumPrintedVertices) {
            break;
        }
        std::cerr << "          MISSING: ";
        print_vertex(vertex);
        std::cerr << '\n';
        ++printed;
    }

    printed = 0;
    for (const VertexRecord& vertex :
         difference.extra_in_partial) {
        if (printed == MaximumPrintedVertices) {
            break;
        }
        std::cerr << "          EXTRA:   ";
        print_vertex(vertex);
        std::cerr << '\n';
        ++printed;
    }
}

Index first_cell_count(Index node_count, unsigned percentage) {
    const std::uint64_t numerator =
        static_cast<std::uint64_t>(node_count) *
        static_cast<std::uint64_t>(percentage);

    return static_cast<Index>(
        std::max<std::uint64_t>(
            1u,
            (numerator + 99u) / 100u));
}

template <class Mesh>
bool compare_first_cells(
    const Mesh& reference,
    const Mesh& partial,
    Index cell_count,
    bool print_difference) {

    for (Index cell = Index{0};
         cell < cell_count;
         ++cell) {

        CellDifference difference =
            compare_cell(
                capture_cell(reference, cell),
                capture_cell(partial, cell));

        if (!difference.equal()) {
            if (print_difference) {
                print_first_difference(cell, difference);
            }
            return false;
        }
    }

    return true;
}

std::uint64_t random_seed(
    int dimension,
    Index node_count,
    std::size_t run) {

    const std::uint64_t dimension_salt =
        dimension == 3
            ? 0x334452414e474500ULL
            : 0x344452414e474500ULL;

    return dimension_salt ^
           (static_cast<std::uint64_t>(node_count) << 24u) ^
           (0x9e3779b97f4a7c15ULL *
            static_cast<std::uint64_t>(run + 1u));
}

template <int Dimension>
void run_random_matrix_for_node_count(Index node_count) {
    std::array<std::size_t, Percentages.size()> passed{};
    std::size_t valid_references = 0;
    bool printed_failure = false;

    for (std::size_t run = 0; run < RandomRuns; ++run) {
        const auto points =
            make_random_points<Dimension>(
                node_count,
                random_seed(Dimension, node_count, run));

        auto reference = make_mesh<Dimension>(points);
        compute_mesh(reference, std::nullopt);

        const auto verification = highvoronoi::verify_mesh(
            reference,
            VerificationVariance,
            false);

        if (verification.valid()) {
            ++valid_references;
        } else if (!printed_failure) {
            std::cerr
                << "        full reference failed geometric verification"
                << " in run " << (run + 1) << '\n';
            printed_failure = true;
        }

        for (std::size_t p = 0; p < Percentages.size(); ++p) {
            const unsigned percentage = Percentages[p];
            const Index cell_count =
                first_cell_count(node_count, percentage);

            auto partial = make_mesh<Dimension>(points);
            compute_mesh(
                partial,
                std::optional<Index>{
                    static_cast<Index>(cell_count - Index{1})});

            const bool equal =
                compare_first_cells(
                    reference,
                    partial,
                    cell_count,
                    !printed_failure);

            if (equal) {
                ++passed[p];
            } else if (!printed_failure) {
                std::cerr
                    << "        D=" << Dimension
                    << " N=" << node_count
                    << " " << percentage << "%"
                    << " failed in random run "
                    << (run + 1) << '\n';
                printed_failure = true;
            }
        }
    }

    check(
        valid_references == RandomRuns,
        "D=" + std::to_string(Dimension) +
        " N=" + std::to_string(node_count) +
        " full references valid in 3/3 random runs");

    for (std::size_t p = 0; p < Percentages.size(); ++p) {
        const unsigned percentage = Percentages[p];

        check(
            passed[p] == RandomRuns,
            "D=" + std::to_string(Dimension) +
            " N=" + std::to_string(node_count) +
            " first " + std::to_string(percentage) +
            "% matches full compute in " +
            std::to_string(passed[p]) + "/3 random runs");
    }
}

template <int Dimension>
void run_random_matrix() {
    for (const Index node_count : NodeCounts) {
        run_random_matrix_for_node_count<Dimension>(node_count);
    }
}

// -----------------------------------------------------------------------------
// Historical 27 + 3 regression data.
// The first 27 points are generated by the same sequence used by the previous
// HighVoronoi incremental test. The three fixed points are exactly its later
// refinement insertion. They are physically moved to the front here so this
// remains a pure VoronoiMesh/range_end test.
// -----------------------------------------------------------------------------

using HistoricalPoint = CaseTypes<3>::Point;

HistoricalPoint historical_point(
    Scalar x,
    Scalar y,
    Scalar z) {
    HistoricalPoint result;
    result << x, y, z;
    return result;
}

std::vector<HistoricalPoint> historical_future_points() {
    return {
        historical_point(0.008, 0.18, 0.37),
        historical_point(0.992, 0.73, 0.63),
        historical_point(0.43, 0.009, 0.84)
    };
}

bool far_enough(
    const HistoricalPoint& candidate,
    const std::vector<HistoricalPoint>& points,
    Scalar minimum_distance) {

    for (const HistoricalPoint& existing : points) {
        if ((candidate - existing).norm() < minimum_distance) {
            return false;
        }
    }
    return true;
}

std::vector<HistoricalPoint>
make_historical_27_plus_3_reordered() {
    constexpr Index OldCount = Index{27};
    constexpr Scalar OldMinimumDistance = Scalar{0.055};

    const std::vector<HistoricalPoint> future =
        historical_future_points();

    std::vector<HistoricalPoint> old_points;
    old_points.reserve(static_cast<std::size_t>(OldCount));

    std::mt19937_64 random(0x494e4352454d454eULL);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{0.025},
        Scalar{0.975});

    while (old_points.size() <
           static_cast<std::size_t>(OldCount)) {
        const HistoricalPoint candidate = historical_point(
            coordinate(random),
            coordinate(random),
            coordinate(random));

        if (!far_enough(
                candidate,
                old_points,
                OldMinimumDistance)) {
            continue;
        }

        if (!far_enough(
                candidate,
                future,
                Scalar{0.085})) {
            continue;
        }

        old_points.push_back(candidate);
    }

    std::vector<HistoricalPoint> reordered;
    reordered.reserve(30);
    reordered.insert(
        reordered.end(),
        future.begin(),
        future.end());
    reordered.insert(
        reordered.end(),
        old_points.begin(),
        old_points.end());

    return reordered;
}

void run_historical_case() {
    const auto points =
        make_historical_27_plus_3_reordered();

    auto reference = make_mesh<3>(points);
    compute_mesh(reference, std::nullopt);

    auto partial = make_mesh<3>(points);
    compute_mesh(
        partial,
        std::optional<Index>{Index{2}});

    const bool reference_valid =
        highvoronoi::verify_mesh(
            reference,
            VerificationVariance,
            false).valid();

    check(
        reference_valid,
        "historical 27+3 reordered full reference is valid");

    const bool equal =
        compare_first_cells(
            reference,
            partial,
            Index{3},
            true);

    check(
        equal,
        "historical 27+3 case: first 3 cells match full compute");
}

} // namespace

int main() {
    std::cout
        << "============================================================\n"
        << "Repeated ComputeVoronoi range_end consistency test\n"
        << "============================================================\n"
        << "3D + 4D; N=30,100,300; first 5%,10%,20%;\n"
        << "three deterministic random node sets per combination.\n"
        << "Plus the historical 27 old + 3 new refinement geometry.\n"
        << "============================================================\n";

    run_random_matrix<3>();
    run_random_matrix<4>();
    run_historical_case();

    std::cout
        << "============================================================\n"
        << "performed checks: " << performed_checks << '\n'
        << "failed checks:    " << failed_checks << '\n'
        << "============================================================\n";

    return failed_checks == 0 ? 0 : 1;
}
