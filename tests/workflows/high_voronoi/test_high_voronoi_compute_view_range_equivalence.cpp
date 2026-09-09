

#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/high_voronoi_compute_mesh.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
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
template <int Dimension>
using DatabaseFor = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters,
    Dimension>;

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
    std::vector<VertexRecord> left_only;
    std::vector<VertexRecord> right_only;

    [[nodiscard]] bool equal() const noexcept {
        return left_only.empty() && right_only.empty();
    }
};

template <int Dimension>
std::shared_ptr<DatabaseFor<Dimension>> make_database() {
    return std::make_shared<DatabaseFor<Dimension>>(
        DatabaseBlockUnits,
        DatabaseParameters{
            highvoronoi::DirectHash{DatabaseHashCapacity}});
}

DatabaseParameters database_parameters() {
    return DatabaseParameters{
        highvoronoi::DirectHash{DatabaseHashCapacity}};
}

template <int Dimension>
struct CaseTypes {
    using HighMesh =
        highvoronoi::HighVoronoiMesh<
            Scalar,
            Dimension,
            DatabaseFor<Dimension>>;

    using RoundState =
        highvoronoi::HighVoronoiComputeRoundState<HighMesh, highvoronoi::detail::BitVector>;

    using ComputeView =
        highvoronoi::HighVoronoiComputeMesh<HighMesh, highvoronoi::detail::BitVector>;

    using PlainMesh =
        highvoronoi::VoronoiMesh<
            Scalar,
            Dimension,
            DatabaseFor<Dimension>>;

    using PlainNodes = typename PlainMesh::InternalNodes;
    using Point = typename PlainMesh::NodePoint;
    using Boundary = typename PlainMesh::BoundaryType;
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
typename CaseTypes<Dimension>::PlainNodes make_plain_nodes(
    const std::vector<typename CaseTypes<Dimension>::Point>& points) {

    using PlainNodes = typename CaseTypes<Dimension>::PlainNodes;

    PlainNodes nodes(static_cast<Index>(points.size()));
    for (Index i = Index{0};
         i < static_cast<Index>(points.size());
         ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }
    return nodes;
}

template <int Dimension>
typename CaseTypes<Dimension>::PlainMesh make_plain_mesh(
    const std::vector<typename CaseTypes<Dimension>::Point>& points) {

    using PlainMesh = typename CaseTypes<Dimension>::PlainMesh;

    return PlainMesh(
        make_plain_nodes<Dimension>(points),
        make_boundary<Dimension>(),
        make_database<Dimension>());
}

template <int Dimension>
std::unique_ptr<typename CaseTypes<Dimension>::HighMesh>
make_high_mesh(
    const std::vector<typename CaseTypes<Dimension>::Point>& points) {

    using HighMesh = typename CaseTypes<Dimension>::HighMesh;

    auto mesh = std::make_unique<HighMesh>(
        Index{Dimension},
        make_boundary<Dimension>(),
        std::in_place,
        DatabaseBlockUnits,
        database_parameters());

    for (const auto& point : points) {
        (void)mesh->append_visible_node(point);
    }

    return mesh;
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
    const std::vector<VertexRecord>& left,
    const std::vector<VertexRecord>& right) {

    CellDifference difference;
    std::vector<bool> right_used(right.size(), false);

    for (const VertexRecord& expected : left) {
        bool found = false;

        for (std::size_t candidate = 0;
             candidate < right.size();
             ++candidate) {
            if (right_used[candidate]) {
                continue;
            }

            if (same_vertex(expected, right[candidate])) {
                right_used[candidate] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            difference.left_only.push_back(expected);
        }
    }

    for (std::size_t candidate = 0;
         candidate < right.size();
         ++candidate) {
        if (!right_used[candidate]) {
            difference.right_only.push_back(right[candidate]);
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
    std::string_view left_name,
    std::string_view right_name,
    Index cell,
    const CellDifference& difference) {

    std::cerr
        << "        comparison " << left_name
        << " vs " << right_name << '\n'
        << "        first differing cell: " << cell << '\n'
        << "        " << left_name << " only: "
        << difference.left_only.size() << '\n'
        << "        " << right_name << " only: "
        << difference.right_only.size() << '\n';

    constexpr std::size_t MaximumPrintedVertices = 6;

    std::size_t printed = 0;
    for (const VertexRecord& vertex : difference.left_only) {
        if (printed == MaximumPrintedVertices) {
            break;
        }
        std::cerr << "          LEFT:  ";
        print_vertex(vertex);
        std::cerr << '\n';
        ++printed;
    }

    printed = 0;
    for (const VertexRecord& vertex : difference.right_only) {
        if (printed == MaximumPrintedVertices) {
            break;
        }
        std::cerr << "          RIGHT: ";
        print_vertex(vertex);
        std::cerr << '\n';
        ++printed;
    }
}

template <class LeftMesh, class RightMesh>
bool compare_first_cells(
    const LeftMesh& left,
    const RightMesh& right,
    Index cell_count,
    std::string_view left_name,
    std::string_view right_name,
    bool print_difference) {

    for (Index cell = Index{0};
         cell < cell_count;
         ++cell) {

        CellDifference difference =
            compare_cell(
                capture_cell(left, cell),
                capture_cell(right, cell));

        if (!difference.equal()) {
            if (print_difference) {
                print_first_difference(
                    left_name,
                    right_name,
                    cell,
                    difference);
            }
            return false;
        }
    }

    return true;
}

Index selected_count(Index node_count, unsigned percentage) {
    const std::uint64_t numerator =
        static_cast<std::uint64_t>(node_count) *
        static_cast<std::uint64_t>(percentage);

    return static_cast<Index>(
        std::max<std::uint64_t>(
            1u,
            (numerator + 99u) / 100u));
}

template <class Point>
std::vector<Point> move_tail_to_front(
    const std::vector<Point>& points,
    Index tail_count) {

    const std::size_t count = points.size();
    const std::size_t tail =
        static_cast<std::size_t>(tail_count);
    const std::size_t split = count - tail;

    std::vector<Point> reordered;
    reordered.reserve(count);

    reordered.insert(
        reordered.end(),
        points.begin() + static_cast<std::ptrdiff_t>(split),
        points.end());

    reordered.insert(
        reordered.end(),
        points.begin(),
        points.begin() + static_cast<std::ptrdiff_t>(split));

    return reordered;
}

std::vector<Index> make_tail_front(
    Index node_count,
    Index tail_count) {

    const Index first =
        static_cast<Index>(node_count - tail_count);

    std::vector<Index> front;
    front.reserve(static_cast<std::size_t>(tail_count));

    for (Index internal = first;
         internal < node_count;
         ++internal) {
        front.push_back(internal);
    }

    return front;
}

template <int Dimension>
bool verify_compute_view_front(
    const typename CaseTypes<Dimension>::ComputeView& view,
    const std::vector<typename CaseTypes<Dimension>::Point>& original_points,
    Index tail_count) {

    const Index node_count =
        static_cast<Index>(original_points.size());
    const Index first =
        static_cast<Index>(node_count - tail_count);

    for (Index i = Index{0}; i < tail_count; ++i) {
        const Index expected_internal =
            static_cast<Index>(first + i);

        if (view.stable_internal_node(i) != expected_internal) {
            return false;
        }

        const auto& expected =
            original_points[
                static_cast<std::size_t>(expected_internal)];

        for (Index d = Index{0};
             d < static_cast<Index>(Dimension);
             ++d) {
            if (std::abs(
                    view.nodes().get_data(i, d) -
                    expected[static_cast<Eigen::Index>(d)]) >
                PositionTolerance) {
                return false;
            }
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
            ? 0x3348564945570000ULL
            : 0x3448564945570000ULL;

    return dimension_salt ^
           (static_cast<std::uint64_t>(node_count) << 24u) ^
           (0x9e3779b97f4a7c15ULL *
            static_cast<std::uint64_t>(run + 1u));
}

struct TripleComparisonResult {
    bool view_front_ok = true;
    bool full_reference_valid = true;
    bool view_vs_partial = true;
    bool view_vs_full = true;
    bool partial_vs_full = true;

    [[nodiscard]] bool all_equal() const noexcept {
        return view_front_ok &&
               full_reference_valid &&
               view_vs_partial &&
               view_vs_full &&
               partial_vs_full;
    }
};

template <int Dimension>
TripleComparisonResult run_one_case(
    const std::vector<typename CaseTypes<Dimension>::Point>& original_points,
    Index tail_count,
    bool print_difference) {

    using Types = CaseTypes<Dimension>;
    using RoundState = typename Types::RoundState;
    using ComputeView = typename Types::ComputeView;

    const Index node_count =
        static_cast<Index>(original_points.size());

    const std::vector<Index> preferred_front =
        make_tail_front(node_count, tail_count);

    // ------------------------------------------------------------------
    // Mesh A:
    // Keep HighVoronoi stable internals in original insertion order.
    // Move only the trailing tail to compute-public [0, tail_count).
    // ------------------------------------------------------------------

    auto high_mesh =
        make_high_mesh<Dimension>(original_points);

    RoundState state(
        high_mesh->internal_node_count(),
        high_mesh->internal_boundary().size(),
        static_cast<Index>(node_count - tail_count));

    ComputeView compute_view(
        *high_mesh,
        state,
        preferred_front);

    TripleComparisonResult result;

    result.view_front_ok =
        verify_compute_view_front<Dimension>(
            compute_view,
            original_points,
            tail_count);

    compute_mesh(
        compute_view,
        std::optional<Index>{
            static_cast<Index>(tail_count - Index{1})});

    // ------------------------------------------------------------------
    // Mesh B + C:
    // Physically copy the same trailing tail to the beginning BEFORE
    // constructing a normal VoronoiMesh.
    // B computes only [0, tail_count).
    // C computes all cells.
    // ------------------------------------------------------------------

    const auto reordered_points =
        move_tail_to_front(
            original_points,
            tail_count);

    auto plain_partial =
        make_plain_mesh<Dimension>(reordered_points);

    compute_mesh(
        plain_partial,
        std::optional<Index>{
            static_cast<Index>(tail_count - Index{1})});

    auto plain_full =
        make_plain_mesh<Dimension>(reordered_points);

    compute_mesh(
        plain_full,
        std::nullopt);

    result.full_reference_valid =
        highvoronoi::verify_mesh(
            plain_full,
            VerificationVariance,
            false).valid();

    result.view_vs_partial =
        compare_first_cells(
            compute_view,
            plain_partial,
            tail_count,
            "HighVoronoi view/partial",
            "plain reordered/partial",
            print_difference);

    const bool may_print_second =
        print_difference && result.view_vs_partial;

    result.view_vs_full =
        compare_first_cells(
            compute_view,
            plain_full,
            tail_count,
            "HighVoronoi view/partial",
            "plain reordered/full",
            may_print_second);

    const bool may_print_third =
        may_print_second && result.view_vs_full;

    result.partial_vs_full =
        compare_first_cells(
            plain_partial,
            plain_full,
            tail_count,
            "plain reordered/partial",
            "plain reordered/full",
            may_print_third);

    return result;
}

template <int Dimension>
void run_random_matrix_for_node_count(Index node_count) {
    std::array<std::size_t, Percentages.size()> passed{};
    bool printed_failure = false;

    for (std::size_t run = 0; run < RandomRuns; ++run) {
        const auto points =
            make_random_points<Dimension>(
                node_count,
                random_seed(Dimension, node_count, run));

        for (std::size_t p = 0; p < Percentages.size(); ++p) {
            const unsigned percentage = Percentages[p];
            const Index tail_count =
                selected_count(node_count, percentage);

            const TripleComparisonResult result =
                run_one_case<Dimension>(
                    points,
                    tail_count,
                    !printed_failure);

            if (result.all_equal()) {
                ++passed[p];
                continue;
            }

            if (!printed_failure) {
                std::cerr
                    << "        D=" << Dimension
                    << " N=" << node_count
                    << " tail=" << percentage << "%"
                    << " failed in random run "
                    << (run + 1) << '\n'
                    << "        view front:       "
                    << (result.view_front_ok ? "OK" : "FAIL") << '\n'
                    << "        full reference:   "
                    << (result.full_reference_valid ? "OK" : "FAIL") << '\n'
                    << "        view vs partial:  "
                    << (result.view_vs_partial ? "MATCH" : "MISMATCH") << '\n'
                    << "        view vs full:     "
                    << (result.view_vs_full ? "MATCH" : "MISMATCH") << '\n'
                    << "        partial vs full:  "
                    << (result.partial_vs_full ? "MATCH" : "MISMATCH") << '\n';

                printed_failure = true;
            }
        }
    }

    for (std::size_t p = 0; p < Percentages.size(); ++p) {
        const unsigned percentage = Percentages[p];

        check(
            passed[p] == RandomRuns,
            "D=" + std::to_string(Dimension) +
            " N=" + std::to_string(node_count) +
            " tail " + std::to_string(percentage) +
            "%: HighVoronoi-view partial == reordered partial == reordered full in " +
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
// Historical 27 old + 3 new case from the previous refinement regression.
// Here the 27 old points remain first and the three fixed new points remain
// physically LAST in HighVoronoi. The compute view alone moves them to front.
// The plain comparison meshes physically reorder those same three points.
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
make_historical_27_plus_3_original_order() {
    constexpr Index OldCount = Index{27};
    constexpr Scalar OldMinimumDistance = Scalar{0.055};

    const std::vector<HistoricalPoint> future =
        historical_future_points();

    std::vector<HistoricalPoint> points;
    points.reserve(30);

    std::mt19937_64 random(0x494e4352454d454eULL);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{0.025},
        Scalar{0.975});

    while (points.size() <
           static_cast<std::size_t>(OldCount)) {
        const HistoricalPoint candidate = historical_point(
            coordinate(random),
            coordinate(random),
            coordinate(random));

        if (!far_enough(
                candidate,
                points,
                OldMinimumDistance)) {
            continue;
        }

        if (!far_enough(
                candidate,
                future,
                Scalar{0.085})) {
            continue;
        }

        points.push_back(candidate);
    }

    points.insert(
        points.end(),
        future.begin(),
        future.end());

    return points;
}

void run_historical_case() {
    const auto points =
        make_historical_27_plus_3_original_order();

    const TripleComparisonResult result =
        run_one_case<3>(
            points,
            Index{3},
            true);

    check(
        result.all_equal(),
        "historical 27 old + 3 new: view partial == reordered partial == reordered full");
}

} // namespace

int main() {
    std::cout
        << "============================================================\n"
        << "HighVoronoi compute-view range equivalence test\n"
        << "============================================================\n"
        << "For every case the SAME trailing node subset is tested as:\n"
        << "  A) HighVoronoi stable order + preferred_front view + short range\n"
        << "  B) normal VoronoiMesh physically reordered + short range\n"
        << "  C) normal VoronoiMesh physically reordered + full compute\n"
        << "Only the first selected cells are compared.\n"
        << "3D + 4D; N=30,100,300; tail 5%,10%,20%; 3 random runs.\n"
        << "Plus the historical 27 old + 3 new geometry.\n"
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




