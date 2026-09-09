
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/storage/neighbour/neighbour_database.hpp>
#include <highvoronoi/integration/high_voronoi_integration_view.hpp>
#include <highvoronoi/integration/integral_data.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
constexpr int Dim = 3;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;
using VertexDatabase = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    Params,
    Dim>;
using Mesh = highvoronoi::HighVoronoiMesh<Scalar, Dim, VertexDatabase>;
using Point = Mesh::NodePoint;
using NeighbourDatabase = highvoronoi::detail::NeighbourDatabase<
    highvoronoi::EmptyLock,
    Index>;
using Data = highvoronoi::IntegralData<
    highvoronoi::EmptyLock,
    Index,
    double,
    double,
    NeighbourDatabase>;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view message) {
    ++performed_checks;
    if (condition) {
        std::cout << "[OK]   " << message << '\n';
    } else {
        ++failed_checks;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

Mesh::BoundaryType boundary() {
    return Mesh::BoundaryType::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{Index{0}, Index{1}});
}

class TestDirtyTracker final {
public:
    explicit TestDirtyTracker(std::size_t size)
        : dirty_(size, std::uint8_t{0}) {}

    [[nodiscard]] bool dirty(std::size_t index) const {
        return dirty_.at(index) != std::uint8_t{0};
    }

    void set_dirty(std::size_t index, bool value = true) {
        dirty_.at(index) = value ? std::uint8_t{1} : std::uint8_t{0};
    }

    void resize(std::size_t size) {
        dirty_.resize(size, std::uint8_t{0});
    }

private:
    std::vector<std::uint8_t> dirty_;
};

/**
 * Minimal persistent-integral owner used to exercise the real HighVoronoiMesh
 * and its data_mesh() geometry against the Stage-4 IntegralData contract.
 * Production code uses VoronoiIntegral<HighVoronoiMesh<...>>; this local owner
 * avoids depending on the newer mesh-level tracker API in the 2026-09-04
 * source snapshot used for this isolated compilation.
 */
class TestIntegralState final {
public:
    using Mesh = ::Mesh;
    using Index = ::Index;
    using AreaScalar = double;
    using IntegralScalar = double;
    using Address = std::size_t;
    using Data = ::Data;

    TestIntegralState(Mesh& mesh, NeighbourDatabase& neighbour_database)
        : mesh_(&mesh),
          data_(
              neighbour_database,
              static_cast<std::size_t>(mesh.internal_node_count()),
              2,
              highvoronoi::IntegralDataOptions{true, true, true, true},
              128,
              128),
          dirty_tracker_(std::make_shared<TestDirtyTracker>(
              static_cast<std::size_t>(mesh.internal_node_count()))) {}

    [[nodiscard]] Mesh& mesh() noexcept { return *mesh_; }
    [[nodiscard]] const Mesh& mesh() const noexcept { return *mesh_; }
    [[nodiscard]] Data& data() noexcept { return data_; }
    [[nodiscard]] const Data& data() const noexcept { return data_; }
    [[nodiscard]] const std::shared_ptr<TestDirtyTracker>&
    dirty_tracker() const noexcept { return dirty_tracker_; }

    void synchronize_size() {
        const auto size = static_cast<std::size_t>(mesh_->internal_node_count());
        data_.resize(size);
        dirty_tracker_->resize(size);
    }

private:
    Mesh* mesh_ = nullptr;
    Data data_;
    std::shared_ptr<TestDirtyTracker> dirty_tracker_;
};

void publish_cell(
    TestIntegralState& integral,
    NeighbourDatabase& neighbour_database,
    Index stable_cell,
    const std::vector<Index>& neighbours,
    double volume,
    const std::vector<double>& area,
    const std::vector<double>& bulk,
    const std::vector<double>& interface_integral) {
    check(std::is_sorted(neighbours.begin(), neighbours.end()),
          "fixture neighbour record is sorted in stable numbering");

    const std::size_t address = neighbour_database.push(neighbours);
    integral.data().set_neighbour_address(stable_cell, address);

    Data::CellData cell;
    check(integral.data().prepare_cell(stable_cell, cell),
          "fixture reads its immutable neighbour record");
    cell.set_volume(volume);
    cell.area() = area;
    cell.bulk_integral() = bulk;
    cell.interface_integral() = interface_integral;
    integral.data().write_cell(stable_cell, cell);
}

void test_high_view_order_projection_and_integral_alignment() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] HighVoronoiIntegrationView ordering and reference geometry\n";
    std::cout << "============================================================\n";

    Mesh mesh(
        Index{Dim},
        boundary(),
        std::in_place,
        std::size_t{1024},
        Params{highvoronoi::DirectHash{1024}});

    // Stable insertion order deliberately interleaves visible nodes and
    // periodic references.  The integration view must pull every visible cell
    // in front while retaining reference nodes as distinct shifted generators.
    const Index v0 = mesh.append_visible_node(point(0.20, 0.20, 0.30)); // stable 0
    const Index v1 = mesh.append_visible_node(point(0.75, 0.20, 0.35)); // stable 1

    Mesh::ShiftMask shift0(
        static_cast<std::size_t>(mesh.external_boundary().size()),
        std::uint8_t{0});
    shift0[0] = std::uint8_t{1};
    const Index r0 = mesh.append_reference_node(v0, shift0);             // stable 2

    const Index v2 = mesh.append_visible_node(point(0.35, 0.75, 0.55)); // stable 3

    Mesh::ShiftMask shift1(
        static_cast<std::size_t>(mesh.external_boundary().size()),
        std::uint8_t{0});
    shift1[1] = std::uint8_t{1};
    const Index r1 = mesh.append_reference_node(v1, shift1);             // stable 4

    const Index v3 = mesh.append_visible_node(point(0.60, 0.65, 0.80)); // stable 5
    (void)v2;
    (void)v3;

    check(r0 == Index{2} && r1 == Index{4},
          "fixture interleaves two stable periodic references among visible nodes");
    check(mesh.visible_public_count() == Index{4},
          "HighVoronoi public mesh contains four visible cells");
    check(mesh.internal_node_count() == Index{6},
          "HighVoronoi stable mesh contains four visible plus two reference nodes");

    NeighbourDatabase neighbour_database(128);
    TestIntegralState integral(mesh, neighbour_database);
    const Index boundary0 = (std::numeric_limits<Index>::max)() - Index{1};

    // v0, v1 and v2 have old integral snapshots. v3 deliberately has none -> NEW.
    publish_cell(
        integral,
        neighbour_database,
        Index{0},
        std::vector<Index>{Index{1}, Index{2}, Index{4}, boundary0},
        10.0,
        std::vector<double>{100.0, 101.0, 102.0, 103.0},
        std::vector<double>{1000.0, 1001.0},
        std::vector<double>{
            10.0, 11.0,
            20.0, 21.0,
            30.0, 31.0,
            40.0, 41.0});

    publish_cell(
        integral,
        neighbour_database,
        Index{1},
        std::vector<Index>{Index{0}, Index{3}, Index{4}},
        20.0,
        std::vector<double>{200.0, 201.0, 202.0},
        std::vector<double>{2000.0, 2001.0},
        std::vector<double>{50.0, 51.0, 60.0, 61.0, 70.0, 71.0});

    publish_cell(
        integral,
        neighbour_database,
        Index{3},
        std::vector<Index>{Index{0}, Index{2}, Index{4}, boundary0},
        30.0,
        std::vector<double>{300.0, 301.0, 302.0, 303.0},
        std::vector<double>{3000.0, 3001.0},
        std::vector<double>{
            80.0, 81.0,
            90.0, 91.0,
            100.0, 101.0,
            110.0, 111.0});

    // Direct dirtiness of v1 and reference-only dirtiness of r0.  The
    // reference bit is deliberately irrelevant for integral scheduling:
    // HighVoronoiMesh is already responsible for propagating every physically
    // relevant change to the corresponding visible cell.
    integral.dirty_tracker()->set_dirty(Index{1}, true);
    integral.dirty_tracker()->set_dirty(Index{2}, true);

    highvoronoi::HighVoronoiIntegrationView<TestIntegralState> view(integral);

    check(view.size() == 6,
          "integration geometry contains every active visible/reference node");
    check(view.visible_count() == 4 && view.reference_count() == 2,
          "view distinguishes four integral cells from two geometry-only references");
    check(view.new_count() == 1,
          "visible cell without an integral snapshot is classified NEW");
    check(view.dirty_old_count() == 1,
          "only directly dirty visible integral cells are scheduled");
    check(view.clean_old_count() == 2,
          "reference-only dirtiness does not make its visible owner DIRTY OLD");
    check(view.update_count() == 2,
          "only NEW + directly DIRTY OLD visible cells form the update prefix");

    // Expected order:
    //   visible stable 5 NEW,
    //   visible stable 1 DIRTY directly,
    //   visible stable 0 and 3 CLEAN,
    //   references stable 2 and 4.
    const std::vector<Index> expected_stable{5, 1, 0, 3, 2, 4};
    for (std::size_t k = 0; k < expected_stable.size(); ++k) {
        check(view.stable_internal_index(static_cast<Index>(k)) == expected_stable[k],
              "integration order matches NEW/DIRTY/CLEAN/REFERENCE partition");
    }

    check(view.is_new(Index{0}), "first visible entry is NEW");
    check(view.is_dirty_old(Index{1}) && !view.is_dirty_old(Index{2}),
          "only the directly dirty visible entry is DIRTY OLD");
    check(view.is_reference(Index{4}) && view.is_reference(Index{5}),
          "reference suffix is explicitly classified as geometry-only");
    check(view.visible_owner_cell(Index{4}) == Index{2},
          "reference stable 2 resolves to visible owner stable 0 in the same view");
    check(view.visible_owner_cell(Index{5}) == Index{1},
          "reference stable 4 resolves to visible owner stable 1 in the same view");
    check(view.was_dirty_stable(Index{2}),
          "raw tracker still exposes the reference-only dirty bit");
    check(!view.was_dirty_integral_cell(Index{2}),
          "reference-only dirty state does not invalidate the visible integral owner");
    check(view.was_dirty_integral_cell(Index{4}),
          "a reference to a dirty visible owner sees that physical cell as dirty");

    // Geometry uses the actual shifted reference coordinate, not projected
    // visible coordinates.  This is the essential HighVoronoi integration-view contract.
    const Point reference0 = mesh.internal_nodes().node(Index{2});
    const Point reference1 = mesh.internal_nodes().node(Index{4});
    check((view.mesh().nodes().node(Index{4}) - reference0).norm() < 1e-15,
          "first reference keeps its actual shifted Euclidean coordinate");
    check((view.mesh().nodes().node(Index{5}) - reference1).norm() < 1e-15,
          "second reference keeps its actual shifted Euclidean coordinate");
    check((view.mesh().nodes().node(Index{2}) - mesh.internal_nodes().node(Index{0})).norm() < 1e-15,
          "visible owner coordinate remains independently addressable");

    typename decltype(view)::CellData data;
    data.reserve(8, 2);
    check(view.read_cell(Index{3}, data),
          "CLEAN visible cell reads its committed integral through HighVoronoi view");
    check(data.volume() == 30.0 &&
              data.bulk_integral() == std::vector<double>({3000.0, 3001.0}),
          "cell-wide integral values follow stable visible identity");

    // Raw stable neighbours [0,2,4,boundary0] become
    // [view 2, view 4, view 5, view-size 6].  Reference nodes remain distinct.
    check(data.neighbours() == std::vector<Index>({2, 4, 5, 6}),
          "integral neighbour presentation preserves distinct periodic reference nodes");
    check(data.area() == std::vector<double>({300.0, 301.0, 302.0, 303.0}),
          "areas remain ordinally aligned after HighVoronoi neighbour translation");
    check(data.interface_integral() == std::vector<double>({
              80.0, 81.0,
              90.0, 91.0,
              100.0, 101.0,
              110.0, 111.0}),
          "vector interface-integral blocks follow the same reference-aware permutation");

    typename decltype(view)::CellData reference_data;
    check(view.read_cell(Index{4}, reference_data),
          "reading integral data through a reference resolves to its visible owner");
    check(reference_data.volume() == 10.0,
          "reference read returns visible owner's cell-wide integral");
    check(reference_data.neighbours() == std::vector<Index>({1, 4, 5, 6}),
          "reference read still presents the owner's raw interfaces in full geometry numbering");

    check(view.external_boundary().size() == mesh.external_boundary().size() &&
              view.internal_boundary().size() == mesh.internal_boundary().size(),
          "both external and internal HighVoronoi boundaries are exposed explicitly");


    // Parallel-integration seam: assign only the second global update cell to
    // one worker. The worker must move that cell to its own prefix while
    // retaining every other global update cell as geometry/update context.
    auto worker = view.make_worker_view(1, 2);
    check(worker->update_count() == 1,
          "HighVoronoi worker owns exactly its assigned update slice");
    check(worker->stable_internal_index(Index{0}) == Index{1} &&
              worker->stable_internal_index(Index{1}) == Index{5},
          "worker moves its assigned stable cell first and keeps the other global update next");
    check(worker->is_update_cell(Index{0}) &&
              worker->is_update_cell(Index{1}),
          "worker recognizes assigned and cross-worker cells from the same global update snapshot");
    check(!worker->is_update_cell(Index{2}) &&
              !worker->is_update_cell(Index{3}),
          "CLEAN visible context remains outside the global update set");
    check(worker->is_reference(Index{4}) && worker->is_reference(Index{5}),
          "worker keeps periodic references behind the complete visible block");

    // Geometry remains unfolded, but function evaluation must be wrapped back
    // to the external fundamental domain in every worker view as well.
    Point wrapped_reference = worker->mesh().nodes().node(Index{4});
    const Point unfolded_reference = wrapped_reference;
    worker->wrap_evaluation_point(wrapped_reference);
    check((unfolded_reference - wrapped_reference).norm() > 1e-12,
          "periodic reference is genuinely unfolded before integrand evaluation");
    check((wrapped_reference - mesh.internal_nodes().node(Index{0})).norm() < 1e-12,
          "worker wraps periodic reference evaluation back to its visible fundamental-domain point");
}

} // namespace

int main() {
    test_high_view_order_projection_and_integral_alignment();

    std::cout << "\n============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    std::cout << "============================================================\n";
    return failed_checks == 0 ? 0 : 1;
}
