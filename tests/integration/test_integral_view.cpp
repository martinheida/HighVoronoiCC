#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/storage/neighbour/neighbour_database.hpp>
#include <highvoronoi/integration/integral_data.hpp>
#include <highvoronoi/integration/integration_view.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
constexpr int Dim = 2;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;
using VertexDatabase = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    Params,
    Dim>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dim, VertexDatabase>;
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
    if (!condition) {
        ++failed_checks;
        std::cerr << "[FAIL] " << message << '\n';
    } else {
        std::cout << "[OK]   " << message << '\n';
    }
}

Point point(double x, double y) {
    Point result;
    result << x, y;
    return result;
}

Mesh make_mesh() {
    Mesh::InternalNodes nodes(Index{5});
    nodes.set(Index{0}, point(0.0, 0.0));
    nodes.set(Index{1}, point(1.0, 0.0));
    nodes.set(Index{2}, point(2.0, 0.0));
    nodes.set(Index{3}, point(3.0, 0.0));
    nodes.set(Index{4}, point(4.0, 0.0));

    Mesh::BoundaryType boundary;
    Point origin = point(0.0, 0.0);
    Point normal = point(-1.0, 0.0);
    boundary.add(highvoronoi::Plane<Dim, Scalar, Index>(
        origin,
        normal,
        highvoronoi::BoundaryCondition::Dirichlet));

    auto database = std::make_shared<VertexDatabase>(
        1024,
        Params{highvoronoi::DirectHash{256}});
    return Mesh(std::move(nodes), std::move(boundary), std::move(database));
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
 * Minimal owner contract used to test the view against the real VoronoiMesh.
 * The production VoronoiIntegral supplies the same mesh/data/tracker access,
 * but the 2026-09-04 source snapshot used for this isolated test predates the
 * mesh-level tracker registration API.
 */
class TestIntegralState final {
public:
    using Mesh = ::Mesh;
    using Index = ::Index;
    using AreaScalar = double;
    using IntegralScalar = double;
    using Address = std::size_t;
    using Data = ::Data;

    TestIntegralState(
        Mesh& mesh,
        NeighbourDatabase& neighbour_database)
        : mesh_(&mesh),
          data_(
              neighbour_database,
              static_cast<std::size_t>(mesh.internal_size()),
              2,
              highvoronoi::IntegralDataOptions{true, true, true, true},
              128,
              128),
          dirty_tracker_(std::make_shared<TestDirtyTracker>(
              static_cast<std::size_t>(mesh.internal_size()))) {}

    [[nodiscard]] Mesh& mesh() noexcept { return *mesh_; }
    [[nodiscard]] const Mesh& mesh() const noexcept { return *mesh_; }
    [[nodiscard]] Data& data() noexcept { return data_; }
    [[nodiscard]] const Data& data() const noexcept { return data_; }

    [[nodiscard]] const std::shared_ptr<TestDirtyTracker>&
    dirty_tracker() const noexcept {
        return dirty_tracker_;
    }

    void synchronize_size() {
        const auto size = static_cast<std::size_t>(mesh_->internal_size());
        data_.resize(size);
        dirty_tracker_->resize(size);
    }

private:
    Mesh* mesh_ = nullptr;
    Data data_;
    std::shared_ptr<TestDirtyTracker> dirty_tracker_;
};

std::size_t publish_cell(
    TestIntegralState& integral,
    NeighbourDatabase& neighbour_database,
    Index stable_cell,
    const std::vector<Index>& neighbours,
    double volume,
    const std::vector<double>& area,
    const std::vector<double>& bulk,
    const std::vector<double>& interface_integral) {
    check(std::is_sorted(neighbours.begin(), neighbours.end()),
          "test fixture neighbour record is sorted in stable numbering");

    const std::size_t address = neighbour_database.push(neighbours);
    integral.data().set_neighbour_address(stable_cell, address);

    Data::CellData cell;
    check(integral.data().prepare_cell(stable_cell, cell),
          "fixture can read its just-published neighbour record");
    cell.set_volume(volume);
    cell.area() = area;
    cell.bulk_integral() = bulk;
    cell.interface_integral() = interface_integral;
    integral.data().write_cell(stable_cell, cell);
    return address;
}

void test_view_order_and_integral_alignment() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] VoronoiIntegrationView ordering and data alignment\n";
    std::cout << "============================================================\n";

    Mesh mesh = make_mesh();
    NeighbourDatabase neighbour_database(128);
    TestIntegralState integral(mesh, neighbour_database);

    const Index boundary0 =
        (std::numeric_limits<Index>::max)() - Index{1};

    // Stable cell 0 has a historical neighbour entry to stable cell 2.  Cell 2
    // is deleted below.  The view must drop exactly that neighbour occurrence
    // together with its area/interface data, while preserving duplicate cell 1.
    (void)publish_cell(
        integral,
        neighbour_database,
        Index{0},
        std::vector<Index>{Index{1}, Index{1}, Index{2}, Index{4}, boundary0},
        7.0,
        std::vector<double>{10.0, 11.0, 12.0, 13.0, 14.0},
        std::vector<double>{70.0, 71.0},
        std::vector<double>{
            100.0, 101.0,
            110.0, 111.0,
            120.0, 121.0,
            130.0, 131.0,
            140.0, 141.0});

    (void)publish_cell(
        integral,
        neighbour_database,
        Index{1},
        std::vector<Index>{Index{0}, Index{3}},
        8.0,
        std::vector<double>{20.0, 21.0},
        std::vector<double>{80.0, 81.0},
        std::vector<double>{200.0, 201.0, 210.0, 211.0});

    (void)publish_cell(
        integral,
        neighbour_database,
        Index{3},
        std::vector<Index>{Index{0}, Index{1}},
        9.0,
        std::vector<double>{30.0, 31.0},
        std::vector<double>{90.0, 91.0},
        std::vector<double>{300.0, 301.0, 310.0, 311.0});

    // stable cell 4 deliberately has no integral neighbour snapshot -> NEW.
    integral.dirty_tracker()->set_dirty(Index{1}, true); // DIRTY OLD

    const std::size_t deleted = mesh.erase_nodes_if(
        [](Index public_index, const Point&) {
            return public_index == Index{2};
        });
    check(deleted == 1, "one wrapped Voronoi node is deleted");
    check(mesh.size() == Index{4}, "wrapped public numbering compacts to four cells");
    check(mesh.internal_size() == Index{5}, "stable internal numbering remains five slots");

    highvoronoi::VoronoiIntegrationView<TestIntegralState> view(integral);

    check(view.size() == 4, "integration view contains every current public cell");
    check(view.new_count() == 1, "integration view finds one NEW cell");
    check(view.dirty_old_count() == 1, "integration view finds one DIRTY OLD cell");
    check(view.clean_old_count() == 2, "integration view leaves two CLEAN OLD cells");
    check(view.update_count() == 2, "only NEW + DIRTY OLD form the update prefix");

    // Wrapped public indices after deletion are [stable 0, stable 1, stable 3,
    // stable 4].  Therefore the expected integration order is
    // [stable 4 NEW, stable 1 DIRTY, stable 0 CLEAN, stable 3 CLEAN].
    check(view.stable_internal_index(Index{0}) == Index{4},
          "NEW stable cell 4 is first");
    check(view.stable_internal_index(Index{1}) == Index{1},
          "DIRTY OLD stable cell 1 follows NEW cells");
    check(view.stable_internal_index(Index{2}) == Index{0},
          "first CLEAN OLD cell follows the update prefix");
    check(view.stable_internal_index(Index{3}) == Index{3},
          "second CLEAN OLD cell remains in stable wrapped order");

    check(view.is_new(Index{0}), "view cell 0 is classified NEW");
    check(view.is_dirty_old(Index{1}), "view cell 1 is classified DIRTY OLD");
    check(!view.is_dirty_old(Index{2}), "clean cell is not classified DIRTY OLD");

    // The same permutation must be visible through the actual mesh facade.
    check(view.mesh().wrapped_public_index(Index{0}) == Index{3},
          "mesh view cell 0 resolves to wrapped public cell 3 (stable 4)");
    check(view.mesh().wrapped_public_index(Index{1}) == Index{1},
          "mesh view cell 1 resolves to wrapped public cell 1");
    check(view.mesh().wrapped_public_index(Index{2}) == Index{0},
          "mesh view cell 2 resolves to wrapped public cell 0");
    check(view.mesh().wrapped_public_index(Index{3}) == Index{2},
          "mesh view cell 3 resolves to wrapped public cell 2 (stable 3)");

    check(view.mesh().nodes().get_data(Index{0}, Index{0}) == 4.0,
          "geometry node access follows the integration permutation");
    check(view.mesh().nodes().get_data(Index{2}, Index{0}) == 0.0,
          "clean-cell geometry remains addressable behind update prefix");

    typename decltype(view)::CellData data;
    data.reserve(8, 2);
    const bool complete = view.read_cell(Index{2}, data); // stable cell 0
    check(complete, "committed integral cell reads completely through view");
    check(data.volume() == 7.0, "volume follows stable cell identity through view");
    check(data.bulk_integral() == std::vector<double>({70.0, 71.0}),
          "bulk integral follows stable cell identity through view");

    // Raw stable neighbours were [1,1,2(deleted),4,boundary0].
    // Under the temporary view: 4->0, 1->1, deleted 2 disappears,
    // boundary0->4.  Sorting presentation neighbours must shuffle all aligned
    // interface payloads by the identical ordinal permutation.
    check(data.neighbours() == std::vector<Index>({0, 1, 1, 4}),
          "view translates, sorts and preserves duplicate neighbours");
    check(data.area() == std::vector<double>({13.0, 10.0, 11.0, 14.0}),
          "area entries remain aligned with reordered neighbour occurrences");
    check(data.interface_integral() == std::vector<double>({
              130.0, 131.0,
              100.0, 101.0,
              110.0, 111.0,
              140.0, 141.0}),
          "interface-integral blocks follow exactly the same neighbour permutation");

    check(std::count(data.neighbours().begin(), data.neighbours().end(), Index{1}) == 2,
          "duplicate neighbour multiplicity survives the integration view");
    check(std::find(data.neighbours().begin(), data.neighbours().end(), Index{4}) !=
              data.neighbours().end(),
          "stable boundary encoding becomes view-size plus plane index");

    typename decltype(view)::CellData dirty_data;
    check(view.read_cell(Index{1}, dirty_data),
          "DIRTY OLD committed data remain readable before recomputation");
    check(dirty_data.neighbours() == std::vector<Index>({2, 3}),
          "DIRTY OLD neighbour indices are translated into the same view world");
    check(dirty_data.area() == std::vector<double>({20.0, 21.0}),
          "DIRTY OLD area alignment survives view translation");
}

} // namespace

int main() {
    // This test exercises the complete presentation contract, not numerical integration:
    // stable IntegralData -> reordered VoronoiIntegrationView -> aligned readback.
    test_view_order_and_integral_alignment();

    std::cout << "\n============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    std::cout << "============================================================\n";
    return failed_checks == 0 ? 0 : 1;
}
