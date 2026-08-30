#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/high_voronoi_mesh.hpp>
#include <highvoronoi/geometry/visible_first_mesh.hpp>

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, Params>;
using HighMesh = highvoronoi::HighVoronoiMesh<Scalar, Dimension, Database>;
using VisibleMesh = highvoronoi::VisibleFirstMesh<HighMesh>;
using Point = HighMesh::NodePoint;
using Boundary = HighMesh::BoundaryType;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
    } else {
        ++failed_checks;
        std::cerr << "    [FAIL] " << description << '\n';
    }
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

Boundary boundary() {
    return Boundary::cuboid(
        point(1, 1, 1),
        point(0, 0, 0),
        std::vector<Index>{Index{0}, Index{1}});
}

} // namespace

int main() {
    HighMesh mesh(
        Index{Dimension},
        boundary(),
        std::in_place,
        std::size_t{1024},
        Params{highvoronoi::DirectHash{1024}});

    const Index v0 = mesh.append_visible_node(point(0.2, 0.2, 0.2));
    const Index v1 = mesh.append_visible_node(point(0.8, 0.2, 0.2));

    HighMesh::ShiftMask shift(
        static_cast<std::size_t>(mesh.external_boundary().size()),
        std::uint8_t{0});
    shift[0] = std::uint8_t{1};
    const Index reference_internal = mesh.append_reference_node(v0, shift);

    const Index v2 = mesh.append_visible_node(point(0.5, 0.8, 0.7));
    (void)v1;
    (void)v2;

    const Index v0_internal = mesh.visible_public_to_internal(Index{0});
    const Index v1_internal = mesh.visible_public_to_internal(Index{1});
    const Index v2_internal = mesh.visible_public_to_internal(Index{2});

    VisibleMesh view(mesh);

    check(view.visible_end() == Index{3},
          "visible_end() is the one-past-end index of the visible prefix");
    check(view.size() == Index{4},
          "VisibleFirstMesh contains all three visible nodes and the reference node");

    check(view.global_internal_node(Index{0}) == v0_internal,
          "first visible node stays first");
    check(view.global_internal_node(Index{1}) == v1_internal,
          "second visible node stays second");
    check(view.global_internal_node(Index{2}) == v2_internal,
          "visible node inserted after an old reference is moved into the visible prefix");
    check(view.global_internal_node(Index{3}) == reference_internal,
          "remaining reference nodes follow the visible prefix in stable insertion order");

    check((view.nodes().node(Index{2}) - point(0.5, 0.8, 0.7)).norm() < 1e-15,
          "reordered public node coordinates belong to the expected visible node");

    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';
    return failed_checks == 0 ? 0 : 1;
}
