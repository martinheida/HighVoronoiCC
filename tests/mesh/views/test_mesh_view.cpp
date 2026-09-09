
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/mesh_view.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint16_t;
constexpr int Dimension = 2;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::detail::EmptyLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Switch = highvoronoi::SwitchView<Index>;
using View = highvoronoi::ReorderedMeshView<Mesh, Switch>;

using Point = Mesh::VertexPoint;
using Nodes = Mesh::InternalNodes;
using Boundary = Mesh::BoundaryType;
using Sigma = Mesh::Sigma;
using Address = Mesh::Address;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;
constexpr Scalar tolerance = 1.0e-12;

void check(bool condition, std::string_view description) {
    ++performed_checks;

    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

[[nodiscard]] bool close(Scalar left, Scalar right) {
    return std::abs(left - right) <= tolerance;
}

[[nodiscard]] Point point(Scalar x, Scalar y) {
    Point result;
    result << x, y;
    return result;
}

void print_point(const Point& value) {
    std::cout << '(' << value[0] << ", " << value[1] << ')';
}

void print_sigma(const Sigma& sigma) {
    std::cout << '[';
    for (std::size_t position = 0; position < sigma.size(); ++position) {
        if (position != 0) {
            std::cout << ", ";
        }
        std::cout << static_cast<unsigned int>(sigma[position]);
    }
    std::cout << ']';
}

template <class MeshLike>
void print_node_with_vertices(
    const MeshLike& mesh,
    Index node,
    std::string_view heading) {
    std::cout << "\n" << heading << '\n';
    std::cout << "node " << static_cast<unsigned int>(node) << " = ";
    print_point(mesh.nodes().node(node));
    std::cout << '\n';

    std::size_t vertex_count = 0;
    for (const auto& vertex : mesh.vertices(node)) {
        ++vertex_count;
        std::cout << "    address " << vertex.address << ": sigma = ";
        print_sigma(vertex.sigma);
        std::cout << ", r = ";
        print_point(vertex.position);
        std::cout << '\n';
    }

    if (vertex_count == 0) {
        std::cout << "    no vertices\n";
    }
}

template <class MeshLike>
[[nodiscard]] bool contains_expected_vertex(
    const MeshLike& mesh,
    Index node,
    Address expected_address,
    const Sigma& expected_sigma,
    const Point& expected_position) {
    for (const auto& vertex : mesh.vertices(node)) {
        if (vertex.address != expected_address) {
            continue;
        }

        return vertex.sigma == expected_sigma &&
               close(vertex.position[0], expected_position[0]) &&
               close(vertex.position[1], expected_position[1]);
    }

    return false;
}

Nodes make_nodes() {
    Nodes nodes(Index{4});
    nodes.set(Index{0}, point(0.0, 0.0));
    nodes.set(Index{1}, point(1.0, 0.0));
    nodes.set(Index{2}, point(2.0, 0.0));
    nodes.set(Index{3}, point(3.0, 0.0));
    return nodes;
}

Boundary make_boundary() {
    return Boundary::cuboid(
        point(4.0, 2.0),
        std::vector<Index>{});
}

} // namespace

int main() {
    // ---------------------------------------------------------------------
    // Construct the ordinary Voronoi mesh.
    // ---------------------------------------------------------------------

    const DatabaseParameters database_parameters{
        highvoronoi::DirectHash{64}
    };

    auto database = std::make_shared<Database>(
        std::size_t{64},
        database_parameters);

    Mesh mesh{
        make_nodes(),
        make_boundary(),
        database
    };

    std::cout << "========================================\n";
    std::cout << "ORIGINAL VORONOI MESH\n";
    std::cout << "========================================\n";
    std::cout << "ordinary node indices: 0, 1, 2, 3\n";
    std::cout << "boundary indices:      4, 5, 6, 7\n";

    // Every signature contains at least one boundary index.
    const Point position_a = point(0.4, 0.2);
    const Point position_b = point(1.5, 0.5);
    const Point position_c = point(2.6, 0.7);
    const Point position_d = point(3.2, 1.0);

    const Sigma original_sigma_a{Index{0}, Index{1}, Index{4}};
    const Sigma original_sigma_b{Index{0}, Index{3}, Index{5}};
    const Sigma original_sigma_c{Index{2}, Index{3}, Index{6}};
    const Sigma original_sigma_d{Index{3}, Index{6}, Index{7}};

    const Address address_a = mesh.store_vertex(
        position_a,
        original_sigma_a);
    const Address address_b = mesh.store_vertex(
        position_b,
        original_sigma_b);
    const Address address_c = mesh.store_vertex(
        position_c,
        original_sigma_c);
    const Address address_d = mesh.store_vertex(
        position_d,
        original_sigma_d);

    std::cout << "\nstored vertices:\n";
    std::cout << "    A: address " << address_a << ", sigma = ";
    print_sigma(original_sigma_a);
    std::cout << ", r = ";
    print_point(position_a);
    std::cout << '\n';

    std::cout << "    B: address " << address_b << ", sigma = ";
    print_sigma(original_sigma_b);
    std::cout << ", r = ";
    print_point(position_b);
    std::cout << '\n';

    std::cout << "    C: address " << address_c << ", sigma = ";
    print_sigma(original_sigma_c);
    std::cout << ", r = ";
    print_point(position_c);
    std::cout << '\n';

    std::cout << "    D: address " << address_d << ", sigma = ";
    print_sigma(original_sigma_d);
    std::cout << ", r = ";
    print_point(position_d);
    std::cout << '\n';

    print_node_with_vertices(
        mesh,
        Index{0},
        "original mesh: node 0 and its vertices");
    print_node_with_vertices(
        mesh,
        Index{3},
        "original mesh: node 3 and its vertices");

    check(
        contains_expected_vertex(
            mesh,
            Index{0},
            address_a,
            original_sigma_a,
            position_a),
        "original node 0 contains vertex A with sigma [0,1,4]");
    check(
        contains_expected_vertex(
            mesh,
            Index{0},
            address_b,
            original_sigma_b,
            position_b),
        "original node 0 contains vertex B with sigma [0,3,5]");
    check(
        contains_expected_vertex(
            mesh,
            Index{3},
            address_b,
            original_sigma_b,
            position_b),
        "original node 3 also contains vertex B");
    check(
        contains_expected_vertex(
            mesh,
            Index{3},
            address_c,
            original_sigma_c,
            position_c),
        "original node 3 contains vertex C with sigma [2,3,6]");
    check(
        contains_expected_vertex(
            mesh,
            Index{3},
            address_d,
            original_sigma_d,
            position_d),
        "original node 3 contains vertex D with sigma [3,6,7]");

    // ---------------------------------------------------------------------
    // Move the last ordinary node to the front.
    //
    // SwitchView(3,3) maps the original order
    //
    //     0, 1, 2, 3
    //
    // to the view order
    //
    //     3, 0, 1, 2.
    //
    // Therefore view node 0 is original node 3 and view node 1 is original
    // node 0. Boundary indices 4,5,6,7 remain unchanged.
    // ---------------------------------------------------------------------

    const Switch switch_view(Index{3}, Index{3});
    View view(mesh, switch_view);

    std::cout << "\n========================================\n";
    std::cout << "REORDERED MESH VIEW\n";
    std::cout << "========================================\n";
    std::cout << "SwitchView(3, 3) moves the last node to the front.\n";
    std::cout << "original index: 0  1  2  3\n";
    std::cout << "view index:     1  2  3  0\n";
    std::cout << "view order:     3  0  1  2\n";

    // The requested order is deliberate: view node 1 corresponds to original
    // node 0, then view node 0 corresponds to original node 3.
    print_node_with_vertices(
        view,
        Index{1},
        "mesh view: node 1 (= original node 0) and its vertices");
    print_node_with_vertices(
        view,
        Index{0},
        "mesh view: node 0 (= original node 3) and its vertices");

    const Sigma view_sigma_a{Index{1}, Index{2}, Index{4}};
    const Sigma view_sigma_b{Index{0}, Index{1}, Index{5}};
    const Sigma view_sigma_c{Index{0}, Index{3}, Index{6}};
    const Sigma view_sigma_d{Index{0}, Index{6}, Index{7}};

    check(
        close(view.nodes().node(Index{1})[0], 0.0) &&
            close(view.nodes().node(Index{1})[1], 0.0),
        "view node 1 has the coordinates of original node 0");
    check(
        close(view.nodes().node(Index{0})[0], 3.0) &&
            close(view.nodes().node(Index{0})[1], 0.0),
        "view node 0 has the coordinates of original node 3");

    check(
        contains_expected_vertex(
            view,
            Index{1},
            address_a,
            view_sigma_a,
            position_a),
        "view node 1 exposes vertex A as sigma [1,2,4]");
    check(
        contains_expected_vertex(
            view,
            Index{1},
            address_b,
            view_sigma_b,
            position_b),
        "view node 1 exposes vertex B as sigma [0,1,5]");
    check(
        contains_expected_vertex(
            view,
            Index{0},
            address_b,
            view_sigma_b,
            position_b),
        "view node 0 also exposes vertex B as sigma [0,1,5]");
    check(
        contains_expected_vertex(
            view,
            Index{0},
            address_c,
            view_sigma_c,
            position_c),
        "view node 0 exposes vertex C as sigma [0,3,6]");
    check(
        contains_expected_vertex(
            view,
            Index{0},
            address_d,
            view_sigma_d,
            position_d),
        "view node 0 exposes vertex D as sigma [0,6,7]");

    check(
        view.wrapped_public_index(Index{0}) == Index{3} &&
            view.wrapped_public_index(Index{1}) == Index{0},
        "the view maps node 0 to original 3 and node 1 to original 0");

    // ---------------------------------------------------------------------
    // Delete one vertex through the reordered view.
    //
    // Vertex B is visible in the reordered numbering as [0,1,5]. The filter
    // therefore receives and tests the view signature, while the shared
    // database stores the corresponding stable internal signature.
    // ---------------------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << "FILTER THROUGH THE REORDERED MESH VIEW\n";
    std::cout << "========================================\n";
    std::cout << "delete vertex B through the view:\n";
    std::cout << "    view sigma     = ";
    print_sigma(view_sigma_b);
    std::cout << '\n';
    std::cout << "    original sigma = ";
    print_sigma(original_sigma_b);
    std::cout << '\n';

    const auto filter_result = view.filter(
        highvoronoi::NoNodeFilter{},
        [&view_sigma_b](
            const auto&,
            const Sigma& public_sigma,
            const Point&) {
            return public_sigma == view_sigma_b;
        });

    std::cout << "filter result:\n";
    std::cout << "    deleted nodes:    " << filter_result.first << '\n';
    std::cout << "    deleted vertices: " << filter_result.second << '\n';

    print_node_with_vertices(
        view,
        Index{1},
        "mesh view after filtering: node 1 (= original node 0)");
    print_node_with_vertices(
        view,
        Index{0},
        "mesh view after filtering: node 0 (= original node 3)");

    check(
        filter_result.first == 0,
        "the view filter deletes no nodes");
    check(
        filter_result.second == 1,
        "the view filter deletes exactly vertex B");
    check(
        !view.contains_vertex(view_sigma_b),
        "vertex B is absent under its view signature [0,1,5]");
    check(
        !mesh.contains_vertex(original_sigma_b),
        "vertex B is also absent from the underlying mesh as [0,3,5]");
    check(
        !contains_expected_vertex(
            mesh,
            Index{0},
            address_b,
            original_sigma_b,
            position_b) &&
        !contains_expected_vertex(
            mesh,
            Index{3},
            address_b,
            original_sigma_b,
            position_b),
        "the underlying node lists no longer yield tombstoned vertex B");

    // ---------------------------------------------------------------------
    // Store a new vertex through the reordered view.
    //
    // View nodes 0 and 1 correspond to original nodes 3 and 0. Therefore the
    // view signature [0,1,4] must appear in the wrapped VoronoiMesh as the
    // sorted original signature [0,3,4]. Boundary index 4 is unchanged.
    // ---------------------------------------------------------------------

    const Point position_e = point(2.1, 1.3);
    const Sigma view_sigma_e{Index{0}, Index{1}, Index{4}};
    const Sigma original_sigma_e{Index{0}, Index{3}, Index{4}};

    std::cout << "\n========================================\n";
    std::cout << "STORE THROUGH THE REORDERED MESH VIEW\n";
    std::cout << "========================================\n";
    std::cout << "insert new vertex E through the view:\n";
    std::cout << "    view sigma     = ";
    print_sigma(view_sigma_e);
    std::cout << '\n';
    std::cout << "    expected original sigma = ";
    print_sigma(original_sigma_e);
    std::cout << '\n';
    std::cout << "    r = ";
    print_point(position_e);
    std::cout << '\n';

    const Address address_e = view.store_vertex(
        position_e,
        view_sigma_e);

    std::cout << "    stored database address = " << address_e << '\n';

    std::cout << "\nNow inspect the original VoronoiMesh directly.\n";
    print_node_with_vertices(
        mesh,
        Index{0},
        "original mesh after view filter and insertion: node 0");
    print_node_with_vertices(
        mesh,
        Index{3},
        "original mesh after view filter and insertion: node 3");

    check(
        address_e != Address{0},
        "storing vertex E through the view creates a database record");
    check(
        view.contains_vertex(view_sigma_e),
        "the new vertex is found through the view as [0,1,4]");
    check(
        mesh.contains_vertex(original_sigma_e),
        "the underlying mesh finds the new vertex as [0,3,4]");
    check(
        contains_expected_vertex(
            mesh,
            Index{0},
            address_e,
            original_sigma_e,
            position_e),
        "original node 0 yields new vertex E with sigma [0,3,4]");
    check(
        contains_expected_vertex(
            mesh,
            Index{3},
            address_e,
            original_sigma_e,
            position_e),
        "original node 3 also yields new vertex E with sigma [0,3,4]");
    check(
        contains_expected_vertex(
            mesh,
            Index{0},
            address_a,
            original_sigma_a,
            position_a) &&
        contains_expected_vertex(
            mesh,
            Index{3},
            address_c,
            original_sigma_c,
            position_c) &&
        contains_expected_vertex(
            mesh,
            Index{3},
            address_d,
            original_sigma_d,
            position_d),
        "all unrelated original vertices remain intact");

    std::cout << "\n========================================\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}


