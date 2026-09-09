
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint16_t;

constexpr int Dimension = 2;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Boundary = Mesh::BoundaryType;
using Point = Mesh::VertexPoint;
using Sigma = Mesh::Sigma;
using Address = Mesh::Address;

int failed_checks = 0;

void check(bool condition, std::string_view description) {
    if (condition) {
        std::cout << "[OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "[FAIL] " << description << '\n';
}

Point point(Scalar x, Scalar y) {
    Point result;
    result << x, y;
    return result;
}

bool same_point(const Point& left, const Point& right) {
    constexpr Scalar tolerance = 1.0e-12;
    return std::abs(left[0] - right[0]) <= tolerance &&
           std::abs(left[1] - right[1]) <= tolerance;
}

bool mesh_contains_vertex(
    const Mesh& mesh,
    const Sigma& expected_sigma,
    const Point& expected_position) {
    if (expected_sigma.empty() || expected_sigma.front() >= mesh.size()) {
        return false;
    }

    for (const auto& vertex : mesh.vertices(expected_sigma.front())) {
        if (vertex.sigma == expected_sigma &&
            same_point(vertex.position, expected_position)) {
            return true;
        }
    }

    return false;
}

bool database_record_is_active(
    const Database& database,
    Address address) {
    Point position;
    Sigma sigma;
    database.read(address, position, sigma);
    return !sigma.empty();
}

std::size_t active_vertex_count(const Mesh& mesh) {
    std::unordered_set<Address> addresses;

    for (Index node = 0; node < mesh.size(); ++node) {
        for (const auto& vertex : mesh.vertices(node)) {
            addresses.insert(vertex.address);
        }
    }

    return addresses.size();
}

std::size_t vertex_count(const Mesh& mesh, Index node) {
    std::size_t count = 0;

    for (const auto& vertex : mesh.vertices(node)) {
        static_cast<void>(vertex);
        ++count;
    }

    return count;
}

} // namespace

int main() {
    // ---------------------------------------------------------------------
    // Construct one database, one boundary, and one Voronoi mesh.
    // ---------------------------------------------------------------------

    const DatabaseParameters database_parameters{
        highvoronoi::DirectHash{64}
    };

    auto database = std::make_shared<Database>(
        64,
        database_parameters);

    Nodes nodes{Index{6}};
    nodes.set(Index{0}, point(0.0, 0.0));
    nodes.set(Index{1}, point(1.0, 0.0));
    nodes.set(Index{2}, point(2.0, 0.0));
    nodes.set(Index{3}, point(0.0, 1.0));
    nodes.set(Index{4}, point(1.0, 1.0));
    nodes.set(Index{5}, point(2.0, 1.0));

    auto boundary = Boundary::cuboid(
        point(2.0, 1.0),
        std::vector<Index>{});

    Mesh mesh{
        std::move(nodes),
        std::move(boundary),
        database
    };

    check(mesh.size() == Index{6},
          "the mesh initially contains six nodes");
    check(mesh.boundary().size() == 4,
          "the two-dimensional cuboid has four boundary planes");

    // ---------------------------------------------------------------------
    // Store ten initial vertices.
    // ---------------------------------------------------------------------

    const Sigma sigma_a{Index{0}, Index{1}, Index{2}};
    const Sigma sigma_b{Index{0}, Index{1}, Index{3}};
    const Sigma sigma_c{Index{1}, Index{2}, Index{8}};
    const Sigma sigma_d{Index{1}, Index{4}, Index{5}};
    const Sigma sigma_e{Index{0}, Index{3}, Index{4}};
    const Sigma sigma_f{Index{2}, Index{4}, Index{5}};
    const Sigma sigma_g{Index{0}, Index{2}, Index{3}};
    const Sigma sigma_h{Index{0}, Index{2}, Index{7}};
    const Sigma sigma_i{Index{2}, Index{3}, Index{5}};
    const Sigma sigma_j{Index{0}, Index{3}, Index{7}};

    const auto point_a = point(0.4, 0.1);
    const auto point_b = point(0.3, 0.3);
    const auto point_c = point(1.2, 0.3);
    const auto point_d = point(1.4, 0.6);
    const auto point_e = point(0.4, 0.7);
    const auto point_f = point(1.6, 0.7);
    const auto point_g = point(0.6, 0.4);
    const auto point_h = point(1.0, 0.5);
    const auto point_i = point(1.2, 0.6);
    const auto point_j = point(0.7, 0.6);

    const auto address_a = mesh.store_vertex(point_a, sigma_a);
    const auto address_b = mesh.store_vertex(point_b, sigma_b);
    const auto address_c = mesh.store_vertex(point_c, sigma_c);
    const auto address_d = mesh.store_vertex(point_d, sigma_d);
    const auto address_e = mesh.store_vertex(point_e, sigma_e);
    const auto address_f = mesh.store_vertex(point_f, sigma_f);
    const auto address_g = mesh.store_vertex(point_g, sigma_g);
    const auto address_h = mesh.store_vertex(point_h, sigma_h);
    const auto address_i = mesh.store_vertex(point_i, sigma_i);
    const auto address_j = mesh.store_vertex(point_j, sigma_j);

    check(address_a != 0 && address_b != 0 &&
              address_c != 0 && address_d != 0 &&
              address_e != 0 && address_f != 0 &&
              address_g != 0 && address_h != 0 &&
              address_i != 0 && address_j != 0,
          "all ten initial vertices were stored");

    check(mesh.contains_vertex(sigma_a) &&
              mesh.contains_vertex(sigma_b) &&
              mesh.contains_vertex(sigma_c) &&
              mesh.contains_vertex(sigma_d) &&
              mesh.contains_vertex(sigma_e) &&
              mesh.contains_vertex(sigma_f) &&
              mesh.contains_vertex(sigma_g) &&
              mesh.contains_vertex(sigma_h) &&
              mesh.contains_vertex(sigma_i) &&
              mesh.contains_vertex(sigma_j),
          "all ten initial signatures are present");

    check(active_vertex_count(mesh) == 10,
          "the mesh exposes ten distinct active vertices");

    // ---------------------------------------------------------------------
    // Delete public nodes 1 and 4.
    //
    // Every vertex containing one of these nodes must disappear. Internal
    // node indices and database signatures remain stable, while the four
    // surviving nodes receive the new public numbering 0, 1, 2, 3.
    // ---------------------------------------------------------------------

    const auto deleted_nodes = mesh.erase_nodes_if(
        [](Index node, const auto&) {
            return node == Index{1} || node == Index{4};
        });

    check(deleted_nodes == 2,
          "the node filter deleted exactly two nodes");
    check(mesh.size() == Index{4},
          "four public nodes remain after node deletion");

    check(!database_record_is_active(*database, address_a) &&
              !database_record_is_active(*database, address_b) &&
              !database_record_is_active(*database, address_c) &&
              !database_record_is_active(*database, address_d) &&
              !database_record_is_active(*database, address_e) &&
              !database_record_is_active(*database, address_f),
          "all vertices touching a deleted node were removed");

    check(database_record_is_active(*database, address_g) &&
              database_record_is_active(*database, address_h) &&
              database_record_is_active(*database, address_i) &&
              database_record_is_active(*database, address_j),
          "all vertices independent of the deleted nodes remain active");

    const Sigma public_g{Index{0}, Index{1}, Index{2}};
    const Sigma public_h{Index{0}, Index{1}, Index{5}};
    const Sigma public_i{Index{1}, Index{2}, Index{3}};
    const Sigma public_j{Index{0}, Index{2}, Index{5}};

    std::cout << "\nExpected public sigma transformations after deleting nodes 1 and 4:\n"
              << "  (0, 2, 3) -> (0, 1, 2)\n"
              << "  (0, 2, 7) -> (0, 1, 5)\n"
              << "  (2, 3, 5) -> (1, 2, 3)\n"
              << "  (0, 3, 7) -> (0, 2, 5)\n";

    check(mesh_contains_vertex(mesh, public_g, point_g),
          "sigma (0, 2, 3) became (0, 1, 2) and kept its point");
    check(mesh_contains_vertex(mesh, public_h, point_h),
          "sigma (0, 2, 7) became (0, 1, 5) and kept boundary plane 1");
    check(mesh_contains_vertex(mesh, public_i, point_i),
          "sigma (2, 3, 5) became (1, 2, 3) and kept its point");
    check(mesh_contains_vertex(mesh, public_j, point_j),
          "sigma (0, 3, 7) became (0, 2, 5) and kept boundary plane 1");

    check(active_vertex_count(mesh) == 4,
          "exactly four active vertices remain after node deletion");

    // ---------------------------------------------------------------------
    // Store four new vertices after the public numbering changed.
    //
    // Public indices 4, 5, 6, and 7 now denote the four boundary mirrors.
    // ---------------------------------------------------------------------

    const Sigma sigma_k{Index{0}, Index{1}, Index{4}};
    const Sigma sigma_l{Index{1}, Index{2}, Index{5}};
    const Sigma sigma_m{Index{2}, Index{3}, Index{6}};
    const Sigma sigma_n{Index{0}, Index{3}, Index{7}};

    const auto point_k = point(0.2, 0.2);
    const auto point_l = point(1.4, 0.2);
    const auto point_m = point(1.6, 0.8);
    const auto point_n = point(0.2, 0.8);

    const auto address_k = mesh.store_vertex(point_k, sigma_k);
    const auto address_l = mesh.store_vertex(point_l, sigma_l);
    const auto address_m = mesh.store_vertex(point_m, sigma_m);
    const auto address_n = mesh.store_vertex(point_n, sigma_n);

    check(address_k != 0 && address_l != 0 &&
              address_m != 0 && address_n != 0,
          "four new vertices were stored after node deletion");

    check(mesh_contains_vertex(mesh, sigma_k, point_k) &&
              mesh_contains_vertex(mesh, sigma_l, point_l) &&
              mesh_contains_vertex(mesh, sigma_m, point_m) &&
              mesh_contains_vertex(mesh, sigma_n, point_n),
          "all four new boundary vertices have the correct sigma and point");

    check(active_vertex_count(mesh) == 8,
          "the mesh now exposes eight distinct active vertices");

    // ---------------------------------------------------------------------
    // Delete three vertices through the vertex filter.
    //
    // The chosen x-coordinates remove old vertex I and new vertices L and M.
    // ---------------------------------------------------------------------

    const auto deleted_vertices = mesh.erase_vertices_if(
        [](const Sigma&, const Point& vertex) {
            return vertex[0] > 1.1;
        });

    check(deleted_vertices == 3,
          "the vertex filter deleted exactly three vertices");

    check(!mesh.contains_vertex(public_i) &&
              !mesh.contains_vertex(sigma_l) &&
              !mesh.contains_vertex(sigma_m),
          "the three selected signatures are no longer present");

    check(mesh_contains_vertex(mesh, public_g, point_g) &&
              mesh_contains_vertex(mesh, public_h, point_h) &&
              mesh_contains_vertex(mesh, public_j, point_j) &&
              mesh_contains_vertex(mesh, sigma_k, point_k) &&
              mesh_contains_vertex(mesh, sigma_n, point_n),
          "all surviving vertices retain both their sigma and point");

    check(active_vertex_count(mesh) == 5,
          "five active vertices remain after vertex filtering");

    // ---------------------------------------------------------------------
    // Store two more vertices after both node and vertex filtering.
    // ---------------------------------------------------------------------

    const Sigma sigma_o{Index{0}, Index{2}, Index{4}};
    const Sigma sigma_p{Index{1}, Index{3}, Index{7}};

    const auto point_o = point(0.5, 0.2);
    const auto point_p = point(0.8, 0.9);

    const auto address_o = mesh.store_vertex(point_o, sigma_o);
    const auto address_p = mesh.store_vertex(point_p, sigma_p);

    check(address_o != 0 && address_p != 0,
          "two final vertices were stored successfully");

    check(mesh_contains_vertex(mesh, sigma_o, point_o) &&
              mesh_contains_vertex(mesh, sigma_p, point_p),
          "both final vertices have the correct sigma and point");

    check(database_record_is_active(*database, address_o) &&
              database_record_is_active(*database, address_p),
          "both final database records are active");

    check(active_vertex_count(mesh) == 7,
          "the final mesh contains seven distinct active vertices");

    check(vertex_count(mesh, Index{0}) == 6,
          "public node 0 exposes all six associated vertices");
    check(vertex_count(mesh, Index{1}) == 4,
          "public node 1 exposes all four associated vertices");
    check(vertex_count(mesh, Index{2}) == 3,
          "public node 2 exposes all three associated vertices");
    check(vertex_count(mesh, Index{3}) == 2,
          "public node 3 exposes both associated vertices");

    std::cout << "\nFailed checks: " << failed_checks << '\n';
    return failed_checks == 0 ? 0 : 1;
}


