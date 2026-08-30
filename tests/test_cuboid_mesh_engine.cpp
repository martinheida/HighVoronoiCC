#include <highvoronoi/geometry/cuboid_mesh_engine.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
using Engine = highvoronoi::CuboidMeshEngine<Scalar, Scalar, Index, 3>;
using Point = Engine::NodePoint;
using VertexPoint = Engine::VertexPoint;
using CountPoint = Engine::CountPoint;
using Sigma = Engine::Sigma;
using Address = Engine::Address;

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

bool close(Scalar left, Scalar right) {
    return std::abs(left - right) <= 1e-12;
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point p;
    p << x, y, z;
    return p;
}

CountPoint counts(Index x, Index y, Index z) {
    CountPoint p;
    p << x, y, z;
    return p;
}

void print_point(const VertexPoint& point) {
    std::cout << '(' << point[0] << ", " << point[1] << ", " << point[2] << ')';
}

void print_node_point(const Point& point) {
    std::cout << '(' << point[0] << ", " << point[1] << ", " << point[2] << ')';
}

void print_sigma(const Sigma& sigma) {
    std::cout << '{';
    for (std::size_t i = 0; i < sigma.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << sigma[i];
    }
    std::cout << '}';
}

void print_vertex(const Engine& engine, Address address) {
    VertexPoint vertex;
    Sigma sigma;
    engine.read_vertex(address, vertex, sigma);

    std::cout << "vertex " << address << ": ";
    if (sigma.empty()) {
        std::cout << "<deleted / inactive>\n";
        return;
    }

    print_point(vertex);
    std::cout << "  sigma=";
    print_sigma(sigma);
    std::cout << '\n';
}

void print_all_nodes(const Engine& engine) {
    std::cout << "\n[NODES] all nodes of the cuboid grid\n";

    Point node;
    for (Index node_index = 0; node_index < engine.node_count(); ++node_index) {
        engine.copy_node(node_index, node.data());
        std::cout << "    node " << node_index << ": ";
        print_node_point(node);
        std::cout << '\n';
    }
}

void print_vertices_of_node(const Engine& engine, Index node_index) {
    Point node;
    engine.copy_node(node_index, node.data());

    std::cout << "\n[VERTICES OF NODE " << node_index << "] ";
    print_node_point(node);
    std::cout << '\n';

    const std::size_t primary_count = engine.primary_vertex_count(node_index);
    std::cout << "    PRIMARY (" << primary_count << ")\n";
    if (primary_count == 0) {
        std::cout << "        <none>\n";
    } else {
        for (std::size_t j = 0; j < primary_count; ++j) {
            const Address address = engine.primary_vertex_address(node_index, j);
            std::cout << "        ";
            print_vertex(engine, address);
        }
    }

    const std::size_t secondary_count = engine.secondary_vertex_count(node_index);
    std::cout << "    SECONDARY (" << secondary_count << ")\n";
    if (secondary_count == 0) {
        std::cout << "        <none>\n";
    } else {
        for (std::size_t j = 0; j < secondary_count; ++j) {
            const Address address = engine.secondary_vertex_address(node_index, j);
            std::cout << "        ";
            print_vertex(engine, address);
        }
    }
}

void test_geometry_and_incidence() {
    std::cout << "\n[TEST] CuboidMeshEngine 3D geometry and incidence\n";

    Engine engine(
        point(0.0, 0.0, 0.0),
        point(2.0, 4.0, 6.0),
        counts(3, 2, 3));

    check(engine.node_count() == Index{18},
          "3*2*3 ordinary nodes are generated");
    check(engine.vertex_address_capacity() == Address{4},
          "(3-1)*(2-1)*(3-1) finite vertices are generated");
    check(!engine.provides_complete_infinite_edges(),
          "unbounded CuboidMeshEngine currently reports no complete infinite-edge set");

    // ------------------------------------------------------------------
    // Human-readable inspection output: all nodes.
    // ------------------------------------------------------------------
    print_all_nodes(engine);

    Point node;
    engine.copy_node(Index{10}, node.data());
    check(close(node[0], 2.0) && close(node[1], 4.0) && close(node[2], 6.0),
          "flattened node 10 maps to grid coordinate (1,1,1)");

    VertexPoint vertex;
    Sigma sigma;
    engine.read_vertex(Address{0}, vertex, sigma);
    check(close(vertex[0], 1.0) && close(vertex[1], 2.0) && close(vertex[2], 3.0),
          "first finite vertex is the half-spacing point of the first box");
    check(sigma == Sigma({Index{0}, Index{1}, Index{3}, Index{4},
                          Index{6}, Index{7}, Index{9}, Index{10}}),
          "first vertex sigma contains exactly the eight box-corner nodes");

    engine.read_vertex(Address{3}, vertex, sigma);
    check(close(vertex[0], 3.0) && close(vertex[1], 2.0) && close(vertex[2], 9.0),
          "last finite vertex has analytically expected coordinates");
    check(engine.primary_vertex_count(Index{7}) == 1,
          "lower-corner node 7 owns one primary vertex");
    check(engine.primary_vertex_address(Index{7}, 0) == Address{3},
          "node 7 owns the last local vertex address");
    check(engine.secondary_vertex_count(Index{7}) == 3,
          "node 7 sees the other incident box vertices as secondary");

    // Node 7 has both a primary and secondary vertices; print all of them.
    print_vertices_of_node(engine, Index{7});

    std::size_t primary_occurrences = 0;
    std::size_t secondary_occurrences = 0;
    std::vector<std::size_t> occurrences(engine.vertex_address_capacity(), 0);

    for (Index node_index = 0; node_index < engine.node_count(); ++node_index) {
        for (std::size_t j = 0; j < engine.primary_vertex_count(node_index); ++j) {
            const Address address = engine.primary_vertex_address(node_index, j);
            ++primary_occurrences;
            ++occurrences[address];
        }
        for (std::size_t j = 0; j < engine.secondary_vertex_count(node_index); ++j) {
            const Address address = engine.secondary_vertex_address(node_index, j);
            ++secondary_occurrences;
            ++occurrences[address];
        }
    }

    check(primary_occurrences == 4,
          "every finite vertex has exactly one primary occurrence");
    check(secondary_occurrences == 4 * 7,
          "every 3D finite vertex has exactly seven secondary occurrences");
    check(std::all_of(occurrences.begin(), occurrences.end(),
                      [](std::size_t n) { return n == 8; }),
          "every finite vertex occurs at all eight generator nodes");

    // ------------------------------------------------------------------
    // Human-readable deletion output.
    // Vertex 0 is primary for node 0 and secondary for node 1.
    // ------------------------------------------------------------------
    std::cout << "\n[DELETE] vertex 0 before deletion\n    ";
    print_vertex(engine, Address{0});

    std::cout << "\n[BEFORE DELETE] primary owner and one secondary owner\n";
    print_vertices_of_node(engine, Index{0});
    print_vertices_of_node(engine, Index{1});

    check(engine.erase_vertex(Address{0}),
          "computed finite vertex can be deleted");

    std::cout << "\n[DELETE] vertex 0 after deletion\n    ";
    print_vertex(engine, Address{0});

    std::cout << "\n[AFTER DELETE] primary owner node 0\n";
    print_vertices_of_node(engine, Index{0});

    std::cout << "\n[AFTER DELETE] secondary owner node 1\n";
    print_vertices_of_node(engine, Index{1});

    sigma = Sigma{Index{999}};
    engine.read_vertex(Address{0}, vertex, sigma);
    check(sigma.empty(),
          "deleted computed vertex follows the empty-sigma tombstone convention");
    check(engine.primary_vertex_count(Index{0}) == 0,
          "deleted vertex disappears from primary incidence");
    check(engine.secondary_vertex_count(Index{1}) == 0,
          "deleted vertex disappears from secondary incidence as well");
}

} // namespace

int main() {
    test_geometry_and_incidence();
    std::cout << "\nperformed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    return failed_checks == 0 ? 0 : 1;
}
