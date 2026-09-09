#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool condition, std::string_view text) {
    ++checks;
    std::cout << (condition ? "    [OK]   " : "    [FAIL] ") << text << '\n';
    if (!condition) {
        ++failures;
    }
}

template <int Dim>
using Database = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    Params,
    Dim>;

template <int Dim>
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dim, Database<Dim>>;

template <int Dim>
std::shared_ptr<Database<Dim>> make_database(std::size_t capacity = 1024) {
    return std::make_shared<Database<Dim>>(
        65536,
        Params{highvoronoi::DirectHash{capacity}});
}

void section(std::string_view name) {
    std::cout << "\n============================================================\n"
              << name << '\n'
              << "============================================================\n";
}

// -----------------------------------------------------------------------------
// General-position shortcut
// -----------------------------------------------------------------------------

void test_general_position() {
    section("1. General-position vertex: every co-generator is a neighbour");

    using M = Mesh<3>;
    using Point = M::NodePoint;

    const auto point = [](Scalar x, Scalar y, Scalar z) {
        Point p;
        p << x, y, z;
        return p;
    };

    M::InternalNodes nodes(Index{4});
    nodes.set(Index{0}, point(1, 1, 1));
    nodes.set(Index{1}, point(-1, -1, 1));
    nodes.set(Index{2}, point(-1, 1, -1));
    nodes.set(Index{3}, point(1, -1, -1));

    M mesh(std::move(nodes), make_database<3>());
    const std::vector<Index> sigma{0, 1, 2, 3};
    check(mesh.store_vertex(point(0, 0, 0), sigma) != 0,
          "store one 3D general-position vertex");

    auto workspace = mesh.make_neighbour_workspace();
    std::vector<Index> neighbours;

    for (Index cell = 0; cell < Index{4}; ++cell) {
        mesh.compute_neighbors(cell, workspace);
        const bool current = mesh.neighbours(cell, neighbours);

        std::vector<Index> expected;
        for (Index other = 0; other < Index{4}; ++other) {
            if (other != cell) {
                expected.push_back(other);
            }
        }

        check(current && neighbours == expected,
              "general-position cell exposes its three co-generators");
    }
}

// -----------------------------------------------------------------------------
// Repeated degenerate Cartesian vertices: this is the important scaling case.
// -----------------------------------------------------------------------------

void test_cartesian_3x3x3_cell() {
    section("2. 3x3x3 Cartesian cell: 26 adjacents, 6 facet neighbours");

    using M = Mesh<3>;
    using Point = M::NodePoint;

    const auto point = [](Scalar x, Scalar y, Scalar z) {
        Point p;
        p << x, y, z;
        return p;
    };
    const auto index = [](Index x, Index y, Index z) {
        return static_cast<Index>(x + Index{3} * y + Index{9} * z);
    };

    M::InternalNodes nodes(Index{27});
    for (Index z = 0; z < 3; ++z) {
        for (Index y = 0; y < 3; ++y) {
            for (Index x = 0; x < 3; ++x) {
                nodes.set(index(x, y, z), point(x, y, z));
            }
        }
    }

    M mesh(std::move(nodes), make_database<3>(2048));

    // Eight highly degenerate Voronoi vertices around the center node.
    for (Index vz = 0; vz < 2; ++vz) {
        for (Index vy = 0; vy < 2; ++vy) {
            for (Index vx = 0; vx < 2; ++vx) {
                std::vector<Index> sigma;
                sigma.reserve(8);
                for (Index dz = 0; dz < 2; ++dz) {
                    for (Index dy = 0; dy < 2; ++dy) {
                        for (Index dx = 0; dx < 2; ++dx) {
                            sigma.push_back(index(vx + dx, vy + dy, vz + dz));
                        }
                    }
                }
                std::sort(sigma.begin(), sigma.end());
                check(mesh.store_vertex(
                          point(vx + Scalar{0.5},
                                vy + Scalar{0.5},
                                vz + Scalar{0.5}),
                          sigma) != 0,
                      "store Cartesian degenerate vertex");
            }
        }
    }

    const Index cell = index(1, 1, 1);

    typename M::AdjacencyWorkspace adjacency_workspace;
    std::vector<Index> adjacents;
    adjacents.reserve(32);
    mesh.compute_adjecents(cell, adjacents, adjacency_workspace);

    check(adjacents.size() == 26,
          "center cell sees all 26 surrounding nodes as vertex-adjacents");

    auto neighbour_workspace = mesh.make_neighbour_workspace();
    mesh.compute_neighbors(cell, neighbour_workspace);

    std::vector<Index> neighbours;
    check(mesh.neighbours(cell, neighbours),
          "computed neighbour record is current");

    std::vector<Index> expected{
        index(0, 1, 1), index(2, 1, 1),
        index(1, 0, 1), index(1, 2, 1),
        index(1, 1, 0), index(1, 1, 2)};
    std::sort(expected.begin(), expected.end());

    check(neighbours == expected,
          "only the six axis cells share positive 2D facets");

    // Reuse both workspaces. This also catches uncleared membership bits.
    std::vector<Index> second_adjacents;
    second_adjacents.reserve(32);
    mesh.compute_adjecents(cell, second_adjacents, adjacency_workspace);
    check(second_adjacents == adjacents,
          "AdjacencyWorkspace is clean and reusable");

    mesh.set_dirty(cell, true);
    mesh.compute_neighbors(cell, neighbour_workspace);
    std::vector<Index> second_neighbours;
    check(mesh.neighbours(cell, second_neighbours) &&
              second_neighbours == expected,
          "NeighbourWorkspace is clean and reusable across cell recomputation");
}

// -----------------------------------------------------------------------------
// Degenerate boundary mirror: diagonal adjacent must not become a facet.
// -----------------------------------------------------------------------------

void test_degenerate_boundary_vertex() {
    section("3. Degenerate boundary vertex: ordinary diagonal is rejected");

    using M = Mesh<3>;
    using Point = M::NodePoint;
    using Boundary = M::BoundaryType;

    const auto point = [](Scalar x, Scalar y, Scalar z) {
        Point p;
        p << x, y, z;
        return p;
    };

    M::InternalNodes nodes(Index{4});
    nodes.set(0, point(0, 0, 0));
    nodes.set(1, point(2, 0, 0));
    nodes.set(2, point(0, 2, 0));
    nodes.set(3, point(2, 2, 0));

    Boundary boundary;
    boundary.add(highvoronoi::Plane<3, Scalar, Index>(
        point(0, 0, -1),
        point(0, 0, -1),
        highvoronoi::BoundaryCondition::Dirichlet));

    M mesh(std::move(nodes), boundary, make_database<3>());
    const Index boundary_public = mesh.size();

    // All four ordinary generators and the active-cell mirror meet at one
    // vertex. For each ordinary cell the opposite diagonal is only adjacent.
    const std::vector<Index> sigma{0, 1, 2, 3, boundary_public};
    check(mesh.store_vertex(point(1, 1, -1), sigma) != 0,
          "store degenerate ordinary+boundary vertex");

    const std::array<std::array<Index, 2>, 4> axis_neighbours{{
        {{1, 2}}, {{0, 3}}, {{0, 3}}, {{1, 2}}
    }};

    auto workspace = mesh.make_neighbour_workspace();
    std::vector<Index> neighbours;
    for (Index cell = 0; cell < 4; ++cell) {
        mesh.compute_neighbors(cell, workspace);
        const bool current = mesh.neighbours(cell, neighbours);
        std::vector<Index> expected{
            axis_neighbours[cell][0],
            axis_neighbours[cell][1],
            boundary_public};
        std::sort(expected.begin(), expected.end());
        check(current && neighbours == expected,
              "cell keeps two axis facets plus the boundary facet");
    }
}


// -----------------------------------------------------------------------------
// Persistent workspace across structural node growth.
// -----------------------------------------------------------------------------

void test_workspace_survives_node_growth() {
    section("4. Persistent NeighbourWorkspace survives stable node growth");

    using Nodes = highvoronoi::VoronoiNodes<Scalar, 2, Index>;
    using Boundary = highvoronoi::Boundary<2, Scalar, Index>;
    using Point = Nodes::Point;
    using Finder = highvoronoi::detail::NeighbourFinder<
        Nodes,
        Boundary,
        Scalar>;

    const auto point = [](Scalar x, Scalar y) {
        Point p;
        p << x, y;
        return p;
    };

    Nodes nodes(Index{3});
    nodes.set(Index{0}, point(0, 0));
    nodes.set(Index{1}, point(2, 0));
    nodes.set(Index{2}, point(0, 2));

    Boundary boundary;
    boundary.add(highvoronoi::Plane<2, Scalar, Index>(
        point(0, -1),
        point(0, -1),
        highvoronoi::BoundaryCondition::Dirichlet));

    Finder finder(nodes, boundary);
    const Index boundary_internal =
        static_cast<Index>((std::numeric_limits<Index>::max)() - Index{1});

    // First populate a boundary slot. With three ordinary nodes, plane 0 uses
    // compact scratch slot 3.
    finder.reset(Index{0});
    std::vector<Index> first_sigma{Index{0}, Index{1}, boundary_internal};
    finder.process_vertex(first_sigma, point(1, 0));
    std::vector<Index> neighbours;
    finder.finish(neighbours);
    check(neighbours == std::vector<Index>({Index{1}, boundary_internal}),
          "workspace first records an ordinary and a boundary neighbour");

    // Append one stable ordinary node. Its compact slot is now exactly the old
    // boundary slot. A persistent workspace must not inherit the old boundary
    // seen/invalid bit for this new node.
    nodes.resize(Index{4});
    nodes.set(Index{3}, point(2, 2));

    finder.reset(Index{0});
    std::vector<Index> second_sigma{Index{0}, Index{2}, Index{3}};
    finder.process_vertex(second_sigma, point(0.5, 1.0));
    finder.finish(neighbours);

    check(neighbours == std::vector<Index>({Index{2}, Index{3}}),
          "node growth cannot alias an old boundary scratch slot");
}

} // namespace

int main() {
    test_general_position();
    test_cartesian_3x3x3_cell();
    test_degenerate_boundary_vertex();
    test_workspace_survives_node_growth();

    std::cout << "\n============================================================\n"
              << "checks: " << checks << " | failed: " << failures << '\n'
              << "============================================================\n";
    return failures == 0 ? 0 : 1;
}
