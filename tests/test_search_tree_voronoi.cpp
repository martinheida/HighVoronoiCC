#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/boundary.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint16_t;
constexpr int Dim = 2;

using Parameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Lock = highvoronoi::SingleThread::RWLock;
using Database = highvoronoi::HVDataBase<Lock, Parameters>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dim, Database>;
using Point = highvoronoi::StaticPoint<Scalar, Dim>;
using Boundary = highvoronoi::Boundary<Dim, Scalar, Index>;

Point make_point(Scalar x, Scalar y) {
    Point result;
    result << x, y;
    return result;
}

template<class PointLike>
void print_point(std::string_view name, const PointLike& point) {
    std::cout << "    " << name << " = ("
              << point[0] << ", " << point[1] << ")\n";
}

void print_mesh_nodes(const Mesh& mesh) {
    std::cout << "\nPUBLIC NODES IN VORONOI MESH\n";
    std::cout << "----------------------------\n";

    for (Index index = 0; index < mesh.size(); ++index) {
        const auto node = mesh.nodes().node(index);
        std::cout << "    mesh node " << index << " = ("
                  << node[0] << ", " << node[1] << ")\n";
    }
}

void print_active_boundary_nodes(const Mesh& mesh) {
    const auto& extended = mesh.concrete_extended_nodes();

    std::cout << "\nACTIVE BOUNDARY NODES\n";
    std::cout << "---------------------\n";

    if (extended.active_planes().empty()) {
        std::cout << "    none\n";
        return;
    }

    for (const Index plane : extended.active_planes()) {
        const Index mirror = extended.mirror_index(plane);
        const auto node = extended.node(mirror);

        std::cout << "    plane " << plane
                  << " -> extended node " << mirror
                  << " = (" << node[0] << ", " << node[1] << ")\n";
    }
}

Boundary make_boundary() {
    // Square [-1,1] x [-1,1], explicitly non-periodic.
    // Plane order created by cuboid():
    //   0: x =  1
    //   1: x = -1
    //   2: y =  1
    //   3: y = -1
    const Point dimensions = make_point(2.0, 2.0);
    const Point offset = make_point(-1.0, -1.0);

    return Boundary::cuboid(
        dimensions,
        offset,
        std::vector<Index>{});
}

std::unique_ptr<Mesh> make_mesh() {
    Mesh::InternalNodes nodes(Index{3});

    nodes.set(Index{0}, make_point( 0.8, 0.0));
    nodes.set(Index{1}, make_point(-0.4, 0.0));
    nodes.set(Index{2}, make_point( 0.0, 0.7));

    auto database = std::make_shared<Database>(
        std::size_t{64},
        Parameters{Parameters::ContainerMode{}});

    return std::make_unique<Mesh>(
        std::move(nodes),
        make_boundary(),
        std::move(database));
}

template<class SearchData>
void print_tree_point(const SearchData& data) {
    std::cout << "    tree buffer point = ("
              << data.point[0] << ", " << data.point[1] << ")\n";
}

template<class SearchData>
void print_search_results(
    std::string_view name,
    const SearchData& data) {
    std::cout << "    " << name << ":\n";

    if (data.list.empty()) {
        std::cout << "        no result\n";
        return;
    }

    for (std::size_t position = 0;
         position < data.list.size();
         ++position) {
        std::cout << "        index = " << data.list.index(position)
                  << ", distance = " << data.list.distance(position)
                  << '\n';
    }
}

template<class Tree>
void run_search_example(
    std::string_view tree_name,
    Tree& tree,
    Mesh& mesh) {
    std::cout << "\n\n============================================================\n";
    std::cout << tree_name << '\n';
    std::cout << "============================================================\n";

    auto data = tree.make_backend_data();
    data.reserve(8);

    const auto skip_nothing = [](Index) {
        return false;
    };

    // ---------------------------------------------------------------------
    // 1. Show conversion from a real mesh point to the tree-owned buffer.
    // ---------------------------------------------------------------------
    std::cout << "\nCONVERT MESH POINT TO TREE BUFFER\n";
    std::cout << "---------------------------------\n";

    const auto mesh_point = mesh.nodes().node(Index{0});
    print_point("mesh node 0", mesh_point);

    tree.write_point(data, mesh_point);
    print_tree_point(data);

    // The query lies outside the square, close to the reflection of node 0
    // at the right boundary x = 1.
    const Point query = make_point(1.15, 0.0);

    std::cout << "\nWRITE QUERY INTO THE SAME TREE BUFFER\n";
    std::cout << "-------------------------------------\n";
    print_point("query in mesh format", query);

    tree.write_point(data, query);
    print_tree_point(data);

    // ---------------------------------------------------------------------
    // 2. Search without active boundary nodes.
    // ---------------------------------------------------------------------
    std::cout << "\nSEARCH WITHOUT ACTIVE BOUNDARY\n";
    std::cout << "------------------------------\n";

    auto& extended = mesh.concrete_extended_nodes();
    const std::vector<Index> no_boundary_neighbours;
    extended.activate_cell(Index{0}, no_boundary_neighbours);

    print_active_boundary_nodes(mesh);

    const auto nearest_without_boundary = tree.nn(data, skip_nothing);
    print_search_results("nn", data);

    tree.inrange(data, Scalar{0.4}, skip_nothing);
    print_search_results("inrange(radius = 0.4)", data);

    // ---------------------------------------------------------------------
    // 3. Activate the reflection of cell/node 0 at x = 1.
    // ---------------------------------------------------------------------
    std::cout << "\nACTIVATE RIGHT BOUNDARY FOR CELL 0\n";
    std::cout << "----------------------------------\n";

    const Index right_plane = Index{0};
    const Index right_mirror = extended.mirror_index(right_plane);
    const std::vector<Index> boundary_neighbours{right_mirror};

    extended.activate_cell(Index{0}, boundary_neighbours);
    print_active_boundary_nodes(mesh);

    // ---------------------------------------------------------------------
    // 4. Repeat exactly the same searches with the same recycled data object.
    // ---------------------------------------------------------------------
    std::cout << "\nSEARCH WITH ACTIVE BOUNDARY\n";
    std::cout << "---------------------------\n";

    tree.write_point(data, query);
    print_tree_point(data);

    const auto nearest_with_boundary = tree.nn(data, skip_nothing);
    print_search_results("nn", data);

    tree.inrange(data, Scalar{0.4}, skip_nothing);
    print_search_results("inrange(radius = 0.4)", data);

    // ---------------------------------------------------------------------
    // 5. Human-readable expectation summary.
    // ---------------------------------------------------------------------
    std::cout << "\nEXPECTED BEHAVIOUR\n";
    std::cout << "------------------\n";
    std::cout << "    without boundary: nearest index 0 at distance 0.35\n";
    std::cout << "    with boundary:    nearest index " << right_mirror
              << " at distance 0.05\n";
    std::cout << "    with boundary:    inrange contains mirror "
              << right_mirror << " and public node 0\n";

    if (nearest_without_boundary) {
        std::cout << "    measured without boundary: index "
                  << nearest_without_boundary->index
                  << ", distance "
                  << nearest_without_boundary->distance << '\n';
    }

    if (nearest_with_boundary) {
        std::cout << "    measured with boundary:    index "
                  << nearest_with_boundary->index
                  << ", distance "
                  << nearest_with_boundary->distance << '\n';
    }
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    auto mesh_owner = make_mesh();
    Mesh& mesh = *mesh_owner;

    std::cout << "HIGHVORONOI SEARCH TREE INTEGRATION TEST\n";
    std::cout << "========================================\n";
    std::cout << mesh.boundary().to_string();
    print_mesh_nodes(mesh);

    {
        auto tree = highvoronoi::geometry::make_search_tree(
            mesh,
            highvoronoi::geometry::BruteForceSearch{});

        run_search_example(
            "BRUTE FORCE WITH REAL VORONOI MESH",
            tree,
            mesh);
    }

    // Reset active mirrors before constructing/running the second backend.
    mesh.concrete_extended_nodes().activate_cell(
        Index{0},
        std::vector<Index>{});

    {
        auto tree = highvoronoi::geometry::make_search_tree(
            mesh,
            highvoronoi::geometry::KDSearch{4, 1});

        run_search_example(
            "NANOFLANN WITH REAL VORONOI MESH",
            tree,
            mesh);
    }

    return 0;
}
