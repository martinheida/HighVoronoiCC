
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/mesh_view.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {
using Scalar = double;
using Index = std::uint32_t;
constexpr int Dimension = 2;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::ReadWriteLock, Params, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Point = Mesh::VertexPoint;
using Sigma = Mesh::Sigma;
using View = highvoronoi::ReorderedMeshView<Mesh, highvoronoi::SwitchView<Index>>;

Point point(Scalar x, Scalar y) { Point p; p << x,y; return p; }
}

int main() {
    Params params{highvoronoi::DirectHash{128}};
    auto db = std::make_shared<Database>(256, params);
    Mesh::InternalNodes nodes(Index{4});
    nodes.set(Index{0}, point(0,0));
    nodes.set(Index{1}, point(1,0));
    nodes.set(Index{2}, point(0,1));
    nodes.set(Index{3}, point(1,1));
    Mesh mesh(std::move(nodes), db);

    const Sigma full_edge{Index{0}, Index{2}};
    const Point origin = point(0.25, 0.5);
    const Point direction = point(-1.0, 0.0);

    std::atomic<int> stored{0};
    std::vector<std::thread> threads;
    for (int t=0;t<8;++t) {
        threads.emplace_back([&] {
            Sigma scratch;
            if (mesh.store_infinite_edge(full_edge, origin, direction, scratch) != 0) {
                stored.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t: threads) t.join();

    if (stored.load() != 1) {
        std::cerr << "expected exactly one persisted infinite edge, got " << stored.load() << '\n';
        return 1;
    }

    std::size_t count = 0;
    for (const auto& edge : mesh.infinite_edges()) {
        ++count;
        if (edge.sigma != full_edge || edge.origin != origin || edge.direction != direction) {
            std::cerr << "underlying mesh returned wrong infinite edge\n";
            return 2;
        }
    }
    if (count != 1) return 3;

    View view(mesh, highvoronoi::SwitchView<Index>(Index{1}, Index{3}));
    const Sigma view_sigma{Index{1}, Index{3}}; // maps to wrapped {0,2}
    Sigma scratch;
    if (view.store_infinite_edge(view_sigma, origin, direction, scratch) != 0) {
        std::cerr << "view failed to deduplicate against wrapped storage\n";
        return 4;
    }

    count = 0;
    for (const auto& edge : view.infinite_edges()) {
        ++count;
        if (edge.sigma != view_sigma) {
            std::cerr << "view returned wrong public sigma\n";
            return 5;
        }
    }
    if (count != 1) return 6;

    std::cout << "infinite-edge mesh storage: PASS\n";
    return 0;
}


