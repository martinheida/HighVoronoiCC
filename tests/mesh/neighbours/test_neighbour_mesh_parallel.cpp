
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {
using Scalar = double;
using Index = std::uint32_t;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;
using DB = highvoronoi::HVDataBase<highvoronoi::detail::ReadWriteLock, Params, 2>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, 2, DB>;
using Point = Mesh::NodePoint;

Point point(Scalar x, Scalar y) {
    Point p;
    p << x, y;
    return p;
}
}

int main() {
    constexpr Index Cells = 128;
    constexpr std::size_t Threads = 8;

    Mesh::InternalNodes nodes(Cells);
    for (Index i = 0; i < Cells; ++i) {
        nodes.set(i, point(static_cast<Scalar>(i), Scalar{0}));
    }

    auto db = std::make_shared<DB>(256, Params{highvoronoi::DirectHash{256}});
    Mesh mesh(std::move(nodes), db);

    std::vector<std::thread> workers;
    for (std::size_t t = 0; t < Threads; ++t) {
        workers.emplace_back([&, t] {
            std::vector<Index> neighbours;
            for (Index cell = static_cast<Index>(t);
                 cell < Cells;
                 cell = static_cast<Index>(cell + Threads)) {
                neighbours = {
                    static_cast<Index>((cell + 1) % Cells),
                    static_cast<Index>((cell + 1) % Cells),
                    static_cast<Index>((cell + 2) % Cells)};
                std::sort(neighbours.begin(), neighbours.end());
                mesh.store_internal_neighbours(cell, neighbours);
                mesh.set_dirty(cell, false);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    bool ok = true;
    std::vector<Index> buffer;
    for (Index cell = 0; cell < Cells; ++cell) {
        const Index a = static_cast<Index>((cell + 1) % Cells);
        const Index b = static_cast<Index>((cell + 2) % Cells);
        std::vector<Index> expected{a, a, b};
        std::sort(expected.begin(), expected.end());
        ok = ok && mesh.neighbours(cell, buffer);
        ok = ok && buffer == expected;
    }

    std::vector<std::thread> dirty_workers;
    for (std::size_t t = 0; t < Threads; ++t) {
        dirty_workers.emplace_back([&, t] {
            for (Index cell = static_cast<Index>(t);
                 cell < Cells;
                 cell = static_cast<Index>(cell + Threads)) {
                mesh.set_dirty(cell, true);
            }
        });
    }
    for (auto& worker : dirty_workers) {
        worker.join();
    }
    for (Index cell = 0; cell < Cells; ++cell) {
        ok = ok && mesh.dirty(cell);
    }

    std::cout << (ok ? "[OK]" : "[FAIL]")
              << " parallel VoronoiMesh neighbour addresses/database/dirty state\n";
    return ok ? 0 : 1;
}


