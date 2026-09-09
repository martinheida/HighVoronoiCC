#include <highvoronoi/algorithm/edge_iterator.hpp>
#include <highvoronoi/core/voronoi_nodes.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 4;
using Nodes = highvoronoi::VoronoiNodes<Scalar, Dimension, Index>;
using Point = Nodes::Point;
using Iterator = highvoronoi::EdgeIterator<Nodes>;

std::vector<Index> expected_neighbours(Index cell) {
    std::vector<Index> result;
    for (int bit = 0; bit < Dimension; ++bit) {
        result.push_back(static_cast<Index>(cell ^ (Index{1} << bit)));
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace

int main() {
    Nodes nodes(Index{16});
    for (Index index = 0; index < Index{16}; ++index) {
        Point point;
        for (int axis = 0; axis < Dimension; ++axis) {
            point[axis] =
                (index & (Index{1} << axis)) != 0 ? Scalar{1} : Scalar{-1};
        }
        nodes.set(index, point);
    }

    std::vector<Index> sigma(16);
    std::iota(sigma.begin(), sigma.end(), Index{0});
    Point vertex = Point::Zero();

    Iterator iterator(nodes);
    std::vector<Index> valid;

    for (Index cell = 0; cell < Index{16}; ++cell) {
        iterator.reset(sigma, vertex, cell, typename Iterator::OnCellEdges{});
        iterator.valid_generators(valid);
        std::sort(valid.begin(), valid.end());

        if (valid != expected_neighbours(cell)) {
            std::cerr << "invalid FEI neighbour set for cell " << cell << '\n';
            return 1;
        }

        std::size_t edge_count = 0;
        while (const auto edge = iterator.next()) {
            if (std::find(
                    edge->full_indices().begin(),
                    edge->full_indices().end(),
                    cell) == edge->full_indices().end()) {
                std::cerr << "cell-local edge does not contain active cell\n";
                return 2;
            }
            ++edge_count;
        }
        if (edge_count != Dimension) {
            std::cerr << "cell " << cell << " returned " << edge_count
                      << " edges instead of " << Dimension << '\n';
            return 3;
        }
    }

    if (!iterator.storage_cache().empty()) {
        std::cerr << "OnCellEdges modified shared FEIStorageCache\n";
        return 4;
    }

    std::cout << "[OK] OnCellEdges: 16 cells x 4 neighbours/edges, FEI cache untouched\n";
    return 0;
}
