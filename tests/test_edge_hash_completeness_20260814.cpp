#include <highvoronoi/detail/edge_hash_table.hpp>
#include <highvoronoi/detail/hash_table_containers.hpp>
#include <highvoronoi/detail/hash_generators.hpp>
#include <highvoronoi/detail/locks.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using Table = highvoronoi::detail::EdgeHashTable<
        highvoronoi::detail::ReadWriteLock,
        highvoronoi::detail::FNV64_128HashGenerator>;

    const std::vector<std::uint32_t> a{1,2,3};
    const std::vector<std::uint32_t> b{4,5,6};

    Table exact(8);
    (void)exact.pushedge(a, 10, true);
    if (exact.all_edges_complete()) return 1;
    (void)exact.pushedge(a, 11, true);
    if (!exact.all_edges_complete()) return 2;
    if (!exact.pushedge(a, 12, true)) return 3;
    if (exact.all_edges_complete()) return 4;

    Table with_infinity(8);
    (void)with_infinity.pushedge(a, 10, true);
    (void)with_infinity.pushedge(a, Table::infinite_cell, true);
    if (!with_infinity.all_edges_complete()) return 5;

    (void)with_infinity.pushedge(b, 20, true);
    if (with_infinity.all_edges_complete()) return 6;
    (void)with_infinity.pushedge(b, 21, true);
    if (!with_infinity.all_edges_complete()) return 7;

    std::cout << "edge-hash exact completion: PASS\n";
    return 0;
}
