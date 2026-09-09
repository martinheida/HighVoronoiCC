#include <highvoronoi/storage/neighbour/hybrid_neighbour_database.hpp>
#include <highvoronoi/storage/neighbour/neighbour_storage.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {
using Index = std::uint32_t;
using Sigma = std::vector<Index>;
using Lock = highvoronoi::detail::EmptyLock;
using HybridDB = highvoronoi::detail::HybridNeighbourDatabase<Lock, Index>;
using Storage = highvoronoi::detail::NeighbourStorage<Lock, Index, HybridDB>;

std::size_t checks=0, failures=0;
void check(bool x, std::string_view s) {
    ++checks;
    if (x) std::cout << "    [OK]   " << s << '\n';
    else { ++failures; std::cerr << "    [FAIL] " << s << '\n'; }
}

class Engine {
public:
    using Index = ::Index;
    using Sigma = ::Sigma;
    bool provides_neighbours() const noexcept { return true; }
    Index node_count() const noexcept { return 2; }
    bool read_neighbours(Index c, Sigma& out) const {
        out = c == 0 ? Sigma{1} : Sigma{0};
        return c < 2;
    }
};
}

int main() {
    Storage storage(4, 64);
    check(storage.address(0) == 0, "new cell has no neighbour address");
    check(storage.dirty(0), "new cell starts dirty");

    const auto first = storage.store(0, Sigma{1,2,2});
    check(first == 1 && storage.address(0) == 1, "stored record publishes one current address");
    check(storage.dirty(0), "publication does not implicitly clear dirty");

    storage.set_dirty(0, false);
    Sigma old;
    check(storage.read_address(first, old) && old == Sigma({1,2,2}),
          "explicit historical address remains readable");

    const auto second = storage.store(0, Sigma{2,3});
    check(second != first && storage.address(0) == second,
          "new publication replaces only the current address");
    check(storage.read_address(first, old) && old == Sigma({1,2,2}),
          "old append-only database record survives address replacement");

    auto engine = std::make_shared<Engine>();
    const auto reg = storage.database().register_engine(engine, Index{10});
    storage.publish_address(2, reg.first_address + 0);
    storage.publish_address(3, reg.first_address + 1);

    Sigma current;
    check(storage.load(2, current) && current == Sigma({11}),
          "cell may publish a virtual engine neighbour address");
    check(storage.load(3, current) && current == Sigma({10}),
          "second virtual engine address routes through same database");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
