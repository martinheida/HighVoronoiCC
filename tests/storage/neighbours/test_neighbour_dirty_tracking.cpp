#include <highvoronoi/storage/neighbour/neighbour_storage.hpp>
#include <highvoronoi/storage/neighbour/neighbour_dirty_tracker.hpp>
#include <highvoronoi/core/detail/locks.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {
using Index = std::uint32_t;
using Lock = highvoronoi::detail::EmptyLock;
using Storage = highvoronoi::detail::NeighbourStorage<Lock, Index>;
using Tracking = highvoronoi::detail::NeighbourTrackingState<Lock>;

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool ok, const char* text) {
    ++checks;
    if (ok) {
        std::cout << "[OK]   " << text << '\n';
    } else {
        ++failures;
        std::cerr << "[FAIL] " << text << '\n';
    }
}
}

int main() {
    Storage storage(130, 128, false);
    Tracking tracking;

    auto first = tracking.request_tracker(storage.dirty_vector());
    auto second = tracking.request_tracker(storage.dirty_vector());
    check(tracking.registered_tracker_count() == 2,
          "two live consumer dirty trackers are registered");

    storage.set_dirty(Index{1});
    storage.set_dirty(Index{63});
    storage.set_dirty(Index{64});
    storage.set_dirty(Index{129});
    tracking.propagate(storage.dirty_vector());

    check(first->dirty(1) && first->dirty(63) && first->dirty(64) && first->dirty(129),
          "word-wise propagation crosses several packed 64-bit words");
    check(second->dirty(64),
          "all registered consumers receive the same mesh dirty state");
    check(storage.dirty(Index{64}),
          "propagation does not clear the mesh-owned dirty bit");

    first->set_dirty(64, false);
    check(!first->dirty(64) && second->dirty(64),
          "consumer trackers can be cleared independently");

    storage.set_dirty(Index{1}, false);
    storage.set_dirty(Index{63}, false);
    storage.set_dirty(Index{64}, false);
    storage.set_dirty(Index{129}, false);
    second->clear();

    storage.resize(133);
    tracking.propagate(storage.dirty_vector());
    check(first->size() == 133 && second->size() == 133,
          "registered trackers grow at the next propagation boundary");
    check(first->dirty(130) && first->dirty(131) && first->dirty(132),
          "newly appended cells propagate as dirty");

    check(tracking.version() == 0,
          "global neighbour version starts at zero");
    tracking.propagate(storage.dirty_vector());
    check(tracking.version() == 0,
          "dirty propagation does not implicitly advance the global version");
    check(tracking.advance_version() == 1 && tracking.version() == 1,
          "global neighbour version advances only by explicit commit");

    std::vector<Index> old_neighbours{Index{2}, Index{3}};
    std::vector<Index> new_neighbours{Index{4}, Index{5}, Index{6}};
    const auto old_address = storage.store(Index{0}, old_neighbours);
    const auto new_address = storage.store(Index{0}, new_neighbours);
    check(storage.address(Index{0}) == new_address && old_address != new_address,
          "one current address is replaced without an address history per cell");

    std::vector<Index> read_back;
    check(storage.read_address(old_address, read_back) && read_back == old_neighbours,
          "old append-only database record remains readable by retained address");
    check(storage.load(Index{0}, read_back) && read_back == new_neighbours,
          "cell load follows only the one current published address");

    first.reset();
    tracking.propagate(storage.dirty_vector());
    check(tracking.registered_tracker_count() == 1,
          "expired tracker registrations are pruned during propagation");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
