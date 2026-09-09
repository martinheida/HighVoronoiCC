
#include <highvoronoi/integration/detail/fast_polygon_cache.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Store = highvoronoi::detail::SerialFastPolygonFacetStore<
    std::uint32_t,
    double,
    double>;
using SharedStore = highvoronoi::detail::SharedFastPolygonFacetStore<
    std::uint32_t,
    double,
    double>;


/** Hash generator that deliberately maps every possible key to one fingerprint. */
class DeliberateCollisionHash final {
public:
    DeliberateCollisionHash() = default;

    template <class Key>
    explicit DeliberateCollisionHash(const Key&) {}

    [[nodiscard]] std::size_t index(
        std::size_t mask,
        std::size_t probe) const noexcept {
        return probe & mask;
    }

    [[nodiscard]] bool operator==(
        const DeliberateCollisionHash&) const noexcept {
        return true;
    }

    [[nodiscard]] bool operator!=(
        const DeliberateCollisionHash&) const noexcept {
        return false;
    }
};

using CollisionPolicy = highvoronoi::FastPolygonCachePolicy<
    DeliberateCollisionHash,
    highvoronoi::StaticHash<1>>;
using CollisionStore = highvoronoi::detail::SharedFastPolygonFacetStore<
    std::uint32_t,
    double,
    double,
    CollisionPolicy>;

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool ok, std::string_view message) {
    ++checks;
    if (ok) {
        std::cout << "[OK]   " << message << '\n';
    } else {
        ++failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}
}

int main() {
    Store store;
    store.begin_pass(2, 2);

    const Store::Key key{3, 7, 11};
    double volume = 0.0;
    std::vector<double> integral(2, 0.0);

    check(!store.load(0, key, volume, integral),
          "fresh serial FastPolygon facet store misses");

    store.store(0, key, 2.5, std::vector<double>{4.0, 9.0});
    check(store.load(0, key, volume, integral),
          "stored facet is found in the same pass");
    check(std::abs(volume - 2.5) < 1e-15 &&
              integral.size() == 2 &&
              std::abs(integral[0] - 4.0) < 1e-15 &&
              std::abs(integral[1] - 9.0) < 1e-15,
          "facet volume and integral payload round-trip exactly");

    const auto stats = store.stats();
    check(stats.hits == 1 && stats.misses == 1 &&
              stats.stores == 1 && stats.entries == 1,
          "cache statistics distinguish hit, miss and unique entry");

    store.begin_pass(2, 2);
    volume = 0.0;
    std::fill(integral.begin(), integral.end(), 0.0);
    check(!store.load(0, key, volume, integral),
          "begin_pass invalidates old geometric facet values");

    std::cout << "\n[TEST] shared Julia-style FastPolygon facet store\n";
    SharedStore shared;
    shared.begin_pass(2, 2, 16);
    SharedStore worker_copy = shared;
    check(shared.shares_state_with(worker_copy),
          "worker store copies share one backing cache state");

    constexpr std::size_t worker_count = 8;
    const SharedStore::Key shared_key{3, 7, 11};
    std::atomic<std::size_t> ready{0};
    std::atomic<std::size_t> unexpected_hits{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            SharedStore local = shared;
            double local_volume = 0.0;
            std::vector<double> local_integral(2, 0.0);
            const bool found = local.load(
                0, shared_key, local_volume, local_integral);
            if (found) {
                unexpected_hits.fetch_add(1, std::memory_order_relaxed);
            }
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            local.store(
                0,
                shared_key,
                static_cast<double>(worker + 1),
                std::vector<double>{
                    static_cast<double>(worker),
                    static_cast<double>(worker * worker)});
        });
    }

    while (ready.load(std::memory_order_acquire) != worker_count) {
        std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }

    double published_volume = 0.0;
    std::vector<double> published_integral;
    check(unexpected_hits.load(std::memory_order_relaxed) == 0,
          "all workers observe the planned cache miss before concurrent publication");
    check(shared.load(
              0,
              shared_key,
              published_volume,
              published_integral),
          "one concurrently published facet is readable by every handle");

    const auto shared_stats = shared.stats();
    check(shared_stats.entries == 1 &&
              shared_stats.stores == worker_count &&
              shared_stats.misses == worker_count &&
              shared_stats.hits == 1,
          "concurrent duplicate publication keeps one shared cache entry");

    const double first_published_volume = published_volume;
    shared.store(0, shared_key, 999.0, std::vector<double>{999.0, 999.0});
    published_volume = 0.0;
    published_integral.clear();
    (void)shared.load(0, shared_key, published_volume, published_integral);
    check(std::abs(published_volume - first_published_volume) < 1e-15,
          "parallel shared cache preserves Julia KeyDict first-writer semantics");

    const SharedStore::Key other_shard_key{9, 10, 12, 14};
    shared.store(1, other_shard_key, 4.0, std::vector<double>{5.0, 6.0});
    double other_volume = 0.0;
    std::vector<double> other_integral;
    check(shared.load(1, other_shard_key, other_volume, other_integral) &&
              std::abs(other_volume - 4.0) < 1e-15,
          "different first-key shard and hierarchy level remain independently readable");

    shared.begin_pass(2, 2, 16);
    published_volume = 0.0;
    published_integral.assign(2, 0.0);
    check(!shared.load(0, shared_key, published_volume, published_integral),
          "shared facet cache is cleared once at the next integration pass");

    std::cout << "\n[TEST] exact identity despite deliberate full hash collision\n";
    CollisionStore collision_store(CollisionPolicy{
        highvoronoi::StaticHash<1>{4}});
    collision_store.begin_pass(1, 2);
    const CollisionStore::Key collision_a{3, 7, 11};
    const CollisionStore::Key collision_b{3, 8, 12};
    collision_store.store(0, collision_a, 1.25, std::vector<double>{2.0, 3.0});
    collision_store.store(0, collision_b, 4.75, std::vector<double>{5.0, 6.0});

    double collision_volume_a = 0.0;
    double collision_volume_b = 0.0;
    std::vector<double> collision_integral_a;
    std::vector<double> collision_integral_b;
    const bool found_a = collision_store.load(
        0, collision_a, collision_volume_a, collision_integral_a);
    const bool found_b = collision_store.load(
        0, collision_b, collision_volume_b, collision_integral_b);

    check(found_a && found_b &&
              std::abs(collision_volume_a - 1.25) < 1e-15 &&
              std::abs(collision_volume_b - 4.75) < 1e-15 &&
              collision_integral_a == std::vector<double>({2.0, 3.0}) &&
              collision_integral_b == std::vector<double>({5.0, 6.0}),
          "full canonical key verification separates keys with identical hashes");
    const auto collision_stats = collision_store.stats();
    check(collision_stats.entries == 2 && collision_stats.hash_collisions > 0,
          "hash overlaps are observable diagnostics but never cache identity");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
