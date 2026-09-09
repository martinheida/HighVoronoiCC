#pragma once

/**
 * @file fast_polygon_cache.hpp
 * @brief Allocation-reusing exact facet caches for FastPolygonIntegrator.
 *
 * FastPolygon caches recursively computed subfacets by one canonical sorted
 * stable-internal generator key.  Hashing is used only to locate candidate
 * entries; cache identity is always verified against the complete stored key.
 * A hash collision therefore costs additional probing/key comparisons but can
 * never return another facet's volume/integral payload.
 *
 * Storage follows the useful part of HighVoronoi.jl's
 * MultiKeyDict -> KeyDict -> IndexHashTable layout:
 *
 * - append-only flat key storage per shard;
 * - append-only flat volume storage;
 * - append-only flat integral-component storage;
 * - an open-addressing hash index storing only hash fingerprints + entry index;
 * - optional sharding by the first key value using the project's existing
 *   DirectHash / StaticHash<N> / DynamicHash<> container policies;
 * - one shared backing state for all parallel FastPolygon workers.
 *
 * Compared with Julia's PolyBufferData{volume, Vector{Float64}}, C++ stores the
 * fixed-size integral payloads flat as well.  After capacity has stabilized,
 * cache hits and ordinary insertions require no per-entry heap allocation.
 */

#include <highvoronoi/parameters.hpp>
#include <highvoronoi/core/detail/locks.hpp>
#include <highvoronoi/storage/hash/fnv_hash.hpp>
#include <highvoronoi/storage/hash/hash_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * Default FastPolygon fingerprint.
 *
 * XXHash determines the initial probe position, MurmurHash64 determines the
 * odd probe step, and FNV1a contributes an additional independent fingerprint
 * used for candidate equality. Exact key verification remains mandatory even
 * when all three hashes agree.
 */
using DefaultFastPolygonFacetHashGenerator = UInt64HashGenerator<
    XXHash64<0x9e3779b97f4a7c15ULL>,
    MurmurHash64<0x243f6a8885a308d3ULL>,
    FNV1a64<>>;

/**
 * Public policy for the shared FastPolygon facet cache.
 *
 * Experienced users may replace the hash generator, sharding/container mode,
 * or shard lock without changing the FastPolygon recursion itself.
 *
 * Example:
 * @code{.cpp}
 * using Policy = highvoronoi::FastPolygonCachePolicy<
 *     highvoronoi::ExtendedHashGenerator<
 *         highvoronoi::Murmur128HashGenerator<17>,
 *         highvoronoi::XXHash64<23>,
 *         highvoronoi::FNV1a64<>>,
 *     highvoronoi::StaticHash<256>>;
 * @endcode
 */
template <
    class HashGeneratorT = DefaultFastPolygonFacetHashGenerator,
    class ContainerModeT = StaticHash<64>,
    class LockT = ReadWriteLock>
struct FastPolygonCachePolicy {
    using HashGenerator = HashGeneratorT;
    using ContainerMode = ContainerModeT;
    using Lock = LockT;

    ContainerMode container_mode{};

    FastPolygonCachePolicy() = default;

    explicit FastPolygonCachePolicy(ContainerMode mode)
        : container_mode(std::move(mode)) {}
};

} // namespace highvoronoi

namespace highvoronoi::detail {

struct FastPolygonCacheStats {
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t stores = 0;
    std::size_t entries = 0;

    /** Fingerprint matches rejected by the subsequent exact-key comparison. */
    std::size_t hash_collisions = 0;
};

namespace fast_polygon_cache_detail {

static constexpr std::size_t npos =
    (std::numeric_limits<std::size_t>::max)();

/**
 * Open-addressing index from HashGenerator fingerprint to flat payload index.
 *
 * No deletion occurs within a FastPolygon pass, so tombstones are unnecessary.
 * Multiple entries with exactly the same HashGenerator value are legal: the
 * caller's exact-key predicate decides whether an encountered candidate is the
 * requested facet or merely a full fingerprint collision.
 */
template <class HashGenerator>
class FacetIndexHashTable final {
public:
    struct LookupResult {
        std::size_t value_index = npos;
        std::size_t hash_collisions = 0;

        [[nodiscard]] bool found() const noexcept {
            return value_index != npos;
        }
    };

    explicit FacetIndexHashTable(std::size_t capacity = 8)
        : initial_capacity_(
              next_power_of_two((std::max)(capacity, std::size_t{1}))) {}

    FacetIndexHashTable(const FacetIndexHashTable&) = delete;
    FacetIndexHashTable& operator=(const FacetIndexHashTable&) = delete;
    FacetIndexHashTable(FacetIndexHashTable&&) = default;
    FacetIndexHashTable& operator=(FacetIndexHashTable&&) = default;

    void clear() noexcept {
        std::fill(occupied_.begin(), occupied_.end(), std::uint8_t{0});
        size_ = 0;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return table_.size(); }

    template <class ExactKeyEqual>
    [[nodiscard]] LookupResult find(
        const HashGenerator& hash,
        ExactKeyEqual&& exact_key_equal) const {
        LookupResult result;
        if (table_.empty()) {
            return result;
        }
        const std::size_t capacity_value = table_.size();

        for (std::size_t probe = 0; probe < capacity_value; ++probe) {
            const std::size_t slot = hash.index(mask_, probe);
            if (occupied_[slot] == 0) {
                return result;
            }

            const Entry& current = table_[slot];
            if (current.hash != hash) {
                continue;
            }

            if (exact_key_equal(current.value_index)) {
                result.value_index = current.value_index;
                return result;
            }

            // All configured hash fingerprints matched, but the canonical
            // facet key did not. This is a true hash overlap, not identity.
            ++result.hash_collisions;
        }

        return result;
    }

    /** Ensure the next insertion can publish without allocating/rehashing. */
    void prepare_insert() {
        if (table_.empty()) {
            rehash(initial_capacity_);
            return;
        }
        if ((size_ + std::size_t{1}) * std::size_t{2} > table_.size()) {
            rehash(table_.size() * std::size_t{2});
        }
    }

    /**
     * Publish a key known not to exist exactly. prepare_insert() must have been
     * called immediately before this operation under the shard write lock.
     */
    void insert_prepared(
        const HashGenerator& hash,
        std::size_t value_index) {
        const std::size_t capacity_value = table_.size();
        for (std::size_t probe = 0; probe < capacity_value; ++probe) {
            const std::size_t slot = hash.index(mask_, probe);
            if (occupied_[slot] == 0) {
                table_[slot] = Entry{hash, value_index};
                occupied_[slot] = 1;
                ++size_;
                return;
            }
        }
        throw std::logic_error(
            "FastPolygon facet index failed to find prepared free slot");
    }

private:
    struct Entry {
        HashGenerator hash{};
        std::size_t value_index = npos;
    };

    static void insert_rehashed(
        std::vector<Entry>& table,
        std::vector<std::uint8_t>& occupied,
        std::size_t mask,
        const Entry& entry) {
        const std::size_t capacity_value = table.size();
        for (std::size_t probe = 0; probe < capacity_value; ++probe) {
            const std::size_t slot = entry.hash.index(mask, probe);
            if (occupied[slot] == 0) {
                table[slot] = entry;
                occupied[slot] = 1;
                return;
            }
        }
        throw std::logic_error(
            "FastPolygon facet index rehash failed to find free slot");
    }

    void rehash(std::size_t requested_capacity) {
        const std::size_t new_capacity =
            next_power_of_two((std::max)(requested_capacity, std::size_t{1}));
        std::vector<Entry> new_table(new_capacity);
        std::vector<std::uint8_t> new_occupied(
            new_capacity,
            std::uint8_t{0});
        const std::size_t new_mask = new_capacity - std::size_t{1};

        for (std::size_t slot = 0; slot < table_.size(); ++slot) {
            if (occupied_[slot] != 0) {
                insert_rehashed(
                    new_table,
                    new_occupied,
                    new_mask,
                    table_[slot]);
            }
        }

        table_.swap(new_table);
        occupied_.swap(new_occupied);
        mask_ = new_mask;
    }

    std::size_t initial_capacity_ = 1;
    std::vector<Entry> table_;
    std::vector<std::uint8_t> occupied_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
};

struct ShardOperationResult {
    bool found_or_inserted = false;
    bool inserted = false;
    std::size_t hash_collisions = 0;
};

/** Flat append-only payload storage owned by one sharded hash table. */
template <
    class Index,
    class AreaScalar,
    class IntegralScalar,
    class HashGenerator>
class FlatFacetShard final {
public:
    using Key = std::vector<Index>;

    explicit FlatFacetShard(std::size_t hash_capacity = 8)
        : initial_payload_capacity_(
              (std::max)(std::size_t{4}, hash_capacity / std::size_t{2})),
          index_((std::max)(hash_capacity, std::size_t{1})) {}

    void reset(
        std::size_t key_length,
        std::size_t integral_components) {
        if (key_length == 0) {
            throw std::invalid_argument(
                "FastPolygon facet shard key length must be positive");
        }
        key_length_ = key_length;
        components_ = integral_components;
        keys_.clear();
        volumes_.clear();
        integrals_.clear();
        index_.clear();
    }

    [[nodiscard]] ShardOperationResult load(
        const Key& key,
        AreaScalar& volume,
        std::vector<IntegralScalar>& integral) const {
        require_key_layout(key);
        const HashGenerator hash(key);
        const auto lookup = index_.find(
            hash,
            [&](std::size_t value_index) {
                return exact_key_equal(value_index, key);
            });

        ShardOperationResult result;
        result.hash_collisions = lookup.hash_collisions;
        result.found_or_inserted = lookup.found();
        if (!lookup.found()) {
            return result;
        }

        const std::size_t entry = lookup.value_index;
        volume = volumes_.at(entry);
        integral.resize(components_);
        if (components_ != 0) {
            const std::size_t offset = entry * components_;
            std::copy_n(
                integrals_.data() + offset,
                components_,
                integral.data());
        }
        return result;
    }

    /** First exact publisher wins; hash collisions remain independent entries. */
    [[nodiscard]] ShardOperationResult store(
        const Key& key,
        AreaScalar volume,
        const std::vector<IntegralScalar>& integral) {
        require_key_layout(key);
        if (integral.size() != components_) {
            throw std::logic_error(
                "FastPolygon facet cache integral component mismatch");
        }

        const HashGenerator hash(key);
        const auto lookup = index_.find(
            hash,
            [&](std::size_t value_index) {
                return exact_key_equal(value_index, key);
            });

        ShardOperationResult result;
        result.hash_collisions = lookup.hash_collisions;
        result.found_or_inserted = true;
        if (lookup.found()) {
            return result;
        }

        // Make every potentially allocating operation happen before mutating
        // the append-only payload vectors. After these reserves, primitive
        // payload publication and hash-slot publication are allocation-free.
        index_.prepare_insert();
        const std::size_t new_entry = volumes_.size();
        prepare_payload_insert(new_entry + std::size_t{1});

        const std::size_t old_key_size = keys_.size();
        const std::size_t old_volume_size = volumes_.size();
        const std::size_t old_integral_size = integrals_.size();

        try {
            keys_.insert(keys_.end(), key.begin(), key.end());
            volumes_.push_back(volume);
            integrals_.insert(
                integrals_.end(),
                integral.begin(),
                integral.end());
            index_.insert_prepared(hash, new_entry);
        } catch (...) {
            keys_.resize(old_key_size);
            volumes_.resize(old_volume_size);
            integrals_.resize(old_integral_size);
            throw;
        }

        result.inserted = true;
        return result;
    }

    [[nodiscard]] std::size_t entry_count() const noexcept {
        return volumes_.size();
    }

    [[nodiscard]] std::size_t key_capacity_entries() const noexcept {
        return key_length_ == 0 ? 0 : keys_.capacity() / key_length_;
    }

    [[nodiscard]] std::size_t integral_capacity_entries() const noexcept {
        return components_ == 0
            ? volumes_.capacity()
            : integrals_.capacity() / components_;
    }

private:
    void prepare_payload_insert(std::size_t required_entries) {
        std::size_t available_entries = volumes_.capacity();
        if (key_length_ != 0) {
            available_entries = (std::min)(
                available_entries,
                keys_.capacity() / key_length_);
        }
        if (components_ != 0) {
            available_entries = (std::min)(
                available_entries,
                integrals_.capacity() / components_);
        }

        if (required_entries <= available_entries) {
            return;
        }

        const std::size_t doubled = available_entries == 0
            ? initial_payload_capacity_
            : available_entries * std::size_t{2};
        const std::size_t target_entries = (std::max)(
            required_entries,
            (std::max)(initial_payload_capacity_, doubled));

        if (keys_.capacity() < target_entries * key_length_) {
            keys_.reserve(target_entries * key_length_);
        }
        if (volumes_.capacity() < target_entries) {
            volumes_.reserve(target_entries);
        }
        if (components_ != 0 &&
            integrals_.capacity() < target_entries * components_) {
            integrals_.reserve(target_entries * components_);
        }
    }

    void require_key_layout(const Key& key) const {
        if (key_length_ == 0) {
            throw std::logic_error(
                "FastPolygon facet shard was not initialized for a pass");
        }
        if (key.size() != key_length_) {
            throw std::logic_error(
                "FastPolygon facet key length does not match hierarchy level");
        }
    }

    [[nodiscard]] bool exact_key_equal(
        std::size_t value_index,
        const Key& key) const noexcept {
        const std::size_t offset = value_index * key_length_;
        if (offset + key_length_ > keys_.size()) {
            return false;
        }
        return std::equal(
            key.begin(),
            key.end(),
            keys_.begin() + static_cast<std::ptrdiff_t>(offset));
    }

    std::size_t initial_payload_capacity_ = 4;
    std::size_t key_length_ = 0;
    std::size_t components_ = 0;
    FacetIndexHashTable<HashGenerator> index_;
    std::vector<Index> keys_;
    std::vector<AreaScalar> volumes_;
    std::vector<IntegralScalar> integrals_;
};

// -------------------------------------------------------------------------
// Shard containers using the existing HighVoronoi container-mode policies.
// -------------------------------------------------------------------------

template <class Shard, class ContainerMode>
class FacetShardContainer;

template <class Shard>
class FacetShardContainer<Shard, DirectHash> final {
public:
    explicit FacetShardContainer(DirectHash mode)
        : mode_(mode),
          shard_(std::make_unique<Shard>(mode_.hash_capacity)) {}

    template <class Key>
    [[nodiscard]] Shard* find(const Key&) noexcept {
        return shard_.get();
    }

    template <class Key>
    [[nodiscard]] Shard& ensure(
        const Key&,
        std::size_t,
        std::size_t) {
        return *shard_;
    }

    void reset_existing(std::size_t key_length, std::size_t components) {
        if (shard_) {
            shard_->reset(key_length, components);
        }
    }

private:
    DirectHash mode_;
    std::unique_ptr<Shard> shard_;
};

template <class Shard, std::size_t N>
class FacetShardContainer<Shard, StaticHash<N>> final {
public:
    explicit FacetShardContainer(StaticHash<N> mode)
        : mode_(mode) {
        for (auto& shard : shards_) {
            shard = std::make_unique<Shard>(mode_.hash_capacity);
        }
    }

    template <class Key>
    [[nodiscard]] Shard* find(const Key& key) noexcept {
        return shards_[table_index(key)].get();
    }

    template <class Key>
    [[nodiscard]] Shard& ensure(
        const Key& key,
        std::size_t,
        std::size_t) {
        return *shards_[table_index(key)];
    }

    void reset_existing(std::size_t key_length, std::size_t components) {
        for (auto& shard : shards_) {
            if (shard) {
                shard->reset(key_length, components);
            }
        }
    }

private:
    template <class Key>
    [[nodiscard]] static std::size_t table_index(const Key& key) {
        require_nonempty_key(key);
        return static_cast<std::size_t>(key[0]) % N;
    }

    template <class Key>
    static void require_nonempty_key(const Key& key) {
        if (key.empty()) {
            throw std::invalid_argument(
                "FastPolygon cache cannot shard an empty facet key");
        }
    }

    StaticHash<N> mode_;
    std::array<std::unique_ptr<Shard>, N> shards_{};
};

template <class Shard, class ResizeLock>
class FacetShardContainer<Shard, DynamicHash<ResizeLock>> final {
public:
    using StructuralLock = std::conditional_t<
        std::is_void_v<ResizeLock>,
        ReadWriteLock,
        ResizeLock>;

    explicit FacetShardContainer(DynamicHash<ResizeLock> mode)
        : mode_(mode) {
        shards_.resize(mode_.initial_table_count);
    }

    template <class Key>
    [[nodiscard]] Shard* find(const Key& key) {
        const std::size_t index = table_index(key);
        ReadLockGuard<StructuralLock> guard(structural_lock_);
        if (index >= shards_.size()) {
            return nullptr;
        }
        return shards_[index].get();
    }

    template <class Key>
    [[nodiscard]] Shard& ensure(
        const Key& key,
        std::size_t key_length,
        std::size_t components) {
        const std::size_t index = table_index(key);
        {
            ReadLockGuard<StructuralLock> guard(structural_lock_);
            if (index < shards_.size() && shards_[index]) {
                return *shards_[index];
            }
        }

        WriteLockGuard<StructuralLock> guard(structural_lock_);
        if (index >= shards_.size()) {
            shards_.resize(index + std::size_t{1});
        }
        if (!shards_[index]) {
            shards_[index] =
                std::make_unique<Shard>(mode_.hash_capacity);
            shards_[index]->reset(key_length, components);
        }
        return *shards_[index];
    }

    void reset_existing(std::size_t key_length, std::size_t components) {
        WriteLockGuard<StructuralLock> guard(structural_lock_);
        for (auto& shard : shards_) {
            if (shard) {
                shard->reset(key_length, components);
            }
        }
    }

private:
    template <class Key>
    [[nodiscard]] std::size_t table_index(const Key& key) const {
        if (key.empty()) {
            throw std::invalid_argument(
                "FastPolygon cache cannot shard an empty facet key");
        }
        return static_cast<std::size_t>(
            static_cast<std::uint64_t>(key[0]) / mode_.block_size);
    }

    DynamicHash<ResizeLock> mode_;
    std::vector<std::unique_ptr<Shard>> shards_;
    mutable StructuralLock structural_lock_{};
};

} // namespace fast_polygon_cache_detail

/**
 * Serial exact flat facet store.
 *
 * This replaces unordered_map<vector<Index>, PolyBufferData> by one flat
 * append-only shard per hierarchy level. Key vectors supplied by
 * FastPolygonWorkspace are persistent scratch buffers and are not retained.
 */
template <
    class Index,
    class AreaScalar,
    class IntegralScalar,
    class HashGenerator = DefaultFastPolygonFacetHashGenerator>
class SerialFastPolygonFacetStore final {
public:
    using Key = std::vector<Index>;
    using Shard = fast_polygon_cache_detail::FlatFacetShard<
        Index,
        AreaScalar,
        IntegralScalar,
        HashGenerator>;

    explicit SerialFastPolygonFacetStore(std::size_t hash_capacity = 8)
        : hash_capacity_(hash_capacity) {}

    void begin_pass(
        std::size_t hierarchy_levels,
        std::size_t integral_components) {
        components_ = integral_components;
        while (levels_.size() < hierarchy_levels) {
            levels_.push_back(std::make_unique<Shard>(hash_capacity_));
        }
        for (std::size_t level = 0; level < levels_.size(); ++level) {
            if (level < hierarchy_levels) {
                levels_[level]->reset(level + std::size_t{3}, components_);
            }
        }
        active_levels_ = hierarchy_levels;
        stats_ = {};
    }

    [[nodiscard]] bool load(
        std::size_t level,
        const Key& key,
        AreaScalar& volume,
        std::vector<IntegralScalar>& integral) {
        Shard& shard = require_level(level);
        const auto result = shard.load(key, volume, integral);
        stats_.hash_collisions += result.hash_collisions;
        if (result.found_or_inserted) {
            ++stats_.hits;
            return true;
        }
        ++stats_.misses;
        return false;
    }

    void store(
        std::size_t level,
        const Key& key,
        AreaScalar volume,
        const std::vector<IntegralScalar>& integral) {
        Shard& shard = require_level(level);
        const auto result = shard.store(key, volume, integral);
        ++stats_.stores;
        stats_.hash_collisions += result.hash_collisions;
        if (result.inserted) {
            ++stats_.entries;
        }
    }

    [[nodiscard]] const FastPolygonCacheStats& stats() const noexcept {
        return stats_;
    }

    [[nodiscard]] std::size_t hierarchy_levels() const noexcept {
        return active_levels_;
    }

private:
    [[nodiscard]] Shard& require_level(std::size_t level) {
        if (level >= active_levels_ || level >= levels_.size()) {
            throw std::out_of_range(
                "FastPolygon serial cache hierarchy level outside active pass");
        }
        return *levels_[level];
    }

    std::size_t hash_capacity_ = 8;
    std::vector<std::unique_ptr<Shard>> levels_;
    std::size_t active_levels_ = 0;
    std::size_t components_ = 0;
    FastPolygonCacheStats stats_{};
};

/**
 * Parallel shared exact flat facet store.
 *
 * Copies are cheap handles to one shared State, so all FastPolygon workers see
 * the same facet database. ContainerMode routes canonical keys to independent
 * shards before hashing, reducing lock contention and hash-table population.
 */
template <
    class Index,
    class AreaScalar,
    class IntegralScalar,
    class Policy = FastPolygonCachePolicy<>>
class SharedFastPolygonFacetStore final {
public:
    using Key = std::vector<Index>;
    using HashGenerator = typename Policy::HashGenerator;
    using ContainerMode = typename Policy::ContainerMode;
    using Lock = typename Policy::Lock;
    using FlatShard = fast_polygon_cache_detail::FlatFacetShard<
        Index,
        AreaScalar,
        IntegralScalar,
        HashGenerator>;

private:
    struct Shard final {
        explicit Shard(std::size_t hash_capacity)
            : data(hash_capacity) {}

        void reset(std::size_t key_length, std::size_t components) {
            data.reset(key_length, components);
        }

        FlatShard data;
        mutable Lock lock{};
    };

    using Container = fast_polygon_cache_detail::FacetShardContainer<
        Shard,
        ContainerMode>;

    struct Level final {
        explicit Level(ContainerMode mode)
            : shards(std::move(mode)) {}
        Container shards;
    };

    struct State final {
        explicit State(Policy policy_value)
            : policy(std::move(policy_value)) {}

        Policy policy;
        std::vector<std::unique_ptr<Level>> levels;
        std::size_t active_levels = 0;
        std::size_t components = 0;
        ReadWriteLock lifecycle_lock{};
        std::atomic<std::size_t> hits{0};
        std::atomic<std::size_t> misses{0};
        std::atomic<std::size_t> stores{0};
        std::atomic<std::size_t> entries{0};
        std::atomic<std::size_t> hash_collisions{0};
    };

public:
    SharedFastPolygonFacetStore()
        : state_(std::make_shared<State>(Policy{})) {}

    explicit SharedFastPolygonFacetStore(Policy policy)
        : state_(std::make_shared<State>(std::move(policy))) {}

    void begin_pass(
        std::size_t hierarchy_levels,
        std::size_t integral_components,
        std::size_t /*legacy_shard_hint*/ = 0) {
        State& state = *state_;
        WriteLockGuard<ReadWriteLock> guard(state.lifecycle_lock);

        state.components = integral_components;
        state.active_levels = hierarchy_levels;
        while (state.levels.size() < hierarchy_levels) {
            state.levels.push_back(std::make_unique<Level>(
                state.policy.container_mode));
        }

        for (std::size_t level = 0; level < state.levels.size(); ++level) {
            if (level < hierarchy_levels) {
                state.levels[level]->shards.reset_existing(
                    level + std::size_t{3},
                    integral_components);
            }
        }

        state.hits.store(0, std::memory_order_relaxed);
        state.misses.store(0, std::memory_order_relaxed);
        state.stores.store(0, std::memory_order_relaxed);
        state.entries.store(0, std::memory_order_relaxed);
        state.hash_collisions.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] bool load(
        std::size_t level,
        const Key& key,
        AreaScalar& volume,
        std::vector<IntegralScalar>& integral) {
        State& state = *state_;
        Level& cache_level = require_level(state, level);
        Shard* shard = cache_level.shards.find(key);
        if (shard == nullptr) {
            state.misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        ReadLockGuard<Lock> guard(shard->lock);
        const auto result = shard->data.load(key, volume, integral);
        state.hash_collisions.fetch_add(
            result.hash_collisions,
            std::memory_order_relaxed);
        if (result.found_or_inserted) {
            state.hits.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        state.misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    void store(
        std::size_t level,
        const Key& key,
        AreaScalar volume,
        const std::vector<IntegralScalar>& integral) {
        State& state = *state_;
        Level& cache_level = require_level(state, level);
        Shard& shard = cache_level.shards.ensure(
            key,
            level + std::size_t{3},
            state.components);

        WriteLockGuard<Lock> guard(shard.lock);
        const auto result = shard.data.store(key, volume, integral);
        state.stores.fetch_add(1, std::memory_order_relaxed);
        state.hash_collisions.fetch_add(
            result.hash_collisions,
            std::memory_order_relaxed);
        if (result.inserted) {
            state.entries.fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] FastPolygonCacheStats stats() const noexcept {
        const State& state = *state_;
        return FastPolygonCacheStats{
            state.hits.load(std::memory_order_relaxed),
            state.misses.load(std::memory_order_relaxed),
            state.stores.load(std::memory_order_relaxed),
            state.entries.load(std::memory_order_relaxed),
            state.hash_collisions.load(std::memory_order_relaxed)};
    }

    [[nodiscard]] std::size_t hierarchy_levels() const noexcept {
        return state_->active_levels;
    }

    [[nodiscard]] bool shares_state_with(
        const SharedFastPolygonFacetStore& other) const noexcept {
        return state_.get() == other.state_.get();
    }

    [[nodiscard]] const Policy& policy() const noexcept {
        return state_->policy;
    }

private:
    [[nodiscard]] static Level& require_level(
        State& state,
        std::size_t level) {
        if (level >= state.active_levels || level >= state.levels.size()) {
            throw std::out_of_range(
                "FastPolygon shared cache hierarchy level outside active pass");
        }
        return *state.levels[level];
    }

    std::shared_ptr<State> state_;
};

} // namespace highvoronoi::detail

namespace highvoronoi {

/** Public alias for expert replacement/injection of the serial facet store. */
template <
    class Index,
    class AreaScalar,
    class IntegralScalar,
    class HashGenerator = DefaultFastPolygonFacetHashGenerator>
using FastPolygonSerialFacetStore = detail::SerialFastPolygonFacetStore<
    Index,
    AreaScalar,
    IntegralScalar,
    HashGenerator>;

/** Public alias for expert replacement/injection of the parallel shared store. */
template <
    class Index,
    class AreaScalar,
    class IntegralScalar,
    class Policy = FastPolygonCachePolicy<>>
using FastPolygonSharedFacetStore = detail::SharedFastPolygonFacetStore<
    Index,
    AreaScalar,
    IntegralScalar,
    Policy>;

} // namespace highvoronoi
