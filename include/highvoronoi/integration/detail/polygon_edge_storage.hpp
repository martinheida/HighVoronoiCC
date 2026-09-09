#pragma once

/**
 * @file polygon_edge_storage.hpp
 * @brief Rewindable cell-local storage for PolygonIntegrator edge occurrences.
 *
 * The ordinary PolygonIntegrator creates a large amount of temporary recursive
 * edge state while integrating one cell.  None of that state survives the cell
 * transaction, so heap-owning unordered_map nodes and shared_ptr payloads are
 * unnecessary.  This header keeps the same Julia semantics with three compact
 * worker-local structures:
 *
 * - PolygonEdgeDatabase stores self-describing immutable edge records in a
 *   reusable block stream: [signature length | signature | r1 | r2 | payload].
 * - PolygonPayloadDatabase stores mutable shared integral payload records.  An
 *   edge copy carries only the payload address, reproducing Julia PolyEdge's
 *   shared Vector value without reference-counted heap objects.
 * - PolygonEdgeDictionary owns only membership/index state for one logical Julia
 *   Dict.  It maps an exact signature to an edge-record address using open
 *   addressing.  Hashes locate candidates; the complete signature in the edge
 *   database is always compared before identity is accepted.
 *
 * Every object retains allocated capacity across rewind()/clear() calls.  One
 * PolygonWorkspace owns one pair of databases and many independent dictionaries.
 */

#include <highvoronoi/storage/hash/hash_generators.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::detail {

/** Default exact-dictionary fingerprint for ordinary PolygonIntegrator. */
using DefaultPolygonEdgeHashGenerator = UInt64HashGenerator<
    XXHash64<0x9e3779b97f4a7c15ULL>,
    MurmurHash64<0x243f6a8885a308d3ULL>,
    FNV1a64<>>;

namespace polygon_storage_detail {

/**
 * Rewindable block stream modeled after HVDataBase / NeighbourDatabase.
 *
 * Positions are counted in 16-bit units.  Rewind resets only the logical top;
 * allocated blocks remain available to the next cell.
 */
class RewindableUnitStream final {
public:
    using Unit = std::uint16_t;
    using size_type = std::size_t;

    explicit RewindableUnitStream(size_type unit_length = size_type{16384})
        : unit_length_(checked_unit_length(unit_length)) {}

    [[nodiscard]] size_type reserve(size_type units) {
        if (units > (std::numeric_limits<size_type>::max)() - top_) {
            throw std::overflow_error(
                "Polygon temporary database stream position overflow");
        }
        const size_type begin = top_;
        top_ += units;
        ensure_capacity(top_);
        return begin;
    }

    void rewind() noexcept { top_ = 0; }

    [[nodiscard]] size_type reserved_unit_count() const noexcept {
        return top_;
    }

    [[nodiscard]] size_type block_count() const noexcept {
        return data_.size();
    }

    [[nodiscard]] size_type block_unit_length() const noexcept {
        return unit_length_;
    }

    template <class Value>
    void write_value(size_type& position, const Value& value) {
        write_values(position, &value, size_type{1});
    }

    template <class Value>
    [[nodiscard]] Value read_value(size_type& position) const {
        Value value{};
        read_values(position, &value, size_type{1});
        return value;
    }

    template <class Value>
    void write_values(
        size_type& position,
        const Value* source,
        size_type count) {
        require_value<Value>();
        const size_type units = unit_count<Value>(count);
        require_extent(position, units);

        const auto* source_bytes =
            reinterpret_cast<const unsigned char*>(source);
        size_type remaining = units;

        while (remaining != size_type{0}) {
            const size_type block = position / unit_length_;
            const size_type offset = position % unit_length_;
            const size_type copied_units =
                (std::min)(remaining, unit_length_ - offset);
            const size_type copied_bytes = copied_units * sizeof(Unit);

            std::memcpy(
                data_[block].data() + offset,
                source_bytes,
                copied_bytes);

            source_bytes += copied_bytes;
            position += copied_units;
            remaining -= copied_units;
        }
    }

    template <class Value>
    void read_values(
        size_type& position,
        Value* destination,
        size_type count) const {
        require_value<Value>();
        const size_type units = unit_count<Value>(count);
        require_extent(position, units);

        auto* destination_bytes =
            reinterpret_cast<unsigned char*>(destination);
        size_type remaining = units;

        while (remaining != size_type{0}) {
            const size_type block = position / unit_length_;
            const size_type offset = position % unit_length_;
            const size_type copied_units =
                (std::min)(remaining, unit_length_ - offset);
            const size_type copied_bytes = copied_units * sizeof(Unit);

            std::memcpy(
                destination_bytes,
                data_[block].data() + offset,
                copied_bytes);

            destination_bytes += copied_bytes;
            position += copied_units;
            remaining -= copied_units;
        }
    }

    template <class Value>
    void skip_values(size_type& position, size_type count) const {
        require_value<Value>();
        const size_type units = unit_count<Value>(count);
        require_extent(position, units);
        position += units;
    }

private:
    [[nodiscard]] static size_type checked_unit_length(size_type value) {
        if (value == size_type{0}) {
            throw std::invalid_argument(
                "Polygon temporary database block length must be positive");
        }
        return value;
    }

    template <class Value>
    static constexpr void require_value() {
        static_assert(
            std::is_trivially_copyable_v<Value>,
            "Polygon temporary database values must be trivially copyable");
        static_assert(
            sizeof(Value) % sizeof(Unit) == 0,
            "Polygon temporary database values must use complete 16-bit units");
    }

    template <class Value>
    [[nodiscard]] static size_type unit_count(size_type count) {
        if (count >
            (std::numeric_limits<size_type>::max)() /
                (sizeof(Value) / sizeof(Unit))) {
            throw std::overflow_error(
                "Polygon temporary database record size overflow");
        }
        return count * sizeof(Value) / sizeof(Unit);
    }

    void ensure_capacity(size_type required_units) {
        if (capacity_ >= required_units) {
            return;
        }
        const size_type required_blocks =
            (required_units + unit_length_ - size_type{1}) / unit_length_;
        const size_type old_blocks = data_.size();
        data_.resize(required_blocks);
        for (size_type block = old_blocks; block < required_blocks; ++block) {
            data_[block].resize(unit_length_);
        }
        capacity_ = required_blocks * unit_length_;
    }

    void require_extent(size_type position, size_type units) const {
        if (position > top_ || units > top_ - position) {
            throw std::out_of_range(
                "Polygon temporary database access outside reserved storage");
        }
    }

    std::vector<std::vector<Unit>> data_;
    size_type unit_length_ = 0;
    size_type top_ = 0;
    size_type capacity_ = 0;
};

} // namespace polygon_storage_detail

/**
 * Self-describing immutable edge-occurrence database.
 *
 * Records are append-only within one cell.  Dictionaries may share one record
 * while their PolyEdge state is identical.  Extending an edge appends a new
 * record and changes only the relevant dictionary addresses, so independent
 * dictionary state is preserved without mutable aliasing.
 */
template <class IndexT, class PointT>
class PolygonEdgeDatabase final {
public:
    using Index = IndexT;
    using Point = PointT;
    using Scalar = typename Point::Scalar;
    using Address = std::size_t;
    using PayloadAddress = std::size_t;
    using size_type = std::size_t;

    explicit PolygonEdgeDatabase(
        size_type dimension,
        size_type unit_length = size_type{16384})
        : stream_(unit_length),
          dimension_(checked_dimension(dimension)) {
        static_assert(
            std::is_integral_v<Index> &&
                !std::is_same_v<std::remove_cv_t<Index>, bool>,
            "PolygonEdgeDatabase Index must be a non-bool integral type");
        static_assert(
            std::is_trivially_copyable_v<Scalar>,
            "PolygonEdgeDatabase point scalar must be trivially copyable");
    }

    template <class Key>
    [[nodiscard]] Address push(
        const Key& signature,
        const Point& first,
        const Point& second,
        PayloadAddress payload) {
        require_point(first);
        require_point(second);
        const size_type length = static_cast<size_type>(signature.size());
        const size_type units = record_units(length);
        size_type position = stream_.reserve(units);
        const size_type begin = position;

        stream_.write_value(position, length);
        for (size_type i = 0; i < length; ++i) {
            const Index value = static_cast<Index>(signature[i]);
            stream_.write_value(position, value);
        }
        stream_.write_values(position, first.data(), dimension_);
        stream_.write_values(position, second.data(), dimension_);
        stream_.write_value(position, payload);

        return begin + Address{1};
    }

    template <class Key>
    [[nodiscard]] Address push_point(
        const Key& signature,
        const Point& point,
        PayloadAddress payload) {
        return push(signature, point, point, payload);
    }

    template <class Key>
    [[nodiscard]] bool signature_equal(
        Address address,
        const Key& key) const {
        size_type position = checked_position(address);
        const size_type length = stream_.template read_value<size_type>(position);
        if (length != static_cast<size_type>(key.size())) {
            return false;
        }
        for (size_type i = 0; i < length; ++i) {
            const Index stored = stream_.template read_value<Index>(position);
            if (stored != static_cast<Index>(key[i])) {
                return false;
            }
        }
        return true;
    }

    void read_signature(
        Address address,
        std::vector<Index>& signature) const {
        size_type position = checked_position(address);
        const size_type length = stream_.template read_value<size_type>(position);
        signature.resize(length);
        stream_.read_values(position, signature.data(), length);
    }

    void read_edge(
        Address address,
        Point& first,
        Point& second,
        PayloadAddress& payload) const {
        prepare_point(first);
        prepare_point(second);
        size_type position = checked_position(address);
        const size_type length = stream_.template read_value<size_type>(position);
        stream_.template skip_values<Index>(position, length);
        stream_.read_values(position, first.data(), dimension_);
        stream_.read_values(position, second.data(), dimension_);
        payload = stream_.template read_value<PayloadAddress>(position);
    }

    void read(
        Address address,
        std::vector<Index>& signature,
        Point& first,
        Point& second,
        PayloadAddress& payload) const {
        prepare_point(first);
        prepare_point(second);
        size_type position = checked_position(address);
        const size_type length = stream_.template read_value<size_type>(position);
        signature.resize(length);
        stream_.read_values(position, signature.data(), length);
        stream_.read_values(position, first.data(), dimension_);
        stream_.read_values(position, second.data(), dimension_);
        payload = stream_.template read_value<PayloadAddress>(position);
    }

    void rewind() noexcept { stream_.rewind(); }

    [[nodiscard]] size_type reserved_unit_count() const noexcept {
        return stream_.reserved_unit_count();
    }

    [[nodiscard]] size_type block_count() const noexcept {
        return stream_.block_count();
    }

    [[nodiscard]] size_type block_unit_length() const noexcept {
        return stream_.block_unit_length();
    }

private:
    [[nodiscard]] static size_type checked_dimension(size_type dimension) {
        if (dimension == size_type{0}) {
            throw std::invalid_argument(
                "PolygonEdgeDatabase dimension must be positive");
        }
        return dimension;
    }

    void prepare_point(Point& point) const {
        if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
            point.resize(static_cast<Eigen::Index>(dimension_));
        }
    }

    void require_point(const Point& point) const {
        if (static_cast<size_type>(point.size()) != dimension_) {
            throw std::invalid_argument(
                "PolygonEdgeDatabase point dimension mismatch");
        }
    }

    [[nodiscard]] size_type checked_position(Address address) const {
        if (address == Address{0}) {
            throw std::out_of_range(
                "PolygonEdgeDatabase address zero is invalid");
        }
        const size_type position = address - Address{1};
        if (position >= stream_.reserved_unit_count()) {
            throw std::out_of_range(
                "PolygonEdgeDatabase address outside current cell storage");
        }
        return position;
    }

    [[nodiscard]] size_type record_units(size_type signature_length) const {
        constexpr size_type unit_size = sizeof(std::uint16_t);
        static_assert(sizeof(size_type) % unit_size == 0);
        static_assert(sizeof(Index) % unit_size == 0);
        static_assert(sizeof(Scalar) % unit_size == 0);
        static_assert(sizeof(PayloadAddress) % unit_size == 0);

        const size_type fixed =
            sizeof(size_type) / unit_size +
            (size_type{2} * dimension_) * sizeof(Scalar) / unit_size +
            sizeof(PayloadAddress) / unit_size;
        const size_type per_signature = sizeof(Index) / unit_size;
        if (signature_length >
            ((std::numeric_limits<size_type>::max)() - fixed) /
                per_signature) {
            throw std::overflow_error(
                "PolygonEdgeDatabase record size overflow");
        }
        return fixed + signature_length * per_signature;
    }

    polygon_storage_detail::RewindableUnitStream stream_;
    size_type dimension_ = 0;
};

/** Mutable shared payload records referenced by immutable edge records. */
template <class ScalarT>
class PolygonPayloadDatabase final {
public:
    using Scalar = ScalarT;
    using Address = std::size_t;
    using size_type = std::size_t;

    explicit PolygonPayloadDatabase(
        size_type components,
        size_type unit_length = size_type{4096})
        : stream_(unit_length),
          components_(components),
          zero_(components, Scalar{}) {
        static_assert(
            std::is_trivially_copyable_v<Scalar>,
            "PolygonPayloadDatabase scalar must be trivially copyable");
    }

    [[nodiscard]] Address allocate() {
        if (components_ == size_type{0}) {
            return Address{0};
        }
        const size_type units = record_units();
        size_type position = stream_.reserve(units);
        const size_type begin = position;
        stream_.write_value(position, components_);
        stream_.write_values(position, zero_.data(), components_);
        return begin + Address{1};
    }

    void write(Address address, const Scalar* values, size_type count) {
        require_components(count);
        if (components_ == size_type{0}) {
            if (address != Address{0}) {
                throw std::out_of_range(
                    "PolygonPayloadDatabase zero-component address must be zero");
            }
            return;
        }
        size_type position = checked_position(address);
        const size_type stored = stream_.template read_value<size_type>(position);
        if (stored != components_) {
            throw std::runtime_error(
                "PolygonPayloadDatabase stored component count mismatch");
        }
        stream_.write_values(position, values, components_);
    }

    void read(Address address, Scalar* values, size_type count) const {
        require_components(count);
        if (components_ == size_type{0}) {
            if (address != Address{0}) {
                throw std::out_of_range(
                    "PolygonPayloadDatabase zero-component address must be zero");
            }
            return;
        }
        size_type position = checked_position(address);
        const size_type stored = stream_.template read_value<size_type>(position);
        if (stored != components_) {
            throw std::runtime_error(
                "PolygonPayloadDatabase stored component count mismatch");
        }
        stream_.read_values(position, values, components_);
    }

    void rewind() noexcept { stream_.rewind(); }

    [[nodiscard]] size_type components() const noexcept { return components_; }
    [[nodiscard]] size_type reserved_unit_count() const noexcept {
        return stream_.reserved_unit_count();
    }
    [[nodiscard]] size_type block_count() const noexcept {
        return stream_.block_count();
    }

private:
    void require_components(size_type count) const {
        if (count != components_) {
            throw std::invalid_argument(
                "PolygonPayloadDatabase component count mismatch");
        }
    }

    [[nodiscard]] size_type checked_position(Address address) const {
        if (address == Address{0}) {
            throw std::out_of_range(
                "PolygonPayloadDatabase address zero is invalid");
        }
        const size_type position = address - Address{1};
        if (position >= stream_.reserved_unit_count()) {
            throw std::out_of_range(
                "PolygonPayloadDatabase address outside current cell storage");
        }
        return position;
    }

    [[nodiscard]] size_type record_units() const {
        constexpr size_type unit_size = sizeof(std::uint16_t);
        static_assert(sizeof(size_type) % unit_size == 0);
        static_assert(sizeof(Scalar) % unit_size == 0);
        const size_type per_component = sizeof(Scalar) / unit_size;
        if (components_ >
            ((std::numeric_limits<size_type>::max)() -
             sizeof(size_type) / unit_size) /
                per_component) {
            throw std::overflow_error(
                "PolygonPayloadDatabase record size overflow");
        }
        return sizeof(size_type) / unit_size + components_ * per_component;
    }

    polygon_storage_detail::RewindableUnitStream stream_;
    size_type components_ = 0;
    std::vector<Scalar> zero_;
};

/**
 * One logical Julia Dict: exact signature -> immutable edge-record address.
 *
 * Each dictionary owns an independent hash/index state, while all dictionaries
 * in one PolygonWorkspace point into the same edge database.  clear() is O(1)
 * with respect to hash capacity by advancing a generation counter; vector and
 * table allocations are retained for reuse by later cells.
 */
template <
    class EdgeDatabaseT,
    class HashGeneratorT = DefaultPolygonEdgeHashGenerator>
class PolygonEdgeDictionary final {
public:
    using EdgeDatabase = EdgeDatabaseT;
    using HashGenerator = HashGeneratorT;
    using Address = typename EdgeDatabase::Address;
    using size_type = std::size_t;

    struct Entry {
        HashGenerator hash{};
        Address address{0};
    };

    PolygonEdgeDictionary() = default;
    explicit PolygonEdgeDictionary(EdgeDatabase& database)
        : database_(&database) {}

    PolygonEdgeDictionary(const PolygonEdgeDictionary&) = delete;
    PolygonEdgeDictionary& operator=(const PolygonEdgeDictionary&) = delete;
    PolygonEdgeDictionary(PolygonEdgeDictionary&&) noexcept = default;
    PolygonEdgeDictionary& operator=(PolygonEdgeDictionary&&) noexcept = default;

    void bind(EdgeDatabase& database) {
        if (database_ != nullptr && database_ != &database) {
            throw std::logic_error(
                "PolygonEdgeDictionary cannot be rebound to another database");
        }
        database_ = &database;
    }

    void clear() noexcept {
        entries_.clear();
        consuming_ = false;
        ++generation_;
        if (generation_ == generation_type{0}) {
            for (Slot& slot : table_) {
                slot.generation = generation_type{0};
            }
            generation_ = generation_type{1};
        }
    }

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] size_type size() const noexcept { return entries_.size(); }

    [[nodiscard]] const std::vector<Entry>& entries() const noexcept {
        return entries_;
    }

    template <class Key>
    [[nodiscard]] bool find_address(const Key& key, Address& address) const {
        require_database();
        require_build_mode();
        if (table_.empty()) {
            return false;
        }
        const HashGenerator hash(key);
        const size_type capacity = table_.size();
        for (size_type probe = 0; probe < capacity; ++probe) {
            const size_type slot_index = hash.index(mask_, probe);
            const Slot& slot = table_[slot_index];
            if (slot.generation != generation_) {
                return false;
            }
            const Entry& entry = entries_.at(slot.entry_index);
            if (entry.hash != hash) {
                continue;
            }
            if (database_->signature_equal(entry.address, key)) {
                address = entry.address;
                return true;
            }
        }
        return false;
    }

    template <class Key>
    [[nodiscard]] bool contains(const Key& key) const {
        Address ignored = Address{0};
        return find_address(key, ignored);
    }

    /** Insert only when absent. Return true iff a new dictionary member was added. */
    template <class Key>
    bool insert(const Key& key, Address address) {
        return insert_impl(key, address, false);
    }

    /** Insert or replace this dictionary's occurrence address for an exact key. */
    template <class Key>
    bool insert_or_assign(const Key& key, Address address) {
        return insert_impl(key, address, true);
    }

    /**
     * Consume one dictionary member without maintaining the hash index.
     *
     * Polygon recursion never performs lookup/insert on a dictionary after it
     * starts Julia-style pop! consumption.  Avoiding tombstones/erase work is a
     * deliberate hot-path optimization.
     */
    [[nodiscard]] Address pop_address() {
        if (entries_.empty()) {
            throw std::out_of_range(
                "PolygonEdgeDictionary pop from empty dictionary");
        }
        consuming_ = true;
        const Address address = entries_.back().address;
        entries_.pop_back();
        return address;
    }

private:
    using generation_type = std::uint64_t;

    struct Slot {
        generation_type generation{0};
        size_type entry_index{0};
    };

    template <class Key>
    bool insert_impl(
        const Key& key,
        Address address,
        bool assign_existing) {
        require_database();
        require_build_mode();
        if (!database_->signature_equal(address, key)) {
            throw std::logic_error(
                "PolygonEdgeDictionary address/signature mismatch");
        }

        ensure_insert_capacity();
        const HashGenerator hash(key);
        const size_type capacity = table_.size();

        for (size_type probe = 0; probe < capacity; ++probe) {
            const size_type slot_index = hash.index(mask_, probe);
            Slot& slot = table_[slot_index];

            if (slot.generation != generation_) {
                const size_type entry_index = entries_.size();
                entries_.push_back(Entry{hash, address});
                slot.generation = generation_;
                slot.entry_index = entry_index;
                return true;
            }

            Entry& entry = entries_.at(slot.entry_index);
            if (entry.hash != hash) {
                continue;
            }
            if (!database_->signature_equal(entry.address, key)) {
                continue;
            }

            if (assign_existing) {
                entry.address = address;
            }
            return false;
        }

        throw std::logic_error(
            "PolygonEdgeDictionary failed to find a free probe slot");
    }

    void ensure_insert_capacity() {
        if (table_.empty()) {
            rehash(size_type{8});
            return;
        }
        if (entries_.size() + size_type{1} > table_.size() / size_type{2}) {
            rehash(table_.size() * size_type{2});
        }
    }

    void rehash(size_type requested_capacity) {
        const size_type capacity = next_power_of_two(
            (std::max)(requested_capacity, size_type{1}));
        std::vector<Slot> replacement(capacity);
        const size_type replacement_mask = capacity - size_type{1};

        for (size_type entry_index = 0;
             entry_index < entries_.size();
             ++entry_index) {
            const HashGenerator& hash = entries_[entry_index].hash;
            bool inserted = false;
            for (size_type probe = 0; probe < capacity; ++probe) {
                const size_type slot_index = hash.index(replacement_mask, probe);
                Slot& slot = replacement[slot_index];
                if (slot.generation == generation_type{0}) {
                    slot.generation = generation_;
                    slot.entry_index = entry_index;
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                throw std::logic_error(
                    "PolygonEdgeDictionary rehash failed");
            }
        }

        table_.swap(replacement);
        mask_ = capacity - size_type{1};
    }

    void require_database() const {
        if (database_ == nullptr) {
            throw std::logic_error(
                "PolygonEdgeDictionary is not bound to an edge database");
        }
    }

    void require_build_mode() const {
        if (consuming_) {
            throw std::logic_error(
                "PolygonEdgeDictionary lookup/insert after pop consumption began");
        }
    }

    EdgeDatabase* database_ = nullptr;
    std::vector<Entry> entries_;
    std::vector<Slot> table_;
    size_type mask_ = 0;
    generation_type generation_ = generation_type{1};
    bool consuming_ = false;
};

} // namespace highvoronoi::detail
