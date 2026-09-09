

#pragma once

/**
 * @file hvdatabase_direct.hpp
 * @brief Append-only block database for HighVoronoi signatures and geometry data.
 *
 * HVDataBase stores records of the forms
 *
 *     sigma, r
 *
 * and
 *
 *     sigma, r, u
 *
 * in a sequence of fixed-size blocks. The element types are taken directly
 * from DataBaseParams:
 *
 * - sigma contains Index values.
 * - r and u contain Scalar values.
 *
 * The supplied vector-like objects must provide contiguous storage through
 * data() and report their number of elements through size(). No element
 * conversion is performed by the database. Complete Index and Scalar ranges
 * are copied directly with memcpy, including records that cross block
 * boundaries.
 *
 * The database is append-only. A QueueHash structure detects duplicate
 * signatures. erase() removes the signature from the hash structure and marks
 * the stored record as deleted without reclaiming its storage.
 */

#include <highvoronoi/storage/hash/hash_types.hpp>
#include <highvoronoi/core/detail/locks.hpp>
#include <highvoronoi/core/point.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

namespace detail {

/** Plain serial counter with the same small API as std::atomic. */
template <class T>
class PlainCounter final {
public:
    constexpr PlainCounter() noexcept = default;
    explicit constexpr PlainCounter(T value) noexcept : value_(value) {}

    [[nodiscard]] T load() const noexcept { return value_; }
    void store(T value) noexcept { value_ = value; }
    [[nodiscard]] T fetch_add(T increment) noexcept {
        const T previous = value_;
        value_ += increment;
        return previous;
    }

private:
    T value_{};
};

template <class Lock, class T>
using LockAwareCounter = std::conditional_t<
    std::is_same_v<Lock, EmptyLock>,
    PlainCounter<T>,
    std::atomic<T>>;

} // namespace detail

/**
 * @brief Append-only block database for signatures and geometric vectors.
 *
 * Each normal record has the layout
 *
 *     sigma length, sigma..., r...
 *
 * Each facet record has the layout
 *
 *     sigma length, sigma..., r..., u...
 *
 * Storage positions are counted in 16-bit units. Public addresses are
 * one-based, so address 0 can be used to report that a signature already
 * exists and no record was written.
 *
 * The atomic top counter reserves a separate storage range for every writer.
 * The write lock is used only when the outer block vector must grow. Ordinary
 * reads and writes hold a shared lock so that the block structure cannot be
 * resized while it is being accessed.
 *
 * @tparam Lock Lock type used for structural synchronization.
 * @tparam DataBaseParams Compile-time and runtime database configuration.
 */
template<class Lock, class DataBaseParams, int Dim>
class HVDataBase final {
public:
    using size_t = std::size_t;
    using UInt16 = std::uint16_t;

    using Parameters = DataBaseParams;
    using Scalar = typename Parameters::Scalar;
    using Index = typename Parameters::Index;
    using Sigma = std::vector<Index>;
    using VertexPoint = PointType<Scalar, Dim>;
    using address_type = std::size_t;

    static constexpr int DimensionAtCompileTime = Dim;

    using QueueHash =
        typename detail::QueueHashFromParams<Lock, DataBaseParams>::type;

    using DataUnit = std::vector<UInt16>;

    using LockType = Lock;
    using ReadGuard = detail::ReadLockGuard<Lock>;
    using WriteGuard = detail::WriteLockGuard<Lock>;

    static constexpr size_t scalar_unit_count =
        sizeof(Scalar) / sizeof(UInt16);

    static constexpr size_t index_unit_count =
        sizeof(Index) / sizeof(UInt16);

    static_assert(
        sizeof(Scalar) == 2 ||
        sizeof(Scalar) == 4 ||
        sizeof(Scalar) == 8,
        "Scalar must have 16, 32, or 64 bits");

    static_assert(
        sizeof(Index) == 2 ||
        sizeof(Index) == 4 ||
        sizeof(Index) == 8,
        "Index must have 16, 32, or 64 bits");

    static_assert(
        std::is_trivially_copyable_v<Scalar>,
        "Scalar must be trivially copyable");

    static_assert(
        std::is_trivially_copyable_v<Index>,
        "Index must be trivially copyable");

    /**
     * @brief Create an empty database.
     *
     * @param unit_length Number of 16-bit units in each storage block.
     * @param parameters Runtime parameters used to construct the QueueHash.
     */
    explicit HVDataBase(
        size_t unit_length,
        const DataBaseParams& parameters,
        size_t runtime_dimension =
            Dim == Dynamic ? size_t{0} : static_cast<size_t>(Dim))
        : unit_length(unit_length),
          keys(detail::make_queue_hash<Lock>(parameters)),
          dimension_(checked_dimension(runtime_dimension))
    {}

    [[nodiscard]] size_t dimension() const noexcept
    {
        return dimension_;
    }

    /** @brief Store one finite Voronoi vertex. */
    [[nodiscard]] size_t push(
        const VertexPoint& r,
        const Sigma& sigma)
    {
        require_vertex_point(r);
        if (keys.pushqueue(sigma, true)) {
            return 0;
        }

        const size_t units =
            index_unit_count * (sigma.size() + 1) +
            scalar_unit_count * dimension_;

        const size_t begin = reserve_units(units);
        ensure_capacity(begin + units);

        ReadGuard guard(lock);
        size_t position = begin;

        const Index sigma_length =
            static_cast<Index>(sigma.size());

        write_value(position, sigma_length);
        write_values(position, sigma.data(), sigma.size());
        write_values(position, r.data(), dimension_);

        return begin + 1;
    }

    /** @brief Read one finite Voronoi vertex into caller-owned storage. */
    void read(
        size_t address,
        VertexPoint& r,
        Sigma& sigma) const
    {
        prepare_vertex_point(r);

        ReadGuard guard(lock);
        size_t position = address - 1;

        const Index stored_length =
            read_value<Index>(position);

        const size_t sigma_length =
            static_cast<size_t>(stored_length);

        sigma.resize(sigma_length);

        if (stored_length == Index{0}) {
            return;
        }

        read_values(position, sigma.data(), sigma_length);
        read_values(position, r.data(), dimension_);
    }

    [[nodiscard]] bool contains(const Sigma& sigma) const
    {
        return keys.contains(sigma);
    }

    /** Register a persistent signature without storing a payload record. */
    [[nodiscard]] bool register_signature(const Sigma& sigma)
    {
        return !keys.pushqueue(sigma, true);
    }

    /** Remove a signature from the persistent identity hash only. */
    [[nodiscard]] bool erase_signature(const Sigma& sigma)
    {
        return keys.erase(sigma);
    }

    /** @brief Store one facet/infinite-edge record. */
    [[nodiscard]] size_t push_facet(
        const VertexPoint& r,
        Sigma& sigma,
        const VertexPoint& u)
    {
        require_vertex_point(r);
        require_vertex_point(u);
        if (sigma.size() > 1) {
            std::sort(sigma.begin(), sigma.end());
        }

        if (keys.pushqueue(sigma, true)) {
            return 0;
        }

        const size_t units =
            index_unit_count * (sigma.size() + 1) +
            scalar_unit_count * (size_t{2} * dimension_);

        const size_t begin = reserve_units(units);
        ensure_capacity(begin + units);

        ReadGuard guard(lock);
        size_t position = begin;

        const Index sigma_length =
            static_cast<Index>(sigma.size());

        write_value(position, sigma_length);
        write_values(position, sigma.data(), sigma.size());
        write_values(position, r.data(), dimension_);
        write_values(position, u.data(), dimension_);

        return begin + 1;
    }

    /** @brief Read one facet/infinite-edge record into caller-owned storage. */
    void read_facet(
        size_t address,
        VertexPoint& r,
        Sigma& sigma,
        VertexPoint& u) const
    {
        prepare_vertex_point(r);
        prepare_vertex_point(u);

        ReadGuard guard(lock);
        size_t position = address - 1;

        const Index stored_length =
            read_value<Index>(position);

        const size_t sigma_length =
            static_cast<size_t>(stored_length);

        sigma.resize(sigma_length);

        if (stored_length == Index{0}) {
            return;
        }

        read_values(position, sigma.data(), sigma_length);
        read_values(position, r.data(), dimension_);
        read_values(position, u.data(), dimension_);
    }

    bool erase(
        size_t address,
        const Sigma& sigma)
    {
        if (!keys.erase(sigma)) {
            return false;
        }

        ReadGuard guard(lock);
        size_t position = address - 1;

        const Index deleted_marker = Index{0};
        write_value(position, deleted_marker);

        return true;
    }

    /**
     * @brief Return the number of 16-bit units reserved so far.
     */
    [[nodiscard]] size_t reserved_unit_count() const noexcept
    {
        return top.load();
    }

    /**
     * @brief Return the current number of allocated storage blocks.
     */
    [[nodiscard]] size_t block_count() const
    {
        ReadGuard guard(lock);
        return data.size();
    }

    /**
     * @brief Return the number of 16-bit units in one storage block.
     */
    [[nodiscard]] size_t block_unit_length() const noexcept
    {
        return unit_length;
    }

private:
    [[nodiscard]] static size_t checked_dimension(size_t dimension)
    {
        if constexpr (Dim == Dynamic) {
            if (dimension == 0) {
                throw std::invalid_argument(
                    "Dynamic HVDataBase requires a positive runtime dimension");
            }
            return dimension;
        } else {
            if (dimension != static_cast<size_t>(Dim)) {
                throw std::invalid_argument(
                    "HVDataBase runtime dimension does not match Dim");
            }
            return static_cast<size_t>(Dim);
        }
    }

    void prepare_vertex_point(VertexPoint& point) const
    {
        if constexpr (Dim == Dynamic) {
            point.resize(static_cast<Eigen::Index>(dimension_));
        }
    }

    void require_vertex_point(const VertexPoint& point) const
    {
        if (static_cast<size_t>(point.size()) != dimension_) {
            throw std::invalid_argument(
                "HVDataBase vertex point has wrong dimension");
        }
    }

    /**
     * @brief Atomically reserve an exclusive range of 16-bit storage units.
     *
     * @param count Number of units to reserve.
     * @return Zero-based first unit of the reserved range.
     */
    [[nodiscard]] size_t reserve_units(
        size_t count) noexcept
    {
        return top.fetch_add(count);
    }

    /**
     * @brief Grow the block vector if the reserved range does not fit.
     *
     * The capacity is checked once without locking. If growth may be needed,
     * the write lock is acquired and the condition is checked again. This is
     * the only function that changes the structure of data.
     *
     * @param required_units Number of units that must fit in data.
     */
    void ensure_capacity(
        size_t required_units)
    {
        if (capacity.load() >= required_units) {
            return;
        }

        WriteGuard guard(lock);

        if (capacity.load() >= required_units) {
            return;
        }

        const size_t required_blocks =
            (required_units + unit_length - 1) /
            unit_length;

        const size_t old_block_count =
            data.size();

        data.resize(required_blocks);

        for (
            size_t i = old_block_count;
            i < required_blocks;
            ++i)
        {
            data[i].resize(unit_length);
        }

        capacity.store(
            required_blocks * unit_length);
    }

    /**
     * @brief Copy a contiguous value range into the block stream.
     *
     * The copy continues in the next block whenever the current block boundary
     * is reached. position is measured in 16-bit units and is advanced by the
     * copied range.
     *
     * @tparam Value Scalar or Index.
     * @param position Current zero-based stream position.
     * @param source Pointer to the first source value.
     * @param count Number of values to write.
     */
    template<class Value>
    void write_values(
        size_t& position,
        const Value* source,
        size_t count)
    {
        static_assert(
            sizeof(Value) % sizeof(UInt16) == 0,
            "Stored values must consist of complete 16-bit units");

        const auto* source_bytes =
            reinterpret_cast<const unsigned char*>(source);

        size_t remaining_units =
            count * sizeof(Value) / sizeof(UInt16);

        while (remaining_units > 0) {
            const size_t block =
                position / unit_length;

            const size_t offset =
                position % unit_length;

            const size_t copied_units =
                std::min(
                    remaining_units,
                    unit_length - offset);

            const size_t copied_bytes =
                copied_units * sizeof(UInt16);

            std::memcpy(
                data[block].data() + offset,
                source_bytes,
                copied_bytes);

            source_bytes += copied_bytes;
            position += copied_units;
            remaining_units -= copied_units;
        }
    }

    /**
     * @brief Copy a contiguous value range from the block stream.
     *
     * The copy continues in the next block whenever the current block boundary
     * is reached. position is measured in 16-bit units and is advanced by the
     * copied range.
     *
     * @tparam Value Scalar or Index.
     * @param position Current zero-based stream position.
     * @param destination Pointer to the first destination value.
     * @param count Number of values to read.
     */
    template<class Value>
    void read_values(
        size_t& position,
        Value* destination,
        size_t count) const
    {
        static_assert(
            sizeof(Value) % sizeof(UInt16) == 0,
            "Stored values must consist of complete 16-bit units");

        auto* destination_bytes =
            reinterpret_cast<unsigned char*>(destination);

        size_t remaining_units =
            count * sizeof(Value) / sizeof(UInt16);

        while (remaining_units > 0) {
            const size_t block =
                position / unit_length;

            const size_t offset =
                position % unit_length;

            const size_t copied_units =
                std::min(
                    remaining_units,
                    unit_length - offset);

            const size_t copied_bytes =
                copied_units * sizeof(UInt16);

            std::memcpy(
                destination_bytes,
                data[block].data() + offset,
                copied_bytes);

            destination_bytes += copied_bytes;
            position += copied_units;
            remaining_units -= copied_units;
        }
    }

    /**
     * @brief Write one Scalar or Index value to the block stream.
     */
    template<class Value>
    void write_value(
        size_t& position,
        const Value& value)
    {
        write_values(
            position,
            &value,
            size_t{1});
    }

    /**
     * @brief Read one Scalar or Index value from the block stream.
     */
    template<class Value>
    [[nodiscard]] Value read_value(
        size_t& position) const
    {
        Value value{};

        read_values(
            position,
            &value,
            size_t{1});

        return value;
    }

    mutable Lock lock{};
    std::vector<DataUnit> data{};
    const size_t unit_length;

    QueueHash keys;
    const size_t dimension_;

    detail::LockAwareCounter<Lock, size_t> top{};
    detail::LockAwareCounter<Lock, size_t> capacity{};
};

} // namespace highvoronoi





