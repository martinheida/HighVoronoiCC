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

#include <highvoronoi/detail/hash_types.hpp>
#include <highvoronoi/detail/locks.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

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
template<class Lock, class DataBaseParams>
class HVDataBase final {
public:
    using size_t = std::size_t;
    using UInt16 = std::uint16_t;

    using Parameters = DataBaseParams;
    using Scalar = typename Parameters::Scalar;
    using Index = typename Parameters::Index;
    using address_type = std::size_t;

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
        const DataBaseParams& parameters)
        : unit_length(unit_length),
          keys(detail::make_queue_hash<Lock>(parameters))
    {}

    /**
     * @brief Store one normal record.
     *
     * r must provide contiguous Scalar storage. sigma must provide contiguous
     * Index storage. The database copies both ranges directly without
     * conversion.
     *
     * @tparam RVector Contiguous Scalar vector type.
     * @tparam SigmaVector Contiguous Index vector type.
     * @param r Geometric data.
     * @param sigma Non-empty signature used as hash key.
     * @return One-based record address, or 0 if sigma already exists.
     */
    template<class RVector, class SigmaVector>
    [[nodiscard]] size_t push(
        const RVector& r,
        const SigmaVector& sigma)
    {
        require_scalar_vector<RVector>();
        require_index_vector<SigmaVector>();

        if (keys.pushqueue(sigma, true)) {
            return 0;
        }

        const size_t units =
            index_unit_count * (sigma.size() + 1) +
            scalar_unit_count * r.size();

        const size_t begin = reserve_units(units);
        ensure_capacity(begin + units);

        ReadGuard guard(lock);
        size_t position = begin;

        const Index sigma_length =
            static_cast<Index>(sigma.size());

        write_value(position, sigma_length);
        write_values(position, sigma.data(), sigma.size());
        write_values(position, r.data(), r.size());

        return begin + 1;
    }

    /**
     * @brief Read one normal record.
     *
     * r must already have the required number of Scalar elements. sigma is
     * resized to the stored signature length before its values are read.
     *
     * If the record was marked as deleted, the function returns without
     * modifying r or sigma.
     *
     * @tparam RVector Mutable contiguous Scalar vector type.
     * @tparam SigmaVector Mutable contiguous Index vector type with resize().
     * @param address One-based record address.
     * @param r Destination for the geometric data.
     * @param sigma Destination for the signature.
     */
    template<class RVector, class SigmaVector>
    void read(
        size_t address,
        RVector& r,
        SigmaVector& sigma) const
    {
        require_mutable_scalar_vector<RVector>();
        require_mutable_index_vector<SigmaVector>();

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
        read_values(position, r.data(), r.size());
    }

    /**
     * @brief Return whether a signature is present in the QueueHash.
     *
     * @tparam SigmaVector Contiguous Index vector type.
     * @param sigma Non-empty signature to test.
     */
    template<class SigmaVector>
    [[nodiscard]] bool contains(
        const SigmaVector& sigma) const
    {
        require_index_vector<SigmaVector>();
        return keys.contains(sigma);
    }

    /**
     * @brief Register a signature in the persistent identity hash without
     *        storing a payload record.
     * @return true iff the signature was newly registered.
     *
     * HybridDataBase uses this for vertices supplied by a compute engine so
     * stored and computed vertices participate in the same duplicate check.
     */
    template<class SigmaVector>
    [[nodiscard]] bool register_signature(const SigmaVector& sigma)
    {
        require_index_vector<SigmaVector>();
        return !keys.pushqueue(sigma, true);
    }

    /**
     * @brief Remove a signature from the persistent identity hash without
     *        touching any payload record.
     * @return true iff an existing signature was removed.
     */
    template<class SigmaVector>
    [[nodiscard]] bool erase_signature(const SigmaVector& sigma)
    {
        require_index_vector<SigmaVector>();
        return keys.erase(sigma);
    }

    /**
     * @brief Store one facet record.
     *
     * sigma is sorted in place before it is used as a hash key and before it is
     * written. r and u must provide contiguous Scalar storage. sigma must
     * provide mutable contiguous Index storage.
     *
     * @tparam RVector Contiguous Scalar vector type.
     * @tparam SigmaVector Mutable contiguous Index vector type.
     * @tparam UVector Contiguous Scalar vector type.
     * @param r First geometric vector.
     * @param sigma Non-empty facet signature. It is sorted in place.
     * @param u Second geometric vector.
     * @return One-based record address, or 0 if the sorted sigma already exists.
     */
    template<class RVector, class SigmaVector, class UVector>
    [[nodiscard]] size_t push_facet(
        const RVector& r,
        SigmaVector& sigma,
        const UVector& u)
    {
        require_scalar_vector<RVector>();
        require_mutable_index_vector<SigmaVector>();
        require_scalar_vector<UVector>();

        if (sigma.size() > 1) {
            std::sort(
                sigma.data(),
                sigma.data() + sigma.size());
        }

        if (keys.pushqueue(sigma, true)) {
            return 0;
        }

        const size_t units =
            index_unit_count * (sigma.size() + 1) +
            scalar_unit_count * (r.size() + u.size());

        const size_t begin = reserve_units(units);
        ensure_capacity(begin + units);

        ReadGuard guard(lock);
        size_t position = begin;

        const Index sigma_length =
            static_cast<Index>(sigma.size());

        write_value(position, sigma_length);
        write_values(position, sigma.data(), sigma.size());
        write_values(position, r.data(), r.size());
        write_values(position, u.data(), u.size());

        return begin + 1;
    }

    /**
     * @brief Read one facet record.
     *
     * r and u must already have the required number of Scalar elements. sigma
     * is resized to the stored signature length before its values are read.
     *
     * If the record was marked as deleted, the function returns without
     * modifying r, sigma, or u.
     *
     * @tparam RVector Mutable contiguous Scalar vector type.
     * @tparam SigmaVector Mutable contiguous Index vector type with resize().
     * @tparam UVector Mutable contiguous Scalar vector type.
     * @param address One-based facet-record address.
     * @param r Destination for the first geometric vector.
     * @param sigma Destination for the sorted signature.
     * @param u Destination for the second geometric vector.
     */
    template<class RVector, class SigmaVector, class UVector>
    void read_facet(
        size_t address,
        RVector& r,
        SigmaVector& sigma,
        UVector& u) const
    {
        require_mutable_scalar_vector<RVector>();
        require_mutable_index_vector<SigmaVector>();
        require_mutable_scalar_vector<UVector>();

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
        read_values(position, r.data(), r.size());
        read_values(position, u.data(), u.size());
    }

    /**
     * @brief Remove a signature and mark its record as deleted.
     *
     * The signature is removed from the QueueHash. The stored signature length
     * is then replaced by zero. The remaining record storage is not reclaimed
     * or reused.
     *
     * For a facet record, sigma must have the same sorted order that was used
     * by push_facet().
     *
     * @tparam SigmaVector Contiguous Index vector type.
     * @param address One-based record address.
     * @param sigma Signature to remove from the QueueHash.
     * @return true if the signature was present and removed.
     */
    template<class SigmaVector>
    bool erase(
        size_t address,
        const SigmaVector& sigma)
    {
        require_index_vector<SigmaVector>();

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
    /**
     * @brief Extract the element type addressed by Vector::data().
     */
    template<class Vector>
    using DataElement = std::remove_cv_t<
        std::remove_pointer_t<
            decltype(std::declval<Vector&>().data())>>;

    /**
     * @brief Check that Vector exposes contiguous Scalar storage.
     */
    template<class Vector>
    static constexpr void require_scalar_vector()
    {
        static_assert(
            std::is_same_v<DataElement<Vector>, Scalar>,
            "The vector must contain Scalar elements and provide data()");
    }

    /**
     * @brief Check that Vector exposes mutable contiguous Scalar storage.
     */
    template<class Vector>
    static constexpr void require_mutable_scalar_vector()
    {
        require_scalar_vector<Vector>();

        static_assert(
            !std::is_const_v<
                std::remove_pointer_t<
                    decltype(std::declval<Vector&>().data())>>,
            "The destination vector must provide mutable Scalar storage");
    }

    /**
     * @brief Check that Vector exposes contiguous Index storage.
     */
    template<class Vector>
    static constexpr void require_index_vector()
    {
        static_assert(
            std::is_same_v<DataElement<Vector>, Index>,
            "The vector must contain Index elements and provide data()");
    }

    /**
     * @brief Check that Vector exposes mutable contiguous Index storage.
     */
    template<class Vector>
    static constexpr void require_mutable_index_vector()
    {
        require_index_vector<Vector>();

        static_assert(
            !std::is_const_v<
                std::remove_pointer_t<
                    decltype(std::declval<Vector&>().data())>>,
            "The destination vector must provide mutable Index storage");
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

    std::atomic<size_t> top{0};
    std::atomic<size_t> capacity{0};
};

} // namespace highvoronoi

