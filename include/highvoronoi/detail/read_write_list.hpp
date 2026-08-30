#pragma once

#include <highvoronoi/detail/locks.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>

namespace highvoronoi {

/**
 * @brief Locked wrapper providing the AddressListT interface.
 *
 * @tparam DataT Vector-like data container.
 * @tparam LockT Read/write lock type.
 *
 * @par Structural moves
 * The contained lock is deliberately not moved. Moving an address list moves
 * only its data and leaves the destination with its own default-constructed
 * lock. This permits containers such as `std::vector<ReadWriteAddressList>` to
 * reallocate during structural mesh changes.
 *
 * Such structural moves require external synchronization: neither the source
 * nor the destination may be accessed concurrently while a move is performed.
 * This matches the mesh mutation contract, where node insertion/removal occurs
 * outside parallel ComputeVoronoi phases.
 */
template <class DataT, class LockT>
class ReadWriteAddressList final {
public:
    using value_type = typename DataT::value_type;
    using size_type = typename DataT::size_type;

    ReadWriteAddressList() = default;

    ReadWriteAddressList(const ReadWriteAddressList&) = delete;
    ReadWriteAddressList& operator=(const ReadWriteAddressList&) = delete;

    /**
     * @brief Move only the protected payload and create a fresh destination lock.
     *
     * No lock state is transferred. External synchronization must guarantee that
     * `other` is not concurrently accessed while this structural move occurs.
     */
    ReadWriteAddressList(ReadWriteAddressList&& other)
        noexcept(std::is_nothrow_move_constructible_v<DataT>)
        : data_(std::move(other.data_)),
          lock_() {}

    /**
     * @brief Move only the protected payload while retaining this object's lock.
     *
     * No lock state is transferred. External synchronization must guarantee that
     * neither object is concurrently accessed while this structural move occurs.
     */
    ReadWriteAddressList& operator=(ReadWriteAddressList&& other)
        noexcept(std::is_nothrow_move_assignable_v<DataT>) {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    [[nodiscard]] size_type size() const {
        detail::ReadLockGuard<LockT> guard(lock_);
        return data_.size();
    }

    [[nodiscard]] value_type operator[](size_type index) const {
        detail::ReadLockGuard<LockT> guard(lock_);
        return data_[index];
    }

    void push_back(value_type value) {
        detail::WriteLockGuard<LockT> guard(lock_);
        data_.push_back(value);
    }

    /**
     * @brief Remove every entry selected by a predicate.
     *
     * The write lock is held for the complete compaction operation. No iterator
     * escapes the protected critical section.
     */
    template <class Predicate>
    void erase_if(Predicate&& predicate) {
        detail::WriteLockGuard<LockT> guard(lock_);
        data_.erase(
            std::remove_if(
                data_.begin(),
                data_.end(),
                std::forward<Predicate>(predicate)),
            data_.end());
    }

private:
    DataT data_;
    mutable LockT lock_;
};

} // namespace highvoronoi
