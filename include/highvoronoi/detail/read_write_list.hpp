#pragma once

#include <highvoronoi/detail/locks.hpp>

#include <algorithm>
#include <utility>

namespace highvoronoi {

/**
 * @brief Locked wrapper providing the AddressListT interface.
 *
 * @tparam DataT Vector-like data container.
 * @tparam LockT Read/write lock type.
 */
template <class DataT, class LockT>
class ReadWriteAddressList final {
public:
    using value_type = typename DataT::value_type;
    using size_type = typename DataT::size_type;

    ReadWriteAddressList() = default;

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
