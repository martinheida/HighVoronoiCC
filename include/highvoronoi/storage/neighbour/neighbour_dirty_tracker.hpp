#pragma once

/**
 * @file neighbour_dirty_tracker.hpp
 * @brief Per-consumer accumulation and mesh-level tracking state.
 *
 * A consumer such as a finite-volume integral owns one NeighbourDirtyTracker.
 * Mesh geometry first marks the mesh-owned neighbour dirty vector. At explicit
 * lifecycle boundaries the mesh ORs those bits word-wise into every registered
 * tracker. The mesh dirty bits remain unchanged, so consumers can update and
 * clear their own trackers independently.
 */

#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/core/detail/locks.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace highvoronoi::detail {

class NeighbourDirtyTracker final {
public:
    explicit NeighbourDirtyTracker(std::size_t cell_count = 0)
        : dirty_(cell_count) {}

    NeighbourDirtyTracker(const NeighbourDirtyTracker&) = delete;
    NeighbourDirtyTracker& operator=(const NeighbourDirtyTracker&) = delete;
    NeighbourDirtyTracker(NeighbourDirtyTracker&&) = delete;
    NeighbourDirtyTracker& operator=(NeighbourDirtyTracker&&) = delete;

    [[nodiscard]] std::size_t size() const noexcept {
        return dirty_.size();
    }

    [[nodiscard]] bool dirty(std::size_t cell) const noexcept {
        return dirty_.test(cell);
    }

    void set_dirty(std::size_t cell, bool value = true) noexcept {
        if (value) {
            dirty_.set(cell);
        } else {
            dirty_.reset(cell);
        }
    }

    void clear() noexcept {
        dirty_.clear();
    }

    void resize(std::size_t cell_count) {
        dirty_.resize(cell_count, false);
    }

    template<class SourceBitVector>
    void merge_from(const SourceBitVector& source) {
        dirty_.or_assign(source);
    }

private:
    AtomicBitVector dirty_;
};

/**
 * Mesh-level registry for consumer dirty trackers and the one global neighbour
 * version. It deliberately contains no neighbour database and no per-cell
 * neighbour addresses; those belong to NeighbourStorage.
 */
template<class Lock>
class NeighbourTrackingState final {
public:
    using Tracker = NeighbourDirtyTracker;
    using TrackerHandle = std::shared_ptr<Tracker>;

    template<class SourceBitVector>
    [[nodiscard]] TrackerHandle request_tracker(
        const SourceBitVector& current_dirty,
        bool include_current_dirty = false) {
        auto tracker = std::make_shared<Tracker>(current_dirty.size());
        if (include_current_dirty) {
            tracker->merge_from(current_dirty);
        }

        WriteLockGuard<Lock> guard(lock_);
        trackers_.push_back(tracker);
        return tracker;
    }

    template<class SourceBitVector>
    void propagate(const SourceBitVector& current_dirty) {
        WriteLockGuard<Lock> guard(lock_);

        std::size_t write = 0;
        for (std::size_t read = 0; read < trackers_.size(); ++read) {
            if (auto tracker = trackers_[read].lock()) {
                if (tracker->size() != current_dirty.size()) {
                    tracker->resize(current_dirty.size());
                }
                tracker->merge_from(current_dirty);
                trackers_[write++] = trackers_[read];
            }
        }
        trackers_.resize(write);
    }

    [[nodiscard]] std::size_t registered_tracker_count() const {
        ReadLockGuard<Lock> guard(lock_);
        std::size_t count = 0;
        for (const auto& tracker : trackers_) {
            if (!tracker.expired()) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] std::size_t version() const noexcept {
        return version_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::size_t advance_version() noexcept {
        return version_.fetch_add(std::size_t{1}, std::memory_order_acq_rel) +
               std::size_t{1};
    }

private:
    mutable Lock lock_{};
    std::vector<std::weak_ptr<Tracker>> trackers_;
    std::atomic<std::size_t> version_{0};
};

} // namespace highvoronoi::detail
