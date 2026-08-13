#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <thread>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace highvoronoi::detail {
/**
 * @brief Perform a bounded busy wait and occasionally yield the CPU.
 *
 * The compiler fence prevents the spin loop from being optimized away. This
 * deliberately avoids platform-specific pause instructions so the lock remains
 * portable C++17. Yielding every 100 wait rounds mirrors the intent of the
 * Julia implementation: spin briefly first, then give another runnable thread
 * a chance to make progress under sustained contention.
 */
inline void active_wait(std::uint32_t& wait_round) noexcept
{
    const std::uint32_t spins = std::min<std::uint32_t>(wait_round, 100U);
    for (std::uint32_t i = 0; i < spins; ++i) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }

    if ((wait_round % 100U) == 0U) {
        std::this_thread::yield();
    }

    ++wait_round;
}

/**
 * @brief No-op lock with the same internal interface as ReadWriteLock.
 *
 * EmptyLock is useful for compile-time configurations in which synchronization
 * is intentionally disabled. All lock and unlock operations are no-ops and
 * islocked() always returns false.
 *
 * The class does not provide any synchronization and must only be used when
 * concurrent access is known to be safe without locking.
 */
class EmptyLock final {
public:
    constexpr EmptyLock() noexcept = default;

    constexpr void lock() noexcept {}
    constexpr void unlock() noexcept {}

    constexpr void readlock() noexcept {}
    constexpr void readunlock() noexcept {}

    constexpr void writelock() noexcept {}
    constexpr void writeunlock() noexcept {}

    [[nodiscard]] constexpr bool islocked() const noexcept { return false; }
};

/**
 * @brief Non-reentrant, ticket-based busy-wait FIFO lock.
 *
 * BusyFIFOLock is intended for extremely short critical sections for which
 * sleeping and rescheduling a thread would likely cost more than spinning.
 * Each caller receives a monotonically increasing ticket and may enter only
 * when that ticket reaches the queue head.
 *
 * The lock is FIFO with respect to issued tickets. It does not track ownership
 * and therefore cannot detect an invalid unlock(). Recursive locking from the
 * same thread deadlocks.
 *
 * @warning Do not perform I/O, blocking operations, or long-running work while
 * holding this lock. Under contention a spin lock consumes CPU time.
 */
class BusyFIFOLock final {
public:
    BusyFIFOLock() noexcept = default;

    BusyFIFOLock(const BusyFIFOLock&) = delete;
    BusyFIFOLock& operator=(const BusyFIFOLock&) = delete;
    BusyFIFOLock(BusyFIFOLock&&) = delete;
    BusyFIFOLock& operator=(BusyFIFOLock&&) = delete;

    /** Acquire the lock, spinning until this thread's ticket reaches the head. */
    void lock() noexcept
        {
            const std::uint64_t my_ticket = tail_.fetch_add(1, std::memory_order_relaxed);
            std::uint32_t wait_round = 1;

            while (head_.load(std::memory_order_acquire) != my_ticket) {
                active_wait(wait_round);
            }
        };

    /** Release the lock and admit the next ticket. */
    void unlock() noexcept
        {
            head_.fetch_add(1, std::memory_order_release);
        };

    /**
     * @brief Return whether at least one issued ticket is still active/waiting.
     *
     * This is an observational helper only. The value may become stale
     * immediately under concurrent access and must not be used for correctness.
     */
    [[nodiscard]] bool islocked() const noexcept
        {
            return head_.load(std::memory_order_relaxed) !=
                   tail_.load(std::memory_order_relaxed);
        };

private:
    std::atomic<std::uint64_t> head_{0};
    std::atomic<std::uint64_t> tail_{0};
};

/**
 * @brief Non-reentrant FIFO read/write spin lock based on ticket counters.
 *
 * Multiple readers may hold the lock simultaneously, while a writer is
 * exclusive. Admission is FIFO with respect to ticket order:
 *
 * - A reader takes a ticket, waits for all earlier tickets, increments the
 *   active-reader count, and advances the queue head immediately.
 * - A writer takes a ticket, waits for all earlier tickets and for the active
 *   reader count to become zero, and advances the queue head only on unlock.
 *
 * Consequently, consecutive readers ahead of a writer may batch together,
 * while readers behind that writer cannot overtake it.
 *
 * lock()/unlock() are aliases for writelock()/writeunlock().
 *
 * @warning This lock is not reentrant. Unlocking without the matching lock
 * operation corrupts its state. It is a spin lock and is intended only for
 * short critical sections.
 */
class ReadWriteLock final {
public:
    ReadWriteLock() noexcept = default;

    ReadWriteLock(const ReadWriteLock&) = delete;
    ReadWriteLock& operator=(const ReadWriteLock&) = delete;
    ReadWriteLock(ReadWriteLock&&) = delete;
    ReadWriteLock& operator=(ReadWriteLock&&) = delete;

    /** Acquire an exclusive write lock. */
    void lock() noexcept
        {
            writelock();
        };

    /** Release an exclusive write lock. */
    void unlock() noexcept
        {
            writeunlock();
        };

    /** Acquire a shared read lock in FIFO ticket order. */
    void readlock() noexcept
        {
            const std::uint64_t my_ticket = tail_.fetch_add(1, std::memory_order_relaxed);
            std::uint32_t wait_round = 1;

            while (head_.load(std::memory_order_acquire) != my_ticket) {
                active_wait(wait_round);
            }

            // The reader is now admitted. Increment the reader count before advancing
            // the queue head so a writer behind us cannot miss this active reader.
            reads_count_.fetch_add(1, std::memory_order_relaxed);
            head_.fetch_add(1, std::memory_order_release);
        };

    /** Release one shared read lock. */
    void readunlock() noexcept
        {
            reads_count_.fetch_sub(1, std::memory_order_release);
        };

    /** Acquire an exclusive write lock in FIFO ticket order. */
    void writelock() noexcept
        {
            const std::uint64_t my_ticket = tail_.fetch_add(1, std::memory_order_relaxed);
            std::uint32_t wait_round = 1;

            while (head_.load(std::memory_order_acquire) != my_ticket ||
                   reads_count_.load(std::memory_order_acquire) != 0) {
                active_wait(wait_round);
            }

            // Unlike a reader, a writer intentionally keeps head_ on its own ticket.
            // This prevents every later ticket from entering until writeunlock().
        };

    /** Release the exclusive write lock and advance the FIFO queue. */
    void writeunlock() noexcept
        {
            head_.fetch_add(1, std::memory_order_release);
        };

    /**
     * @brief Return whether the lock has active readers, a writer, or waiters.
     *
     * This is an observational helper only. The value may become stale
     * immediately under concurrent access and must not be used for correctness.
     */
    [[nodiscard]] bool islocked() const noexcept
        {
            return reads_count_.load(std::memory_order_relaxed) != 0 ||
                   head_.load(std::memory_order_relaxed) !=
                       tail_.load(std::memory_order_relaxed);
        };

private:
    std::atomic<std::uint64_t> head_{0};
    std::atomic<std::uint64_t> tail_{0};
    std::atomic<std::uint64_t> reads_count_{0};
};

/**
 * @brief RAII guard for a shared/read lock.
 *
 * The constructor calls readlock() and the destructor calls readunlock().
 */
template <class Lock>
class ReadLockGuard final {
public:
    explicit ReadLockGuard(Lock& lock) noexcept(noexcept(lock.readlock()))
        : lock_(lock)
    {
        lock_.readlock();
    }

    ~ReadLockGuard() noexcept(noexcept(std::declval<Lock&>().readunlock()))
    {
        lock_.readunlock();
    }

    ReadLockGuard(const ReadLockGuard&) = delete;
    ReadLockGuard& operator=(const ReadLockGuard&) = delete;
    ReadLockGuard(ReadLockGuard&&) = delete;
    ReadLockGuard& operator=(ReadLockGuard&&) = delete;

private:
    Lock& lock_;
};

/**
 * @brief RAII guard for an exclusive/write lock.
 *
 * The constructor calls writelock() and the destructor calls writeunlock().
 */
template <class Lock>
class WriteLockGuard final {
public:
    explicit WriteLockGuard(Lock& lock) noexcept(noexcept(lock.writelock()))
        : lock_(lock)
    {
        lock_.writelock();
    }

    ~WriteLockGuard() noexcept(noexcept(std::declval<Lock&>().writeunlock()))
    {
        lock_.writeunlock();
    }

    WriteLockGuard(const WriteLockGuard&) = delete;
    WriteLockGuard& operator=(const WriteLockGuard&) = delete;
    WriteLockGuard(WriteLockGuard&&) = delete;
    WriteLockGuard& operator=(WriteLockGuard&&) = delete;

private:
    Lock& lock_;
};

/**
 * @brief Temporarily replace an already-held read lock by a write lock.
 *
 * Construction releases one read lock and then acquires a write lock.
 * Destruction releases the write lock and reacquires the read lock.
 * This is intended to be nested inside a ReadLockGuard or with_read_lock().
 *
 * @warning This is not an atomic lock upgrade. Other readers or writers may
 * enter after the read lock is released and before the write lock is acquired.
 * Any condition observed before the upgrade must therefore be revalidated
 * while holding the write lock. The same applies to assumptions spanning the
 * write-to-read transition in the destructor.
 */
template <class Lock>
class ReadToWriteLockGuard final {
public:
    explicit ReadToWriteLockGuard(Lock& lock) noexcept(
        noexcept(lock.readunlock()) && noexcept(lock.writelock()))
        : lock_(lock)
    {
        lock_.readunlock();
        lock_.writelock();
    }

    ~ReadToWriteLockGuard() noexcept(
        noexcept(std::declval<Lock&>().writeunlock()) &&
        noexcept(std::declval<Lock&>().readlock()))
    {
        lock_.writeunlock();
        lock_.readlock();
    }

    ReadToWriteLockGuard(const ReadToWriteLockGuard&) = delete;
    ReadToWriteLockGuard& operator=(const ReadToWriteLockGuard&) = delete;
    ReadToWriteLockGuard(ReadToWriteLockGuard&&) = delete;
    ReadToWriteLockGuard& operator=(ReadToWriteLockGuard&&) = delete;

private:
    Lock& lock_;
};

/**
 * @brief Execute a callable while holding a shared/read lock.
 *
 * This is the C++ equivalent of HighVoronoi's Julia @readlocked wrapper.
 * The lock is released automatically if the callable returns or throws.
 */
template <class Lock, class Function>
decltype(auto) with_read_lock(Lock& lock, Function&& function)
{
    ReadLockGuard<Lock> guard(lock);
    return std::forward<Function>(function)();
}

/**
 * @brief Execute a callable while holding an exclusive/write lock.
 *
 * This is the C++ equivalent of HighVoronoi's Julia @writelocked wrapper.
 * The lock is released automatically if the callable returns or throws.
 */
template <class Lock, class Function>
decltype(auto) with_write_lock(Lock& lock, Function&& function)
{
    WriteLockGuard<Lock> guard(lock);
    return std::forward<Function>(function)();
}

/**
 * @brief Temporarily execute a callable under a write lock while a read lock
 * is already held by the current thread.
 *
 * The existing read lock is released, a write lock is acquired, the callable
 * is executed, then the write lock is released and the read lock is reacquired.
 * This is exception-safe through RAII.
 *
 * @warning The transition is not atomic. Revalidate all data-dependent
 * assumptions after the write lock has been acquired.
 */
template <class Lock, class Function>
decltype(auto) with_write_lock_from_read(Lock& lock, Function&& function)
{
    ReadToWriteLockGuard<Lock> guard(lock);
    return std::forward<Function>(function)();
}



/**
 * @brief Execution policy for strictly single-threaded operation.
 *
 * Selecting this policy replaces synchronization primitives with no-op locks,
 * allowing the compiler to remove the synchronization code entirely.
 */
class SingleThread
{
public:
    using RWLock = EmptyLock;

    /**
     * @brief Returns the number of worker threads.
     */
    [[nodiscard]]
    static constexpr std::size_t thread_count() noexcept
    {
        return 1;
    }

    /**
     * @brief Creates the read/write lock associated with this policy.
     */
    [[nodiscard]]
    static constexpr RWLock RWLOCK() noexcept
    {
        return {};
    }

    /**
     * @brief Indicates whether this policy supports multiple threads.
     */
    static constexpr bool is_multithreaded = false;
};


/**
 * @brief Execution policy for multi-threaded operation.
 *
 * The thread count is stored at runtime, while the lock type is selected
 * statically from the policy type.
 */
class MultiThread
{
public:
    using RWLock = ReadWriteLock;

    /**
     * @brief Constructs a multi-thread execution policy.
     *
     * @param count Number of worker threads.
     *
     * @throws std::invalid_argument if count is zero.
     */
    explicit MultiThread(std::size_t count)
        : count_(count)
    {
        if (count_ == 0) {
            throw std::invalid_argument(
                "MultiThread requires at least one thread"
            );
        }
    }

    /**
     * @brief Returns the configured number of worker threads.
     */
    [[nodiscard]]
    std::size_t thread_count() const noexcept
    {
        return count_;
    }

    /**
     * @brief Creates the read/write lock associated with this policy.
     */
    [[nodiscard]]
    static RWLock RWLOCK() noexcept
    {
        return {};
    }

    /**
     * @brief Indicates whether this policy supports multiple threads.
     */
    static constexpr bool is_multithreaded = true;

private:
    std::size_t count_;
};

} // namespace highvoronoi::detail
