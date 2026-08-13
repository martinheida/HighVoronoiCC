#include "highvoronoi/detail/locks.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <thread>
#include <vector>
#include <iostream>

using highvoronoi::detail::BusyFIFOLock;
using highvoronoi::detail::EmptyLock;
using highvoronoi::detail::ReadWriteLock;
using highvoronoi::detail::with_read_lock;
using highvoronoi::detail::with_write_lock;
using highvoronoi::detail::with_write_lock_from_read;

namespace {

void test_empty_lock()
{
    EmptyLock lock;
    lock.lock();
    lock.unlock();
    lock.readlock();
    lock.readunlock();
    lock.writelock();
    lock.writeunlock();
    assert(!lock.islocked());

    int value = 0;
    with_read_lock(lock, [&] { assert(value == 0); });
    with_write_lock(lock, [&] { value = 1; });
    assert(value == 1);
}

void test_busy_fifo_lock_mutual_exclusion()
{
    BusyFIFOLock lock;
    std::size_t counter = 0;

    constexpr std::size_t thread_count = 8;
    constexpr std::size_t increments_per_thread = 5000;

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (std::size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&] {
            for (std::size_t i = 0; i < increments_per_thread; ++i) {
                lock.lock();
                ++counter;
                lock.unlock();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    assert(counter == thread_count * increments_per_thread);
    assert(!lock.islocked());
}
/*
void test_read_write_lock_exclusion()
{
    ReadWriteLock lock;
    std::atomic<int> active_readers{0};
    std::atomic<int> active_writers{0};
    std::atomic<bool> failed{false};
    std::size_t value = 0;

    constexpr std::size_t reader_threads = 4;
    constexpr std::size_t writer_threads = 2;
    constexpr std::size_t rounds = 3000;

    std::vector<std::thread> threads;
    threads.reserve(reader_threads + writer_threads);

    for (std::size_t t = 0; t < reader_threads; ++t) {
        threads.emplace_back([&] {
            for (std::size_t i = 0; i < rounds; ++i) {
                with_read_lock(lock, [&] {
                    active_readers.fetch_add(1, std::memory_order_relaxed);
                    if (active_writers.load(std::memory_order_relaxed) != 0) {
                        failed.store(true, std::memory_order_relaxed);
                    }
                    (void)value;
                    active_readers.fetch_sub(1, std::memory_order_relaxed);
                });
            }
        });
    }

    for (std::size_t t = 0; t < writer_threads; ++t) {
        threads.emplace_back([&] {
            for (std::size_t i = 0; i < rounds; ++i) {
                with_write_lock(lock, [&] {
                    if (active_writers.fetch_add(1, std::memory_order_relaxed) != 0 ||
                        active_readers.load(std::memory_order_relaxed) != 0) {
                        failed.store(true, std::memory_order_relaxed);
                    }
                    ++value;
                    active_writers.fetch_sub(1, std::memory_order_relaxed);
                });
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    assert(!failed.load(std::memory_order_relaxed));
    assert(value == writer_threads * rounds);
    assert(!lock.islocked());
}*/

void test_temporary_write_lock_from_read()
{
    ReadWriteLock lock;
    int value = 10;

    with_read_lock(lock, [&] {
        assert(value == 10);

        with_write_lock_from_read(lock, [&] {
            // Any condition observed before the transition must be checked
            // again here because the upgrade is intentionally non-atomic.
            assert(value == 10);
            value = 11;
        });

        // The original read lock has been reacquired here.
        assert(value == 11);
    });

    assert(value == 11);
    assert(!lock.islocked());
}

} // namespace


// -----------------------------------------------------------------------------
// Global test state
// -----------------------------------------------------------------------------



/**
 * @brief Atomically updates target to max(target, value).
 */
void atomic_max(std::atomic<int>& target, int value)
{
    int current = target.load();

    while (current < value &&
           !target.compare_exchange_weak(current, value)) {
    }
}


void test_read_write_lock_exclusion()
{
    std::atomic<int> global_count{0};

    std::atomic<int> num_parallel_threads{0};
    std::atomic<int> max_num_parallel_threads{0};

    std::atomic<int> num_read_threads{0};
    std::atomic<int> max_num_read_threads{0};

    std::atomic<int> num_write_threads{0};
    std::atomic<int> max_num_write_threads{0};

    std::atomic<int> num_read_tasks{0};
    std::atomic<int> num_write_tasks{0};
    std::atomic<int> num_read_to_write_tasks{0};

    std::atomic<bool> failed{false};
    ReadWriteLock lock;

    // Atomic on purpose: if the lock is broken, the test itself must not
    // introduce undefined behavior through a C++ data race.
    //
    // Writers still perform separate load/store operations, so concurrent
    // writers can lose updates and are therefore detectable.
    std::atomic<std::uint64_t> protected_value{0};

    constexpr int thread_count = 12;

    // Used to make all threads start their actual task at roughly the same time.
    std::atomic<int> ready_threads{0};
    std::atomic<bool> start{false};

    // Reset global test state.
    global_count.store(0);

    num_parallel_threads.store(0);
    max_num_parallel_threads.store(0);

    num_read_threads.store(0);
    max_num_read_threads.store(0);

    num_write_threads.store(0);
    max_num_write_threads.store(0);

    num_read_tasks.store(0);
    num_write_tasks.store(0);
    num_read_to_write_tasks.store(0);

    failed.store(false);

    std::vector<std::thread> threads;
    threads.reserve(thread_count);


    // -------------------------------------------------------------------------
    // Common read operation
    // -------------------------------------------------------------------------

    auto read_task = [&] {
        const int readers = num_read_threads.fetch_add(1) + 1;
        atomic_max(max_num_read_threads, readers);

        // A reader must never overlap with a writer.
        if (num_write_threads.load() != 0) {
            failed.store(true);
        }

        // Perform an actual read of protected data.
        const auto value = protected_value.load(std::memory_order_relaxed);
        (void)value;

        // Increase the chance that multiple readers are simultaneously
        // inside the read-protected section.
        std::this_thread::yield();

        if (num_write_threads.load() != 0) {
            failed.store(true);
        }

        num_read_threads.fetch_sub(1);
    };


    // -------------------------------------------------------------------------
    // Common write operation
    // -------------------------------------------------------------------------

    auto write_task = [&] {
        const int writers = num_write_threads.fetch_add(1) + 1;
        atomic_max(max_num_write_threads, writers);

        // A writer must be completely exclusive.
        if (writers != 1 || num_read_threads.load() != 0) {
            failed.store(true);
        }

        // Deliberately use separate load/store operations instead of
        // fetch_add(). If two writers incorrectly enter simultaneously,
        // an update may be lost.
        const auto value =
            protected_value.load(std::memory_order_relaxed);

        std::this_thread::yield();

        protected_value.store(
            value + 1,
            std::memory_order_relaxed
        );

        // Check again before leaving the protected section.
        if (num_read_threads.load() != 0 ||
            num_write_threads.load() != 1) {
            failed.store(true);
        }

        num_write_threads.fetch_sub(1);
    };


    // -------------------------------------------------------------------------
    // Worker threads
    // -------------------------------------------------------------------------

    for (int t = 0; t < thread_count; ++t) {
        threads.emplace_back([&] {

            // Tell the main thread that this worker exists.
            ready_threads.fetch_add(1);

            // Wait until every worker has been created.
            while (!start.load()) {
                std::this_thread::yield();
            }

            // Measure how many worker functions overlap in lifetime.
            const int parallel =
                num_parallel_threads.fetch_add(1) + 1;

            atomic_max(max_num_parallel_threads, parallel);

            for (int iteration = 0; iteration < 200; ++iteration) {

            // Obtain a unique task number.
            const int task_number =
                global_count.fetch_add(1);


            // -----------------------------------------------------------------
            // WRITE TASK
            // -----------------------------------------------------------------
            //
            // 0, 4, 8, 12, ...
            //
            if (task_number % 4 == 0) {

                num_write_tasks.fetch_add(1);

                with_write_lock(lock, [&] {
                    write_task();
                });
            }


            // -----------------------------------------------------------------
            // READ -> WRITE -> READ TASK
            // -----------------------------------------------------------------
            //
            // Multiples of 5 which were not already caught by % 4.
            //
            else if (task_number % 5 == 0) {

                num_read_to_write_tasks.fetch_add(1);

                with_read_lock(lock, [&] {

                    // We currently hold a real read lock.
                    int readers =
                        num_read_threads.fetch_add(1) + 1;

                    atomic_max(max_num_read_threads, readers);

                    if (num_write_threads.load() != 0) {
                        failed.store(true);
                    }

                    const auto before =
                        protected_value.load(std::memory_order_relaxed);

                    (void)before;

                    std::this_thread::yield();


                    // ---------------------------------------------------------
                    // Temporarily leave the read state.
                    //
                    // with_write_lock_from_read() will:
                    //
                    //   readunlock()
                    //   writelock()
                    //   ...
                    //   writeunlock()
                    //   readlock()
                    //
                    // Therefore our instrumentation must mirror that state.
                    // ---------------------------------------------------------

                    num_read_threads.fetch_sub(1);

                    with_write_lock_from_read(lock, [&] {
                        write_task();
                    });


                    // The read lock has been reacquired here.
                    readers =
                        num_read_threads.fetch_add(1) + 1;

                    atomic_max(max_num_read_threads, readers);

                    if (num_write_threads.load() != 0) {
                        failed.store(true);
                    }

                    const auto after =
                        protected_value.load(std::memory_order_relaxed);

                    (void)after;

                    std::this_thread::yield();

                    if (num_write_threads.load() != 0) {
                        failed.store(true);
                    }

                    num_read_threads.fetch_sub(1);
                });
            }


            // -----------------------------------------------------------------
            // READ TASK
            // -----------------------------------------------------------------

            else {

                num_read_tasks.fetch_add(1);

                with_read_lock(lock, [&] {
                    read_task();
                });
            }
            } // for-loop

            num_parallel_threads.fetch_sub(1);
        });
    }


    // Wait until all workers have actually been created.
    while (ready_threads.load() != thread_count) {
        std::this_thread::yield();
    }

    // Release all workers at once.
    start.store(true);


    for (auto& thread : threads) {
        thread.join();
    }


    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------

    const int reads =
        num_read_tasks.load();

    const int writes =
        num_write_tasks.load();

    const int read_to_writes =
        num_read_to_write_tasks.load();

    const auto expected_value =
        static_cast<std::uint64_t>(writes + read_to_writes);


    constexpr int total_tasks =
        thread_count * 200;

    assert(global_count.load() == total_tasks);
    // All three task types should actually have occurred.
    assert(reads > 0);
    assert(writes > 0);
    assert(read_to_writes > 0);

    // At least two worker threads must have overlapped.
    assert(max_num_parallel_threads.load() > 1);

    // This is important for a ReadWriteLock:
    // there must actually have been concurrent readers.
    assert(max_num_read_threads.load() > 1);

    // Writers must always be exclusive.
    assert(max_num_write_threads.load() == 1);

    // No forbidden reader/writer or writer/writer overlap was observed.
    assert(!failed.load());

    // Every write task and every read->write task increments exactly once.
    assert(protected_value.load() == expected_value);

    // Everything must be clean after all threads have terminated.
    assert(num_parallel_threads.load() == 0);
    assert(num_read_threads.load() == 0);
    assert(num_write_threads.load() == 0);

    assert(!lock.islocked());


    // Optional diagnostic output:
    //
     std::cout
         << "tasks:                  " << thread_count << '\n'
         << "read tasks:             " << reads << '\n'
         << "write tasks:            " << writes << '\n'
         << "read->write tasks:      " << read_to_writes << '\n'
         << "max parallel threads:   "
         << max_num_parallel_threads.load() << '\n'
         << "max parallel readers:   "
         << max_num_read_threads.load() << '\n'
         << "max parallel writers:   "
         << max_num_write_threads.load() << '\n';
}

int main()
{
    test_empty_lock();
    test_busy_fifo_lock_mutual_exclusion();
    test_read_write_lock_exclusion();
    test_temporary_write_lock_from_read();
}
