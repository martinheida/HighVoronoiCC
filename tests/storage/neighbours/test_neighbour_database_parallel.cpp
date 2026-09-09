#include <highvoronoi/storage/neighbour/neighbour_database.hpp>
#include <highvoronoi/storage/read_write_list.hpp>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
using Index = std::uint32_t;
using Address = std::size_t;
std::size_t checks = 0;
std::size_t failures = 0;

void check(bool value, const char* text) {
    ++checks;
    if (value) {
        std::cout << "[OK]   " << text << '\n';
    } else {
        ++failures;
        std::cerr << "[FAIL] " << text << '\n';
    }
}

void test_large_record() {
    highvoronoi::detail::NeighbourDatabase<
        highvoronoi::detail::EmptyLock,
        Index> db(17);

    std::vector<Index> input;
    input.reserve(70000);
    for (std::size_t i = 0; i < 70000; ++i) {
        input.push_back(static_cast<Index>((i / 3) % 4096));
    }

    const Address address = db.push(input);
    std::vector<Index> output;
    check(address != 0, "large record obtains non-zero address");
    check(db.read(address, output), "large record is readable");
    check(output == input, "70,000 entries and duplicates survive exactly");
}

void test_parallel_database_writes() {
    using DB = highvoronoi::detail::NeighbourDatabase<
        highvoronoi::detail::ReadWriteLock,
        Index>;

    constexpr std::size_t Threads = 8;
    constexpr std::size_t RecordsPerThread = 128;
    DB db(23); // force frequent block growth under contention

    std::vector<std::vector<Address>> addresses(
        Threads,
        std::vector<Address>(RecordsPerThread));
    std::vector<std::thread> workers;
    workers.reserve(Threads);

    for (std::size_t t = 0; t < Threads; ++t) {
        workers.emplace_back([&, t] {
            std::vector<Index> list;
            list.reserve(64);
            for (std::size_t r = 0; r < RecordsPerThread; ++r) {
                list.clear();
                for (std::size_t k = 0; k < 64; ++k) {
                    list.push_back(static_cast<Index>(
                        t * 100000 + r * 100 + k / 2));
                }
                addresses[t][r] = db.push(list);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    bool ok = true;
    std::vector<Index> output;
    for (std::size_t t = 0; t < Threads && ok; ++t) {
        for (std::size_t r = 0; r < RecordsPerThread && ok; ++r) {
            ok = ok && db.read(addresses[t][r], output);
            ok = ok && output.size() == 64;
            for (std::size_t k = 0; k < output.size() && ok; ++k) {
                ok = output[k] == static_cast<Index>(
                    t * 100000 + r * 100 + k / 2);
            }
        }
    }
    check(ok, "parallel DB writers produce intact disjoint records");
}

void test_read_write_address_list_reuse() {
    using List = highvoronoi::ReadWriteAddressList<
        std::vector<Address>,
        highvoronoi::detail::ReadWriteLock>;

    constexpr std::size_t Threads = 8;
    constexpr std::size_t Pushes = 500;
    List list;
    std::vector<std::thread> workers;
    for (std::size_t t = 0; t < Threads; ++t) {
        workers.emplace_back([&, t] {
            for (std::size_t i = 0; i < Pushes; ++i) {
                list.push_back(Address{1} + t * Pushes + i);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    Address last = 0;
    check(list.size() == Threads * Pushes,
          "ReadWriteAddressList accepts parallel neighbour-version appends");
    check(list.try_back(last) && last != 0,
          "try_back reads one published neighbour address under one read lock");
}

void test_dirty_type_selection() {
    using SerialDirty = highvoronoi::detail::NeighbourDirtyVector<
        highvoronoi::detail::EmptyLock>;
    using ParallelDirty = highvoronoi::detail::NeighbourDirtyVector<
        highvoronoi::detail::ReadWriteLock>;

    static_assert(std::is_same_v<SerialDirty, highvoronoi::detail::BitVector>);
    static_assert(std::is_same_v<ParallelDirty, highvoronoi::detail::AtomicBitVector>);

    constexpr std::size_t Bits = 4096;
    ParallelDirty dirty(Bits);
    std::vector<std::thread> workers;
    for (std::size_t t = 0; t < 8; ++t) {
        workers.emplace_back([&, t] {
            for (std::size_t i = t; i < Bits; i += 8) {
                dirty.set(i);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    bool all = true;
    for (std::size_t i = 0; i < Bits; ++i) {
        all = all && dirty.test(i);
    }
    check(all, "parallel lock policy selects AtomicBitVector dirty state");
}
}

int main() {
    test_large_record();
    test_parallel_database_writes();
    test_read_write_address_list_reuse();
    test_dirty_type_selection();
    std::cout << "checks: " << checks << "\nfailures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
