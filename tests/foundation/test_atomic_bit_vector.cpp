#include <highvoronoi/core/detail/atomic_bit_vector.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using highvoronoi::detail::AtomicBitVector;
using highvoronoi::detail::BitVector;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;

    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

void print_test_header(std::string_view title) {
    std::cout << "\n[TEST] " << title << '\n';
}


void test_non_atomic_basic_bit_operations() {
    print_test_header("BitVector: basic set / test / reset / flip semantics");
    std::cout
        << "  The non-atomic BitVector uses the same packed 64-bit indexing as\n"
        << "  AtomicBitVector. We again cross the 63/64 and 127/128 word boundaries.\n";

    BitVector bits(130);

    check(bits.size() == 130,
          "BitVector size() reports the requested number of bits");
    check(!bits.empty(), "a 130-bit BitVector is not empty");

    bool initially_clear = true;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        initially_clear = initially_clear && !bits.test(i);
    }
    check(initially_clear, "all BitVector bits are initially false");

    const std::vector<std::size_t> selected{0, 1, 63, 64, 65, 127, 128, 129};
    for (const std::size_t index : selected) {
        bits.set(index);
    }

    bool selected_are_set = true;
    for (const std::size_t index : selected) {
        selected_are_set = selected_are_set && bits.test(index);
    }
    check(selected_are_set,
          "BitVector set() works across 64-bit word boundaries");

    check(!bits.test(62),
          "setting BitVector bit 63 does not accidentally set bit 62");
    check(!bits.test(66),
          "setting BitVector bit 65 does not accidentally set bit 66");

    bits.reset(64);
    check(!bits.test(64), "BitVector reset() clears exactly the requested bit");
    check(bits.test(63) && bits.test(65),
          "BitVector reset() preserves neighbouring bits");

    bits.flip(65);
    check(!bits.test(65), "BitVector flip() changes true to false");
    bits.flip(65);
    check(bits.test(65), "BitVector flip() changes false back to true");

    bits.clear();
    bool all_clear = true;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        all_clear = all_clear && !bits.test(i);
    }
    check(all_clear, "BitVector clear() resets every stored word to zero");

    bits.set_all();
    bool all_set = true;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        all_set = all_set && bits.test(i);
    }
    check(all_set,
          "BitVector set_all() sets every logical bit, including the tail word");
    check(!bits.test(130),
          "BitVector set_all() does not expose unused tail bits as logical bits");
}

void test_non_atomic_bounds_and_empty_vector() {
    print_test_header("BitVector: empty vector and out-of-range behaviour");
    std::cout
        << "  BitVector deliberately has the same defensive single-bit bounds\n"
        << "  semantics as AtomicBitVector: writes are ignored and reads return false.\n";

    BitVector empty;
    check(empty.empty(), "a default-constructed BitVector reports empty()");
    check(empty.size() == 0, "a default-constructed BitVector reports size 0");

    empty.set(0);
    empty.reset(0);
    empty.flip(0);
    check(!empty.test(0), "operations on an empty BitVector remain harmless");

    BitVector bits(10);
    bits.set(3);
    bits.set(1000);
    bits.reset(1000);
    bits.flip(1000);

    check(bits.test(3),
          "out-of-range BitVector writes do not disturb valid bits");
    check(!bits.test(1000),
          "out-of-range BitVector test() returns false");
}

void test_non_atomic_structural_operations() {
    print_test_header("BitVector: resize / push_back / reserve semantics");
    std::cout
        << "  Only the non-atomic BitVector is structurally mutable. This test checks\n"
        << "  that resizing preserves the old prefix, initializes new bits correctly,\n"
        << "  and that push_back works across a packed-word boundary.\n";

    BitVector bits(65);
    bits.set(0);
    bits.set(63);
    bits.set(64);

    bits.reserve(300);
    check(bits.size() == 65,
          "reserve() changes capacity only, not the logical BitVector size");
    check(bits.test(0) && bits.test(63) && bits.test(64),
          "reserve() preserves existing BitVector contents");

    bits.resize(130, false);
    check(bits.size() == 130, "resize() grows the logical BitVector size");
    check(bits.test(0) && bits.test(63) && bits.test(64),
          "growing resize() preserves the old BitVector prefix");

    bool appended_bits_are_clear = true;
    for (std::size_t i = 65; i < 130; ++i) {
        appended_bits_are_clear = appended_bits_are_clear && !bits.test(i);
    }
    check(appended_bits_are_clear,
          "resize(..., false) initializes every newly appended bit to false");

    bits.set(129);
    bits.resize(64);
    check(bits.size() == 64, "shrinking resize() reduces the logical size");
    check(bits.test(0) && bits.test(63),
          "shrinking resize() preserves the retained prefix");
    check(!bits.test(64) && !bits.test(129),
          "bits removed by shrinking are no longer readable");

    bits.resize(130, true);
    check(bits.test(0) && bits.test(63),
          "resize(..., true) still preserves the old prefix");

    bool newly_appended_bits_are_set = true;
    for (std::size_t i = 64; i < 130; ++i) {
        newly_appended_bits_are_set =
            newly_appended_bits_are_set && bits.test(i);
    }
    check(newly_appended_bits_are_set,
          "resize(..., true) initializes every newly appended bit to true");

    BitVector appended(63, true);
    appended.push_back(false); // index 63, final bit of the first word
    appended.push_back(true);  // index 64, first bit of the second word

    check(appended.size() == 65,
          "two push_back() calls increase BitVector size from 63 to 65");
    check(appended.test(62),
          "push_back() preserves the existing prefix at the word boundary");
    check(!appended.test(63),
          "push_back(false) stores false at bit 63");
    check(appended.test(64),
          "push_back(true) stores true at bit 64 in the next word");
}

void test_basic_bit_operations() {
    print_test_header("Basic set / test / reset / flip semantics");
    std::cout
        << "  We use 130 bits so that indices 63/64 and 127/128 cross\n"
        << "  two different 64-bit word boundaries.\n";

    AtomicBitVector bits(130);

    check(bits.size() == 130, "size() reports the requested number of bits");
    check(!bits.empty(), "a 130-bit vector is not empty");

    // The constructor must initialize every atomic word to zero. In C++17 this
    // is not automatic for default-constructed std::atomic<T>, which is why the
    // implementation explicitly uses std::atomic_init.
    bool initially_clear = true;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        initially_clear = initially_clear && !bits.test(i);
    }
    check(initially_clear, "all bits are initially false");

    // Exercise positions immediately before and after the 64-bit word borders.
    const std::vector<std::size_t> selected{0, 1, 63, 64, 65, 127, 128, 129};
    for (const std::size_t index : selected) {
        bits.set(index);
    }

    bool selected_are_set = true;
    for (const std::size_t index : selected) {
        selected_are_set = selected_are_set && bits.test(index);
    }
    check(selected_are_set, "set() works across 64-bit word boundaries");

    check(!bits.test(62), "setting bit 63 does not accidentally set bit 62");
    check(!bits.test(66), "setting bit 65 does not accidentally set bit 66");

    bits.reset(64);
    check(!bits.test(64), "reset() clears exactly the requested bit");
    check(bits.test(63) && bits.test(65),
          "reset() preserves neighbouring bits in the same/adjacent words");

    bits.flip(65);
    check(!bits.test(65), "flip() changes true to false");
    bits.flip(65);
    check(bits.test(65), "flip() changes false back to true");

    bits.clear();
    bool all_clear = true;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        all_clear = all_clear && !bits.test(i);
    }
    check(all_clear, "clear() resets every stored word to zero");
}

void test_bounds_and_empty_vector() {
    print_test_header("Empty vector and out-of-range behaviour");
    std::cout
        << "  The class deliberately treats out-of-range writes as no-ops and\n"
        << "  out-of-range reads as false. This keeps HighVoronoiAffectedNodes::mark()\n"
        << "  cheap and noexcept even when handed a defensive invalid index.\n";

    AtomicBitVector empty(0);
    check(empty.empty(), "a zero-sized AtomicBitVector reports empty()");
    check(empty.size() == 0, "a zero-sized AtomicBitVector reports size 0");

    // None of these operations may dereference storage because the vector owns
    // no atomic words at all.
    empty.set(0);
    empty.reset(0);
    empty.flip(0);
    check(!empty.test(0), "operations on an empty vector remain harmless");

    AtomicBitVector bits(10);
    bits.set(3);
    bits.set(1000);   // ignored
    bits.reset(1000); // ignored
    bits.flip(1000);  // ignored

    check(bits.test(3), "out-of-range writes do not disturb valid bits");
    check(!bits.test(1000), "out-of-range test() returns false");
}

void test_concurrent_sets_in_shared_words() {
    print_test_header("Concurrent set() operations on shared atomic words");
    std::cout
        << "  Eight threads start together. Each thread sets a different subset\n"
        << "  of the same 256-bit vector, so several threads perform fetch_or()\n"
        << "  on the same underlying 64-bit atomic words. The final state must be\n"
        << "  the union of all thread updates: every bit is true.\n";

    constexpr std::size_t bit_count = 256;
    constexpr std::size_t thread_count = 8;

    AtomicBitVector bits(bit_count);
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([thread, &bits, &ready, &start]() {
            ready.fetch_add(1, std::memory_order_relaxed);

            // The acquire/release pair here synchronizes only the test start.
            // AtomicBitVector itself intentionally uses relaxed atomics because
            // its flags are data, not a publication/synchronization mechanism.
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (std::size_t index = thread;
                 index < bit_count;
                 index += thread_count) {
                bits.set(index);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != thread_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    bool every_bit_set = true;
    for (std::size_t i = 0; i < bit_count; ++i) {
        every_bit_set = every_bit_set && bits.test(i);
    }

    check(every_bit_set,
          "concurrent fetch_or() updates preserve every bit in shared words");
}

void test_concurrent_resets_in_shared_words() {
    print_test_header("Concurrent reset() operations on shared atomic words");
    std::cout
        << "  We first set all 256 bits. Eight threads then clear disjoint bit\n"
        << "  subsets in the same underlying words using fetch_and(). If updates\n"
        << "  were non-atomic, one thread could overwrite another thread's clear.\n";

    constexpr std::size_t bit_count = 256;
    constexpr std::size_t thread_count = 8;

    AtomicBitVector bits(bit_count);
    for (std::size_t i = 0; i < bit_count; ++i) {
        bits.set(i);
    }

    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([thread, &bits, &ready, &start]() {
            ready.fetch_add(1, std::memory_order_relaxed);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (std::size_t index = thread;
                 index < bit_count;
                 index += thread_count) {
                bits.reset(index);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != thread_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    bool every_bit_clear = true;
    for (std::size_t i = 0; i < bit_count; ++i) {
        every_bit_clear = every_bit_clear && !bits.test(i);
    }

    check(every_bit_clear,
          "concurrent fetch_and() updates preserve every clear in shared words");
}

} // namespace

int main() {
    std::cout
        << "BitVector / AtomicBitVector regression test\n"
        << "=======================================\n"
        << "This test checks the shared packed-bit semantics of both classes,\n"
        << "BitVector structural operations, and AtomicBitVector concurrency.\n";

    test_non_atomic_basic_bit_operations();
    test_non_atomic_bounds_and_empty_vector();
    test_non_atomic_structural_operations();

    test_basic_bit_operations();
    test_bounds_and_empty_vector();
    test_concurrent_sets_in_shared_words();
    test_concurrent_resets_in_shared_words();

    std::cout << "\nSummary\n-------\n"
              << "Performed checks: " << performed_checks << '\n'
              << "Failed checks:    " << failed_checks << '\n';

    if (failed_checks == 0) {
        std::cout << "RESULT: PASS - BitVector and AtomicBitVector behaved as expected.\n";
        return 0;
    }

    std::cerr << "RESULT: FAIL - inspect the failed checks above.\n";
    return 1;
}
