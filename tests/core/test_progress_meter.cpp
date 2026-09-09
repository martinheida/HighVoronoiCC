#include <highvoronoi/core/detail/progress_meter.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using highvoronoi::detail::ProgressMeter;
using namespace std::chrono_literals;

std::size_t line_count(const std::string& text)
{
    return static_cast<std::size_t>(
        std::count(text.begin(), text.end(), '\n'));
}

struct TestState {
    int failures = 0;

    void check(bool condition, const char* message)
    {
        if (condition) {
            std::cout << "[OK]   " << message << '\n';
        } else {
            std::cerr << "[FAIL] " << message << '\n';
            ++failures;
        }
    }
};

void test_basic_timing_and_output(TestState& test)
{
    std::ostringstream output;
    ProgressMeter::TimeCounter total_ns{123};

    ProgressMeter meter(
        10,
        50ms,
        total_ns,
        "computing mesh...",
        output);

    test.check(meter.counter() == 0,
               "counter starts at zero");
    test.check(meter.max_steps() == 10,
               "max_steps is retained");
    test.check(total_ns.load(std::memory_order_relaxed) == 0,
               "external TOTAL counter is reset to zero");
    test.check(meter.last_progress_nanoseconds() == 0,
               "no progress timestamp exists before first output");

    meter.increase(2);

    test.check(meter.counter() == 2,
               "increase(steps) updates the atomic counter");
    test.check(output.str().empty(),
               "no output is produced before Delta_T expires");

    std::this_thread::sleep_for(70ms);
    meter.update();

    const std::string first_output = output.str();
    const auto first_progress = meter.last_progress_nanoseconds();

    test.check(line_count(first_output) == 1,
               "first expired interval produces exactly one line");
    test.check(first_output.find("computing mesh...") != std::string::npos,
               "subject is printed at the beginning of the progress line");
    test.check(first_output.find("20.0 %") != std::string::npos,
               "progress percentage reflects counter/max_steps");
    test.check(first_output.find(" s") != std::string::npos,
               "elapsed time is printed in seconds");
    test.check(first_progress >= 50'000'000,
               "last-progress timestamp measures from construction");
    test.check(total_ns.load(std::memory_order_relaxed) >= first_progress,
               "TOTAL contains at least the last published progress time");

    meter.update();
    test.check(line_count(output.str()) == 1,
               "a second call inside Delta_T produces no output");

    std::this_thread::sleep_for(70ms);
    meter.increase(3);

    test.check(meter.counter() == 5,
               "subsequent batched increase is accumulated");
    test.check(line_count(output.str()) == 2,
               "a later expired interval produces the next line");
    test.check(output.str().find("50.0 %") != std::string::npos,
               "later output uses the updated percentage");
}

void test_parallel_counter(TestState& test)
{
    constexpr std::size_t thread_count = 8;
    constexpr std::size_t iterations = 20'000;

    std::ostringstream output;
    ProgressMeter meter(
        thread_count * iterations,
        std::chrono::hours(1),
        "parallel counter",
        output);

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (std::size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&meter]() {
            for (std::size_t i = 0; i < iterations; ++i) {
                meter.increase();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    test.check(meter.counter() == thread_count * iterations,
               "parallel increase() calls lose no counter updates");
    test.check(output.str().empty(),
               "parallel counter updates do not bypass Delta_T");
}

void test_parallel_single_reporter(TestState& test)
{
    constexpr std::size_t thread_count = 8;

    std::ostringstream output;
    ProgressMeter::TimeCounter total_ns{0};
    ProgressMeter meter(
        thread_count,
        100ms,
        total_ns,
        "parallel reporter",
        output);

    // Put every worker beyond the same first output deadline before releasing
    // them together. Their increase() calls then contend for the same expired
    // last_output_ns_ value; compare/exchange must elect one reporter.
    std::this_thread::sleep_for(130ms);

    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (std::size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            meter.increase();
        });
    }

    start.store(true, std::memory_order_release);

    for (auto& thread : threads) {
        thread.join();
    }

    test.check(meter.counter() == thread_count,
               "all parallel reporter candidates increment the counter");
    test.check(line_count(output.str()) == 1,
               "one expired time window elects exactly one reporter");
    test.check(meter.last_progress_nanoseconds() >= 100'000'000,
               "parallel reporter publishes the expired-window timestamp");
    test.check(total_ns.load(std::memory_order_relaxed) >=
                   meter.last_progress_nanoseconds(),
               "parallel TOTAL updates remain monotone");
}

void test_finish_and_elapsed_access(TestState& test)
{
    std::ostringstream output;
    ProgressMeter::TimeCounter total_ns{0};

    ProgressMeter meter(
        10,
        std::chrono::hours(1),
        total_ns,
        "finish",
        output);

    meter.increase(5);
    std::this_thread::sleep_for(2ms);

    const auto elapsed_before_finish = meter.elapsed_nanoseconds();
    meter.finish();

    test.check(line_count(output.str()) == 1,
               "finish() forces one final progress line");
    test.check(output.str().find("50.0 %") != std::string::npos,
               "forced final line reports the current percentage");
    test.check(meter.last_progress_nanoseconds() >= elapsed_before_finish,
               "last-progress time is readable before destruction");
    test.check(meter.elapsed_seconds() > 0.0,
               "current elapsed time is readable in seconds");
    test.check(total_ns.load(std::memory_order_relaxed) >=
                   meter.last_progress_nanoseconds(),
               "elapsed access keeps external TOTAL current");
}

} // namespace

int main()
{
    TestState test;

    std::cout << "============================================================\n";
    std::cout << "ProgressMeter: timing, output gating and parallel semantics\n";
    std::cout << "============================================================\n";

    test_basic_timing_and_output(test);
    test_parallel_counter(test);
    test_parallel_single_reporter(test);
    test_finish_and_elapsed_access(test);

    if (test.failures != 0) {
        std::cerr << "ProgressMeter test failed with "
                  << test.failures << " failure(s).\n";
        return 1;
    }

    std::cout << "All ProgressMeter checks passed.\n";
    return 0;
}
