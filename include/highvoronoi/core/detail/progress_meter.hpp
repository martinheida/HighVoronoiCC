#pragma once

#include <highvoronoi/core/detail/locks.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace highvoronoi::detail {

/**
 * @brief Parallel-safe progress and elapsed-time reporter.
 *
 * ProgressMeter is intended for coarse progress reporting from performance-
 * critical parallel code. Calls to increase() only perform an atomic counter
 * increment, one steady-clock query and a small number of relaxed atomic
 * operations on the normal path. Terminal output is produced at most once per
 * configured time interval.
 *
 * If several threads notice that the output interval has expired at the same
 * time, compare/exchange on last_output_ns_ elects exactly one of them as the
 * reporter. The actual stream write is serialized by the project's FIFO spin
 * lock so progress lines from ProgressMeter instances do not interleave.
 *
 * An optional external TimeCounter receives the latest elapsed time observed
 * by the meter, expressed in nanoseconds since construction. Updates are
 * monotone even when concurrent calls complete out of order.
 *
 * @note The external TimeCounter, when supplied, must outlive this object.
 * @note The destructor does not print. Call finish() explicitly if a final
 *       progress line is desired.
 */
class ProgressMeter final {
public:
    using Clock = std::chrono::steady_clock;
    using Nanoseconds = std::chrono::nanoseconds;
    using TimeRep = std::int64_t;
    using TimeCounter = std::atomic<TimeRep>;

    /**
     * @brief Construct a progress meter without an external time counter.
     *
     * @param max_steps Number of steps corresponding to 100 percent.
     * @param delta_t Minimum elapsed time between automatic outputs.
     * @param subject Optional text prepended to every progress line.
     * @param output Output stream; defaults to std::cout.
     */
    explicit ProgressMeter(
        std::size_t max_steps,
        Nanoseconds delta_t,
        std::string subject = {},
        std::ostream& output = std::cout)
        : ProgressMeter(
              max_steps,
              delta_t,
              nullptr,
              std::move(subject),
              output)
    {}

    /**
     * @brief Construct a progress meter with an external elapsed-time counter.
     *
     * The external counter is reset to zero at construction and subsequently
     * stores elapsed nanoseconds since construction.
     */
    ProgressMeter(
        std::size_t max_steps,
        Nanoseconds delta_t,
        TimeCounter& total,
        std::string subject = {},
        std::ostream& output = std::cout)
        : ProgressMeter(
              max_steps,
              delta_t,
              &total,
              std::move(subject),
              output)
    {
        total.store(0, std::memory_order_relaxed);
    }

    ProgressMeter(const ProgressMeter&) = delete;
    ProgressMeter& operator=(const ProgressMeter&) = delete;
    ProgressMeter(ProgressMeter&&) = delete;
    ProgressMeter& operator=(ProgressMeter&&) = delete;

    ~ProgressMeter() = default;

    /**
     * @brief Increase the completed-step counter and update progress state.
     *
     * This function may be called concurrently by arbitrary worker threads.
     */
    void increase(std::size_t steps = 1)
    {
        counter_.fetch_add(steps, std::memory_order_relaxed);
        update();
    }

    /**
     * @brief Refresh elapsed time and emit a line if the output interval expired.
     *
     * This does not modify the step counter.
     */
    void update()
    {
        const TimeRep elapsed = current_elapsed_ns();
        publish_total(elapsed);

        if (claim_output(elapsed, false)) {
            print_progress(elapsed);
        }
    }

    /**
     * @brief Emit a final progress line without waiting for delta_t.
     *
     * The call is intended after worker threads have joined. The destructor is
     * deliberately silent, so callers retain explicit control over I/O.
     */
    void finish()
    {
        const TimeRep elapsed = current_elapsed_ns();
        publish_total(elapsed);

        if (claim_output(elapsed, true)) {
            print_progress(elapsed);
        }
    }

    /** @brief Return the number of completed steps observed so far. */
    [[nodiscard]]
    std::size_t counter() const noexcept
    {
        return counter_.load(std::memory_order_relaxed);
    }

    /** @brief Return the number of steps corresponding to 100 percent. */
    [[nodiscard]]
    std::size_t max_steps() const noexcept
    {
        return max_steps_;
    }

    /** @brief Return the current elapsed time in nanoseconds since construction. */
    [[nodiscard]]
    TimeRep elapsed_nanoseconds() const noexcept
    {
        const TimeRep elapsed = current_elapsed_ns();
        publish_total(elapsed);
        return elapsed;
    }

    /** @brief Return the current elapsed time in seconds since construction. */
    [[nodiscard]]
    double elapsed_seconds() const noexcept
    {
        return static_cast<double>(elapsed_nanoseconds()) * 1.0e-9;
    }

    /**
     * @brief Return constructor-to-last-progress-output time in nanoseconds.
     *
     * Returns zero until the first progress line has been claimed.
     */
    [[nodiscard]]
    TimeRep last_progress_nanoseconds() const noexcept
    {
        return last_output_ns_.load(std::memory_order_relaxed);
    }

    /** @brief Return constructor-to-last-progress-output time in seconds. */
    [[nodiscard]]
    double last_progress_seconds() const noexcept
    {
        return static_cast<double>(last_progress_nanoseconds()) * 1.0e-9;
    }

private:
    ProgressMeter(
        std::size_t max_steps,
        Nanoseconds delta_t,
        TimeCounter* total,
        std::string subject,
        std::ostream& output)
        : max_steps_(max_steps),
          delta_t_ns_(delta_t.count()),
          subject_(std::move(subject)),
          total_(total),
          output_(&output),
          start_(Clock::now())
    {
        if (max_steps_ == 0) {
            throw std::invalid_argument(
                "ProgressMeter requires max_steps > 0");
        }
        if (delta_t_ns_ <= 0) {
            throw std::invalid_argument(
                "ProgressMeter requires delta_t > 0");
        }
    }

    [[nodiscard]]
    TimeRep current_elapsed_ns() const noexcept
    {
        return std::chrono::duration_cast<Nanoseconds>(
                   Clock::now() - start_)
            .count();
    }

    /**
     * @brief Monotonically publish elapsed time to the optional external counter.
     *
     * A simple store would allow an older timestamp from a delayed thread to
     * overwrite a newer timestamp. C++17 has no atomic fetch_max, so the maximum
     * is implemented with compare/exchange.
     */
    void publish_total(TimeRep elapsed) const noexcept
    {
        if (total_ == nullptr) {
            return;
        }

        TimeRep current = total_->load(std::memory_order_relaxed);
        while (current < elapsed &&
               !total_->compare_exchange_weak(
                   current,
                   elapsed,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }

    /**
     * @brief Claim the right to produce the next progress output.
     *
     * compare_exchange provides the complete reporter election. If another
     * thread moves last_output_ns_ first, the failed caller observes the new
     * timestamp and normally falls back into the fresh delta_t interval.
     */
    [[nodiscard]]
    bool claim_output(TimeRep elapsed, bool force) noexcept
    {
        TimeRep previous =
            last_output_ns_.load(std::memory_order_relaxed);

        for (;;) {
            if (elapsed <= previous) {
                return false;
            }

            if (!force && elapsed - previous < delta_t_ns_) {
                return false;
            }

            if (last_output_ns_.compare_exchange_weak(
                    previous,
                    elapsed,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                return true;
            }

            // On failure, compare_exchange updates 'previous'. Re-check the
            // interval against the timestamp published by the winning thread.
        }
    }

    void print_progress(TimeRep elapsed)
    {
        const std::size_t current =
            counter_.load(std::memory_order_relaxed);

        const double percent =
            100.0 * static_cast<double>(current) /
            static_cast<double>(max_steps_);

        // Build the line before taking the spin lock. Only the actual stream
        // write/flush is serialized, keeping the FIFO critical section short.
        std::ostringstream line;
        if (!subject_.empty()) {
            line << subject_ << ' ';
        }

        line << std::fixed
             << std::setprecision(1)
             << percent
             << " %   "
             << std::setprecision(3)
             << static_cast<double>(elapsed) * 1.0e-9
             << " s\n";

        const std::string text = line.str();

        std::lock_guard<BusyFIFOLock> guard(output_lock_);
        output_->write(
            text.data(),
            static_cast<std::streamsize>(text.size()));
        output_->flush();
    }

    const std::size_t max_steps_;
    const TimeRep delta_t_ns_;
    const std::string subject_;

    TimeCounter* const total_;
    std::ostream* const output_;
    const Clock::time_point start_;

    std::atomic<std::size_t> counter_{0};
    std::atomic<TimeRep> last_output_ns_{0};

    // Shared by all ProgressMeter instances so their output lines cannot
    // interleave with one another. It does not serialize unrelated std::cout
    // users elsewhere in the program.
    inline static BusyFIFOLock output_lock_{};
};

} // namespace highvoronoi::detail
