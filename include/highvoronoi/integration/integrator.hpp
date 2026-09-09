#pragma once

#include <highvoronoi/core/detail/progress_meter.hpp>

/**
 * @file integrator.hpp
 * @brief Algorithm-neutral cellwise integration driver.
 *
 * This follows Julia's `_integrate(...)` control flow deliberately:
 *
 *   1. construct an IntegralView / IntegrationView,
 *   2. for every finite dirty/new cell: prepare that cell, compute it immediately,
 *      before moving to the next cell,
 *   3. after the complete first pass, run `cleanup_cell` over those finite cells,
 *   4. finalize any unbounded dirty/new cells numerically without invoking the
 *      concrete integration algorithm,
 *   5. publish all prepared transactions and clear dirty state.
 *
 * The driver does not contain Monte-Carlo, polygon, or heuristic mathematics.
 * Those details belong to an algorithm object that provides `integrate_cell`
 * and, optionally, `cleanup_cell`.
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

struct IntegrationReport {
    /** Cells passed through the selected concrete integration algorithm. */
    std::size_t updated_cells = 0;
    std::size_t recomputed_interfaces = 0;

    /** NEW/DIRTY unbounded cells finalized without invoking the algorithm. */
    std::size_t unbounded_cells = 0;

    /** Interface occurrences proven unbounded and therefore stored as +infinity. */
    std::size_t unbounded_interfaces = 0;
};

/** Optional progress reporting for one complete integration update. */
struct IntegrationProgressOptions {
    bool enabled = false;
    std::chrono::nanoseconds delta_t = std::chrono::seconds(1);
    std::string subject = "Integration";
    std::ostream* output = &std::cout;
};

/** Execute the first integration pass on the caller thread. */
struct SerialIntegrationExecution {
    [[nodiscard]] std::size_t worker_count(std::size_t) const noexcept {
        return std::size_t{1};
    }
};

/**
 * Parallel integration policy.
 *
 * The global finite NEW/DIRTY work prefix is split into contiguous ranges
 * [k*N/K,(k+1)*N/K). Each worker gets a private integration view which moves
 * exactly its range to the front. The concrete algorithm must explicitly
 * provide make_worker_algorithm(worker), preventing algorithms with shared
 * mutable pass state (currently FastPolygon) from being parallelized by
 * accident. Cleanup and publication remain serial transactional phases.
 */
struct ParallelIntegrationExecution {
    std::size_t workers = 1;

    explicit ParallelIntegrationExecution(std::size_t worker_count) noexcept
        : workers(worker_count == 0 ? 1 : worker_count) {}

    [[nodiscard]] std::size_t worker_count(std::size_t work_size) const noexcept {
        if (work_size == 0) {
            return 1;
        }
        return std::min(workers, work_size);
    }
};

namespace detail {

/** Guard against recursive integral dependency cycles such as A -> B -> A. */
inline std::vector<const void*>& active_integral_stack() {
    static thread_local std::vector<const void*> stack;
    return stack;
}

class IntegrationCycleGuard final {
public:
    explicit IntegrationCycleGuard(const void* integral)
        : integral_(integral) {
        auto& stack = active_integral_stack();
        if (std::find(stack.begin(), stack.end(), integral_) != stack.end()) {
            throw std::logic_error(
                "Cyclic integration dependency detected");
        }
        stack.push_back(integral_);
    }

    ~IntegrationCycleGuard() {
        auto& stack = active_integral_stack();
        if (!stack.empty() && stack.back() == integral_) {
            stack.pop_back();
            return;
        }
        const auto found = std::find(stack.begin(), stack.end(), integral_);
        if (found != stack.end()) {
            stack.erase(found);
        }
    }

    IntegrationCycleGuard(const IntegrationCycleGuard&) = delete;
    IntegrationCycleGuard& operator=(const IntegrationCycleGuard&) = delete;

private:
    const void* integral_ = nullptr;
};

template <class Algorithm, class Integral, class = void>
struct HasPrepareIntegration : std::false_type {};

template <class Algorithm, class Integral>
struct HasPrepareIntegration<
    Algorithm,
    Integral,
    std::void_t<decltype(std::declval<Algorithm&>().prepare_integration(
        std::declval<Integral&>()))>> : std::true_type {};

template <class Algorithm, class Integral, class Execution, class = void>
struct HasExecutionAwarePrepareIntegration : std::false_type {};

template <class Algorithm, class Integral, class Execution>
struct HasExecutionAwarePrepareIntegration<
    Algorithm,
    Integral,
    Execution,
    std::void_t<decltype(std::declval<Algorithm&>().prepare_integration(
        std::declval<Integral&>(),
        std::declval<Execution>()))>> : std::true_type {};

/**
 * Preflight before the target IntegrationView freezes NEW/DIRTY scheduling.
 *
 * Source-dependent algorithms may optionally accept the selected execution
 * policy. HeuristicAlgorithm uses this to update a parallel-capable geometry
 * source with the same execution policy before the target pass is constructed.
 * Existing one-argument prepare_integration() hooks remain unchanged.
 */
template <class Algorithm, class Integral, class Execution>
void maybe_prepare_integration(
    Algorithm& algorithm,
    Integral& integral,
    Execution execution) {
    if constexpr (HasExecutionAwarePrepareIntegration<
                      Algorithm, Integral, Execution>::value) {
        algorithm.prepare_integration(integral, execution);
    } else if constexpr (HasPrepareIntegration<Algorithm, Integral>::value) {
        algorithm.prepare_integration(integral);
    }
}

template <class Algorithm, class Integral, class = void>
struct HasPrepareParallelIntegration : std::false_type {};

template <class Algorithm, class Integral>
struct HasPrepareParallelIntegration<
    Algorithm,
    Integral,
    std::void_t<decltype(std::declval<Algorithm&>().prepare_parallel_integration(
        std::declval<Integral&>(),
        std::declval<std::size_t>()))>> : std::true_type {};

template <class Algorithm, class Integral>
void maybe_prepare_parallel_integration(
    Algorithm& algorithm,
    Integral& integral,
    std::size_t workers) {
    if constexpr (HasPrepareParallelIntegration<Algorithm, Integral>::value) {
        algorithm.prepare_parallel_integration(integral, workers);
    }
}

template <class Algorithm, class Pass, class = void>
struct HasBeginPass : std::false_type {};

template <class Algorithm, class Pass>
struct HasBeginPass<
    Algorithm,
    Pass,
    std::void_t<decltype(std::declval<Algorithm&>().begin_pass(
        std::declval<Pass&>()))>> : std::true_type {};

template <class Algorithm, class Pass, class = void>
struct HasEndPass : std::false_type {};

template <class Algorithm, class Pass>
struct HasEndPass<
    Algorithm,
    Pass,
    std::void_t<decltype(std::declval<Algorithm&>().end_pass(
        std::declval<Pass&>()))>> : std::true_type {};

template <class Algorithm, class = void>
struct HasMakeWorkerAlgorithm : std::false_type {};

template <class Algorithm>
struct HasMakeWorkerAlgorithm<
    Algorithm,
    std::void_t<decltype(std::declval<Algorithm&>().make_worker_algorithm(
        std::declval<std::size_t>()))>> : std::true_type {};

template <class View, class = void>
struct HasMakeWorkerView : std::false_type {};

template <class View>
struct HasMakeWorkerView<
    View,
    std::void_t<decltype(std::declval<View&>().make_worker_view(
        std::declval<std::size_t>(),
        std::declval<std::size_t>()))>> : std::true_type {};

template <class View, class = void>
struct HasEnsureUpdateNeighboursCurrent : std::false_type {};

template <class View>
struct HasEnsureUpdateNeighboursCurrent<
    View,
    std::void_t<decltype(std::declval<View&>().ensure_update_neighbours_current())>>
    : std::true_type {};

template <class View, class Update, class = void>
struct HasRebaseWorkerUpdate : std::false_type {};

template <class View, class Update>
struct HasRebaseWorkerUpdate<
    View,
    Update,
    std::void_t<decltype(std::declval<View&>().rebase_worker_update(
        std::declval<Update&>(),
        std::declval<typename View::Index>(),
        std::declval<Update&>()))>> : std::true_type {};

template <class View, class Update, class = void>
struct HasPrepareCleanup : std::false_type {};

template <class View, class Update>
struct HasPrepareCleanup<
    View,
    Update,
    std::void_t<decltype(std::declval<View&>().prepare_cleanup(
        std::declval<Update&>()))>> : std::true_type {};

template <class Algorithm, class Pass, class = void>
struct HasCleanupPassCell : std::false_type {};

template <class Algorithm, class Pass>
struct HasCleanupPassCell<
    Algorithm,
    Pass,
    std::void_t<decltype(std::declval<Algorithm&>().cleanup_cell(
        std::declval<Pass&>(),
        std::declval<std::size_t>()))>> : std::true_type {};

template <class Algorithm, class Update, class = void>
struct HasCleanupUpdate : std::false_type {};

template <class Algorithm, class Update>
struct HasCleanupUpdate<
    Algorithm,
    Update,
    std::void_t<decltype(std::declval<Algorithm&>().cleanup_cell(
        std::declval<Update&>()))>> : std::true_type {};

template <class Algorithm, class Pass>
void maybe_begin_pass(Algorithm& algorithm, Pass& pass) {
    if constexpr (HasBeginPass<Algorithm, Pass>::value) {
        algorithm.begin_pass(pass);
    }
}

template <class Algorithm, class Pass>
void maybe_end_pass(Algorithm& algorithm, Pass& pass) {
    if constexpr (HasEndPass<Algorithm, Pass>::value) {
        algorithm.end_pass(pass);
    }
}

template <class Algorithm, class Pass, class = void>
struct HasIntegratePassCell : std::false_type {};

template <class Algorithm, class Pass>
struct HasIntegratePassCell<
    Algorithm,
    Pass,
    std::void_t<decltype(std::declval<Algorithm&>().integrate_cell(
        std::declval<Pass&>(),
        std::declval<std::size_t>()))>> : std::true_type {};

/**
 * Dispatch the first-pass call. Algorithms that need cross-cell first-pass
 * state (the Julia Polygon_Integrator is the first such case) may accept the
 * complete IntegrationPass plus the current position. Existing algorithms keep
 * the simpler integrate_cell(Update&) contract unchanged.
 */
template <class Algorithm, class Pass>
void call_integrate_cell(
    Algorithm& algorithm,
    Pass& pass,
    std::size_t position) {
    if constexpr (HasIntegratePassCell<Algorithm, Pass>::value) {
        algorithm.integrate_cell(pass, position);
    } else {
        algorithm.integrate_cell(pass.writable_cell(position));
    }
}

template <class Algorithm, class Pass>
void maybe_cleanup_cell(Algorithm& algorithm, Pass& pass, std::size_t cell) {
    using Update = typename Pass::Update;
    if constexpr (HasCleanupPassCell<Algorithm, Pass>::value) {
        algorithm.cleanup_cell(pass, cell);
    } else if constexpr (HasCleanupUpdate<Algorithm, Update>::value) {
        algorithm.cleanup_cell(pass.writable_cell(cell));
    }
}

template <class View, class Pass, class = void>
struct HasUnboundedUpdateFinalization : std::false_type {};

template <class View, class Pass>
struct HasUnboundedUpdateFinalization<
    View,
    Pass,
    std::void_t<
        decltype(std::declval<const View&>().unbounded_update_count()),
        decltype(std::declval<const View&>().unbounded_update_cell(
            std::declval<std::size_t>())),
        decltype(std::declval<View&>().finalize_unbounded_update(
            std::declval<const Pass&>(),
            std::declval<typename Pass::Update&>()))>> : std::true_type {};

} // namespace detail

/**
 * @brief Mutable state of the finite-cell portion of one integration update.
 *
 * The pass owns the CellUpdate objects needed by the later cleanup phase.  The
 * first pass is intentionally cellwise: one cell is prepared, calculated and
 * completed before the next cell is prepared.  This preserves the Julia
 * algorithm's cell-local ordering while keeping C++ publication deferred until
 * the cleanup phase has completed.
 */
template <class IntegrationViewT>
class IntegrationPass final {
public:
    using View = IntegrationViewT;
    using Index = typename View::Index;
    using Update = typename View::CellUpdate;

    explicit IntegrationPass(View& view)
        : view_(&view), updates_(view.update_count()) {}

    IntegrationPass(const IntegrationPass&) = delete;
    IntegrationPass& operator=(const IntegrationPass&) = delete;

    [[nodiscard]] View& view() noexcept { return *view_; }
    [[nodiscard]] const View& view() const noexcept { return *view_; }

    [[nodiscard]] std::size_t size() const noexcept { return updates_.size(); }

    [[nodiscard]] Update& writable_cell(std::size_t position) {
        return updates_.at(position);
    }

    [[nodiscard]] const Update& cell(std::size_t position) const {
        return updates_.at(position);
    }

    /** Prepare exactly one cell and leave its cell-local geometry active. */
    void prepare_cell(std::size_t position) {
        view_->prepare_update(
            static_cast<Index>(position),
            updates_.at(position));
    }

    /** Restore the cell-local geometry required by the cleanup algorithm. */
    void prepare_cleanup(std::size_t position) {
        if constexpr (detail::HasPrepareCleanup<View, Update>::value) {
            view_->prepare_cleanup(updates_.at(position));
        } else {
            (void)position;
        }
    }

    /** Publish cleanup corrections and finalize every prepared transaction. */
    void commit_all() {
        for (auto& update : updates_) {
            view_->commit_update(update);
        }
    }

private:
    View* view_ = nullptr;
    std::vector<Update> updates_;
};

/**
 * @brief Full integration runner owning only orchestration.
 *
 * The first pass preserves the Julia order
 * `prepare(cell) -> integrate(cell) -> next cell`. Most algorithms receive the
 * prepared CellUpdate directly; algorithms that need already-computed first-pass
 * cells may opt into the IntegrationPass + position callback. Cleanup runs only
 * after that complete cellwise pass and may access/modify all CellUpdate objects.
 */
template <class IntegralT,
          class AlgorithmT,
          class ExecutionT = SerialIntegrationExecution>
class Integrator final {
public:
    using Integral = IntegralT;
    using Algorithm = AlgorithmT;
    using Execution = ExecutionT;

    Integrator(
        Integral& integral,
        Algorithm& algorithm,
        Execution execution = Execution{},
        IntegrationProgressOptions progress = {})
        : integral_(&integral),
          algorithm_(&algorithm),
          execution_(execution),
          progress_options_(std::move(progress)) {}

    IntegrationReport integrate() {
        detail::IntegrationCycleGuard cycle_guard(
            static_cast<const void*>(integral_));

        // Source-dependent algorithms may validate/update prerequisite
        // integrals before NEW/DIRTY/CLEAN target scheduling is frozen.
        detail::maybe_prepare_integration(
            *algorithm_,
            *integral_,
            execution_);

        auto view = make_integration_view(*integral_);
        IntegrationPass<decltype(view)> pass(view);
        IntegrationReport report;

        std::unique_ptr<detail::ProgressMeter> progress;
        if (progress_options_.enabled && pass.size() != std::size_t{0}) {
            if (progress_options_.output == nullptr) {
                throw std::invalid_argument(
                    "Integration progress output stream must not be null");
            }
            progress = std::make_unique<detail::ProgressMeter>(
                pass.size(),
                progress_options_.delta_t,
                progress_options_.subject,
                *progress_options_.output);
        }

        const std::size_t workers = execution_.worker_count(pass.size());
        if (workers <= std::size_t{1} || pass.size() <= std::size_t{1}) {
            detail::maybe_begin_pass(*algorithm_, pass);

            // Julia first phase: prepare one cell and calculate it immediately,
            // then continue with the next cell.
            run_serial_first_pass(pass, report, progress.get());
        } else {
            run_parallel_first_pass(view, pass, report, workers, progress.get());

            // Parallel workers own their first-pass scratch only. Cleanup is a
            // global serial phase over the rebased master transactions, so give
            // the original algorithm one fresh master-view workspace here.
            detail::maybe_begin_pass(*algorithm_, pass);
        }

        // Julia second phase: cleanup only after every update cell completed its
        // first pass.  Re-activate each cell's master-view geometry before the
        // algorithm-specific cleanup.
        for (std::size_t position = 0; position < pass.size(); ++position) {
            pass.prepare_cleanup(position);
            detail::maybe_cleanup_cell(*algorithm_, pass, position);
        }

        detail::maybe_end_pass(*algorithm_, pass);

        // Unbounded cells are deliberately outside the complete concrete
        // integrator lifecycle. Prepare their transactions only after every
        // finite cell has completed first-pass, cleanup, and end_pass so no
        // algorithm hook ever observes unbounded cell-local geometry. Finite
        // reciprocal interfaces are copied from the still-unpublished pass state.
        std::vector<typename decltype(pass)::Update> unbounded_updates;
        if constexpr (detail::HasUnboundedUpdateFinalization<
                          decltype(view), decltype(pass)>::value) {
            const std::size_t count = view.unbounded_update_count();
            unbounded_updates.resize(count);
            for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
                const auto cell = view.unbounded_update_cell(ordinal);
                auto& update = unbounded_updates[ordinal];
                view.prepare_update(cell, update);
                report.unbounded_interfaces +=
                    view.finalize_unbounded_update(pass, update);
            }
            report.unbounded_cells += count;
        }

        // Publish only after all finite and unbounded transactions were prepared.
        pass.commit_all();
        for (auto& update : unbounded_updates) {
            view.commit_update(update);
        }
        view.finish_update();

        if (progress) {
            progress->finish();
        }

        return report;
    }

private:
    template <class Pass>
    void run_serial_first_pass(
        Pass& pass,
        IntegrationReport& report,
        detail::ProgressMeter* progress) {
        for (std::size_t position = 0; position < pass.size(); ++position) {
            pass.prepare_cell(position);
            auto& update = pass.writable_cell(position);
            count_interfaces(update, report);
            detail::call_integrate_cell(*algorithm_, pass, position);
            ++report.updated_cells;
            if (progress != nullptr) {
                progress->increase();
            }
        }
    }

    template <class View, class Pass>
    void run_parallel_first_pass(
        View& master_view,
        Pass& master_pass,
        IntegrationReport& report,
        std::size_t workers,
        detail::ProgressMeter* progress) {
        if constexpr (!detail::HasMakeWorkerAlgorithm<Algorithm>::value) {
            throw std::logic_error(
                "This integration algorithm does not provide parallel worker state");
        } else if constexpr (!detail::HasMakeWorkerView<View>::value ||
                             !detail::HasRebaseWorkerUpdate<View, typename Pass::Update>::value) {
            throw std::logic_error(
                "Parallel integration is currently implemented only for ordinary "
                "VoronoiIntegrationView");
        } else {
            // Shared algorithm state needed only by parallel execution is
            // initialized once here, after worker_count is known and before any
            // worker algorithm/thread exists. Serial integration never pays for it.
            detail::maybe_prepare_parallel_integration(
                *algorithm_,
                *integral_,
                workers);

            // Neighbour computation owns mesh-local persistent workspace. Resolve
            // it once, serially, before worker views start reading the database.
            if constexpr (detail::HasEnsureUpdateNeighboursCurrent<View>::value) {
                master_view.ensure_update_neighbours_current();
            }

            using WorkerViewPointer = decltype(
                master_view.make_worker_view(std::size_t{0}, std::size_t{0}));
            using WorkerView = typename WorkerViewPointer::element_type;
            using WorkerPass = IntegrationPass<WorkerView>;

            static_assert(
                std::is_same_v<typename WorkerPass::Update, typename Pass::Update>,
                "Worker and master IntegrationView must share one CellUpdate type");

            std::vector<WorkerViewPointer> worker_views;
            std::vector<std::unique_ptr<WorkerPass>> worker_passes;
            std::vector<IntegrationReport> worker_reports(workers);
            std::vector<std::exception_ptr> worker_exceptions(workers);
            std::vector<std::thread> threads;

            worker_views.reserve(workers);
            worker_passes.reserve(workers);
            threads.reserve(workers);

            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = (master_pass.size() * worker) / workers;
                const std::size_t end =
                    (master_pass.size() * (worker + std::size_t{1})) / workers;
                worker_views.push_back(master_view.make_worker_view(begin, end));
                worker_passes.push_back(
                    std::make_unique<WorkerPass>(*worker_views.back()));
            }

            try {
                for (std::size_t worker = 0; worker < workers; ++worker) {
                    threads.emplace_back([&, worker] {
                        try {
                            auto worker_algorithm =
                                algorithm_->make_worker_algorithm(worker);
                            WorkerPass& pass = *worker_passes[worker];
                            detail::maybe_begin_pass(worker_algorithm, pass);

                            for (std::size_t position = 0;
                                 position < pass.size();
                                 ++position) {
                                pass.prepare_cell(position);
                                auto& update = pass.writable_cell(position);
                                count_interfaces(update, worker_reports[worker]);
                                detail::call_integrate_cell(
                                    worker_algorithm,
                                    pass,
                                    position);
                                ++worker_reports[worker].updated_cells;
                                if (progress != nullptr) {
                                    progress->increase();
                                }
                            }

                            detail::maybe_end_pass(worker_algorithm, pass);
                        } catch (...) {
                            worker_exceptions[worker] = std::current_exception();
                        }
                    });
                }
            } catch (...) {
                // std::thread construction itself may fail. Never let a vector
                // containing joinable threads unwind into std::terminate.
                for (auto& thread : threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
                throw;
            }

            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            for (const auto& exception : worker_exceptions) {
                if (exception) {
                    std::rethrow_exception(exception);
                }
            }

            // Worker numbering is intentionally local. Convert each completed
            // transaction back to the single master [NEW][DIRTY][CLEAN] view
            // before global cleanup. This remains purely in-memory: persistent
            // IntegralData is still untouched if a later cleanup throws.
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = (master_pass.size() * worker) / workers;
                WorkerPass& worker_pass = *worker_passes[worker];
                for (std::size_t local = 0; local < worker_pass.size(); ++local) {
                    const std::size_t master_position = begin + local;
                    master_view.rebase_worker_update(
                        worker_pass.writable_cell(local),
                        static_cast<typename View::Index>(master_position),
                        master_pass.writable_cell(master_position));
                }

                report.updated_cells += worker_reports[worker].updated_cells;
                report.recomputed_interfaces +=
                    worker_reports[worker].recomputed_interfaces;
            }
        }
    }

    template <class Update>
    static void count_interfaces(const Update& update, IntegrationReport& report) {
        for (std::size_t ordinal = 0;
             ordinal < update.neighbours().size();
             ++ordinal) {
            if (update.interface_requires_recompute(ordinal)) {
                ++report.recomputed_interfaces;
            }
        }
    }

    Integral* integral_ = nullptr;
    Algorithm* algorithm_ = nullptr;
    Execution execution_{};
    IntegrationProgressOptions progress_options_{};
};

/** Convenience function for the default serial execution policy. */
template <class IntegralT, class AlgorithmT>
IntegrationReport integrate(IntegralT& integral, AlgorithmT& algorithm) {
    return Integrator<IntegralT, AlgorithmT>(integral, algorithm).integrate();
}

/** Convenience function when the caller wants to select the execution policy. */
template <class IntegralT, class AlgorithmT, class ExecutionT>
IntegrationReport integrate(
    IntegralT& integral,
    AlgorithmT& algorithm,
    ExecutionT execution) {
    return Integrator<IntegralT, AlgorithmT, ExecutionT>(
        integral,
        algorithm,
        execution).integrate();
}

/** Convenience function with explicit execution and progress policies. */
template <class IntegralT, class AlgorithmT, class ExecutionT>
IntegrationReport integrate(
    IntegralT& integral,
    AlgorithmT& algorithm,
    ExecutionT execution,
    IntegrationProgressOptions progress) {
    return Integrator<IntegralT, AlgorithmT, ExecutionT>(
        integral,
        algorithm,
        execution,
        std::move(progress)).integrate();
}

/** Convenience function for serial integration with progress reporting. */
template <class IntegralT, class AlgorithmT>
IntegrationReport integrate(
    IntegralT& integral,
    AlgorithmT& algorithm,
    IntegrationProgressOptions progress) {
    return Integrator<IntegralT, AlgorithmT, SerialIntegrationExecution>(
        integral,
        algorithm,
        SerialIntegrationExecution{},
        std::move(progress)).integrate();
}

} // namespace highvoronoi
