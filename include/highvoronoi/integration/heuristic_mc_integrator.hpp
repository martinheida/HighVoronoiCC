#pragma once

/**
 * @file heuristic_mc_integrator.hpp
 * @brief Monte-Carlo geometry followed by heuristic function quadrature.
 *
 * This is the C++ counterpart of Julia HeuristicMCIntegrator. The current C++
 * transaction lifecycle makes one ordering adjustment deliberate: Monte-Carlo
 * first computes geometry for every update cell and its cleanup then finalizes
 * shared interface areas. Only after that complete MC cleanup does the heuristic
 * stage consume those final areas. This avoids constructing function integrals
 * from pre-symmetrization MC geometry and requires no second correction pass.
 */

#include <highvoronoi/integration/heuristic_integrator.hpp>
#include <highvoronoi/integration/monte_carlo_integrator.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace highvoronoi {

template <class IntegralT, class FunctionT>
class HeuristicMCAlgorithm final {
public:
    using Integral = IntegralT;
    using Function = FunctionT;
    using MonteCarlo = MonteCarloAlgorithm<
        NoMonteCarloFunction,
        NoMonteCarloFunction>;
    using Evaluator = detail::HeuristicCellEvaluator<Integral, Function>;

    HeuristicMCAlgorithm(
        Integral& integral,
        Function function,
        MonteCarloOptions options = {})
        : integral_(&integral),
          function_(std::move(function)),
          monte_carlo_(make_mc_options(options)),
          evaluator_(integral, function_) {}

    void prepare_integration(Integral& integral) const {
        if (&integral != integral_) {
            throw std::logic_error(
                "HeuristicMCAlgorithm used with a different target integral");
        }
        const auto& data = integral.data();
        if (!data.stores_volume() || !data.stores_area()) {
            throw std::invalid_argument(
                "HeuristicMC target must store volume and area");
        }
        if (!data.stores_bulk_integral() &&
            !data.stores_interface_integral()) {
            throw std::invalid_argument(
                "HeuristicMC target must store bulk and/or interface integrals");
        }
    }

    template <class Update>
    void integrate_cell(Update& update) const {
        // Julia's MC sub-integrator runs with heuristic=true: geometry only.
        monte_carlo_.integrate_cell(update);
    }

    /**
     * Parallel first-pass worker.
     *
     * Deliberately exposes only integrate_cell(): worker-local end_pass must not
     * run heuristic quadrature, because Monte-Carlo interface areas are not
     * globally symmetrized until the master cleanup phase after all workers join.
     */
    class ParallelWorker final {
    public:
        ParallelWorker(
            const HeuristicMCAlgorithm& parent,
            std::size_t worker)
            : monte_carlo_worker_(
                  parent.monte_carlo_.make_worker_algorithm(worker)) {}

        template <class Update>
        void integrate_cell(Update& update) {
            monte_carlo_worker_.integrate_cell(update);
        }

    private:
        typename MonteCarlo::ParallelWorker monte_carlo_worker_;
    };

    [[nodiscard]] ParallelWorker make_worker_algorithm(
        std::size_t worker) const {
        return ParallelWorker(*this, worker);
    }

    template <class Pass>
    void cleanup_cell(Pass& pass, std::size_t position) const {
        // First finish the MC geometry/symmetrization for the complete update
        // prefix. The heuristic stage is intentionally deferred to end_pass().
        monte_carlo_.cleanup_cell(pass, position);
    }

    template <class Pass>
    void end_pass(Pass& pass) {
        // At this point every MC cleanup_cell has run. Re-activate each cell's
        // geometry and construct function integrals from the final MC areas.
        for (std::size_t position = 0; position < pass.size(); ++position) {
            pass.prepare_cleanup(position);
            auto& update = pass.writable_cell(position);
            evaluator_.integrate_cell(
                update,
                update.data().volume(),
                update.data().area());
        }
    }

    [[nodiscard]] const MonteCarloOptions& monte_carlo_options() const noexcept {
        return monte_carlo_.options();
    }

private:
    [[nodiscard]] static MonteCarloOptions make_mc_options(
        MonteCarloOptions options) {
        options.calculate_area = true;
        options.heuristic = true;
        // Julia fixes the unused bulk sample count to one in HeuristicMC.
        options.bulk_samples_per_ray = std::size_t{1};
        return options;
    }

    Integral* integral_ = nullptr;
    Function function_;
    MonteCarlo monte_carlo_;
    Evaluator evaluator_;
};

template <class Integral, class Function>
[[nodiscard]] auto make_heuristic_mc_algorithm(
    Integral& integral,
    Function&& function,
    MonteCarloOptions options = {}) {
    using StoredFunction = std::decay_t<Function>;
    return HeuristicMCAlgorithm<Integral, StoredFunction>(
        integral,
        std::forward<Function>(function),
        options);
}

} // namespace highvoronoi
