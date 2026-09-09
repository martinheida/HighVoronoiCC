#pragma once

/**
 * @file monte_carlo_integrator.hpp
 * @brief First concrete integral algorithm: Julia-style Monte-Carlo cell raycast.
 *
 * This file contains only the mathematical algorithm.  The surrounding
 * integration lifecycle is still owned by integrator.hpp:
 *
 *   - IntegrationView prepares one dirty/new cell and snapshots its complete
 *     cell-local generator geometry, including active boundary mirrors.
 *   - MonteCarloAlgorithm::integrate_cell(...) estimates volume, interface
 *     area and optional integrals for that prepared cell.
 *   - MonteCarloAlgorithm::cleanup_cell(...) performs the serial MC-specific
 *     interface symmetrisation for pairs that are part of the current update
 *     prefix.
 *   - The generic driver commits all cells only after cleanup has completed.
 *
 * The implementation intentionally mirrors Julia mcintegrator.jl.  Rays start
 * at the current generator, the first bisector hit among the prepared neighbour
 * list determines the interface ordinal, and boundary planes are represented by
 * the owning reflected neighbour points materialized in the CellUpdate.
 */

#include <highvoronoi/integration/high_voronoi_integration_view.hpp>
#include <highvoronoi/integration/integration_view.hpp>
#include <highvoronoi/integration/integrator.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

struct NoMonteCarloFunction final {};

namespace detail {

template <class Function>
struct IsNoMonteCarloFunction : std::false_type {};

template <>
struct IsNoMonteCarloFunction<NoMonteCarloFunction> : std::true_type {};

template <class T, class = void>
struct IsRangeLike : std::false_type {};

template <class T>
struct IsRangeLike<
    T,
    std::void_t<
        decltype(std::begin(std::declval<const T&>())),
        decltype(std::end(std::declval<const T&>()))>> : std::true_type {};

template <class Function, class Point, class Scalar>
void accumulate_function_value(
    const Function& function,
    const Point& point,
    Scalar weight,
    std::vector<Scalar>& target) {
    if (target.empty()) {
        return;
    }

    const auto value = function(point);
    using Value = std::decay_t<decltype(value)>;

    if constexpr (std::is_arithmetic_v<Value>) {
        if (target.size() != 1) {
            throw std::logic_error(
                "scalar Monte-Carlo integrand requires exactly one integral component");
        }
        target.front() += weight * static_cast<Scalar>(value);
    } else if constexpr (IsRangeLike<Value>::value) {
        std::size_t component = 0;
        for (const auto& entry : value) {
            if (component >= target.size()) {
                throw std::logic_error(
                    "Monte-Carlo integrand returned too many components");
            }
            target[component++] += weight * static_cast<Scalar>(entry);
        }
        if (component != target.size()) {
            throw std::logic_error(
                "Monte-Carlo integrand returned too few components");
        }
    } else {
        static_assert(
            std::is_arithmetic_v<Value> || IsRangeLike<Value>::value,
            "Monte-Carlo integrand must return a scalar or a range of scalars");
    }
}

template <class Function, class Point>
std::size_t function_component_count(const Function& function, const Point& point) {
    const auto value = function(point);
    using Value = std::decay_t<decltype(value)>;
    if constexpr (std::is_arithmetic_v<Value>) {
        return std::size_t{1};
    } else if constexpr (IsRangeLike<Value>::value) {
        return static_cast<std::size_t>(
            std::distance(std::begin(value), std::end(value)));
    } else {
        static_assert(
            std::is_arithmetic_v<Value> || IsRangeLike<Value>::value,
            "Monte-Carlo integrand must return a scalar or a range of scalars");
        return std::size_t{0};
    }
}

} // namespace detail

struct MonteCarloOptions {
    std::size_t interface_rays = 1000;
    std::size_t bulk_samples_per_ray = 100;
    std::uint64_t seed = 0x6d2b79f5d1b54a32ULL;
    bool calculate_area = true;
    bool heuristic = false;
};

template <class BulkFunctionT = NoMonteCarloFunction,
          class InterfaceFunctionT = NoMonteCarloFunction>
class MonteCarloAlgorithm final {
public:
    using BulkFunction = BulkFunctionT;
    using InterfaceFunction = InterfaceFunctionT;

    explicit MonteCarloAlgorithm(MonteCarloOptions options = {})
        : options_(options) {}

    MonteCarloAlgorithm(
        BulkFunction bulk,
        InterfaceFunction interface,
        MonteCarloOptions options = {})
        : options_(options),
          bulk_(std::move(bulk)),
          interface_(std::move(interface)) {}

    [[nodiscard]] const MonteCarloOptions& options() const noexcept {
        return options_;
    }

    template <class Integral>
    void configure_integral(Integral& integral) const {
        auto opts = integral.data().options();
        opts.volume = true;
        opts.area = options_.calculate_area || has_interface_function();
        opts.bulk_integral = has_bulk_function();
        opts.interface_integral = has_interface_function();
        (void)opts;
        // IntegralData options are construction-time options in the current
        // architecture.  This hook intentionally only documents the required
        // settings for callers/tests.  Reconfiguration belongs to a later
        // explicit enable/disable API, not to the algorithm constructor.
    }

    template <class Update>
    void integrate_cell(Update& update) const {
        WorkerData worker(options_.seed ^
            (0x9e3779b97f4a7c15ULL +
             static_cast<std::uint64_t>(update.cell()) * 0xbf58476d1ce4e5b9ULL));
        integrate_cell(update, worker);
    }

    struct WorkerData {
        explicit WorkerData(std::uint64_t seed)
            : rng(seed) {}
        std::mt19937_64 rng;
    };

    [[nodiscard]] WorkerData make_worker_data(std::size_t worker) const {
        return WorkerData(options_.seed ^
            (0xd1b54a32d192ed03ULL +
             static_cast<std::uint64_t>(worker) * 0x94d049bb133111ebULL));
    }

    /** One persistent RNG stream per parallel integration worker. */
    class ParallelWorker final {
    public:
        ParallelWorker(
            const MonteCarloAlgorithm& parent,
            std::size_t worker)
            : parent_(&parent),
              data_(parent.make_worker_data(worker)) {}

        template <class Update>
        void integrate_cell(Update& update) {
            parent_->integrate_cell(update, data_);
        }

    private:
        const MonteCarloAlgorithm* parent_ = nullptr;
        WorkerData data_;
    };

    [[nodiscard]] ParallelWorker make_worker_algorithm(
        std::size_t worker) const {
        return ParallelWorker(*this, worker);
    }

    template <class Update>
    void integrate_cell(Update& update, WorkerData& worker) const {
        auto& cell = update.data();
        auto& areas = cell.area();
        auto& bulk = cell.bulk_integral();
        auto& interfaces = cell.interface_integral();
        const auto& neighbours = cell.neighbours();

        const bool store_area = !areas.empty();
        const bool store_bulk = !bulk.empty() && !options_.heuristic;
        const bool store_interface = !interfaces.empty() && !options_.heuristic;

        if (store_area) {
            std::fill(areas.begin(), areas.end(), std::decay_t<decltype(cell.volume())>{});
        }
        if (store_bulk) {
            std::fill(bulk.begin(), bulk.end(), std::decay_t<decltype(bulk.front())>{});
        }
        if (store_interface) {
            std::fill(interfaces.begin(), interfaces.end(), std::decay_t<decltype(bulk.front())>{});
        }

        if (neighbours.empty()) {
            cell.set_volume(std::decay_t<decltype(cell.volume())>{});
            return;
        }

        // The generic IntegrationView has already activated this cell.  Consume
        // that cell-local extended-node state immediately, exactly as Julia's
        // integrate_cell -> integrate sequence does.  Neighbour points are only
        // temporary numerical scratch for this one Monte-Carlo call.
        auto& extended_nodes = update.extended_nodes();
        auto x0 = extended_nodes.node(update.cell());
        using Point = std::decay_t<decltype(x0)>;
        using Scalar = typename Point::Scalar;
        Point evaluation_point = x0;

        std::vector<Point> neighbour_points;
        neighbour_points.reserve(neighbours.size());
        for (const auto neighbour : neighbours) {
            neighbour_points.push_back(extended_nodes.node(neighbour));
        }

        const int dimension = static_cast<int>(x0.size());
        if (dimension <= 0) {
            throw std::logic_error("Monte-Carlo integrator requires positive dimension");
        }
        if (options_.interface_rays == 0) {
            throw std::invalid_argument("Monte-Carlo integrator requires at least one ray");
        }

        std::vector<Point> normals;
        normals.reserve(neighbour_points.size());
        for (const Point& xn : neighbour_points) {
            Point normal = xn - x0;
            const Scalar norm = normal.norm();
            if (!(norm > Scalar{0})) {
                throw std::logic_error(
                    "Monte-Carlo integrator found coincident cell/neighbour nodes");
            }
            normal /= norm;
            normals.push_back(std::move(normal));
        }

        Scalar volume_accumulator = Scalar{0};
        for (std::size_t sample = 0; sample < options_.interface_rays; ++sample) {
            const Point direction = random_unit_direction<Point>(worker.rng, dimension);
            const auto hit = raycast(x0, direction, neighbour_points);
            if (!hit.finite) {
                continue;
            }

            const Scalar t = hit.t;
            volume_accumulator += std::pow(t, dimension);

            if constexpr (!detail::IsNoMonteCarloFunction<BulkFunction>::value) {
                if (store_bulk) {
                    std::uniform_real_distribution<Scalar> radius_distribution(
                        Scalar{0}, Scalar{1});
                    for (std::size_t r_sample = 0;
                         r_sample < options_.bulk_samples_per_ray;
                         ++r_sample) {
                        const Scalar radius = t * radius_distribution(worker.rng);
                        evaluation_point = x0 + direction * radius;
                        detail::wrap_integration_evaluation_point(
                            update.integration_view(),
                            evaluation_point);
                        const Scalar weight =
                            std::pow(radius, dimension - 1) * t;
                        detail::accumulate_function_value(
                            bulk_, evaluation_point, weight, bulk);
                    }
                }
            }

            if (store_area && update.interface_requires_recompute(hit.ordinal)) {
                const Scalar denominator =
                    std::abs(normals[hit.ordinal].dot(direction));
                if (!(denominator > Scalar{0})) {
                    continue;
                }
                const Scalar d_area = std::pow(t, dimension - 1) / denominator;
                areas[hit.ordinal] += d_area;

                if constexpr (!detail::IsNoMonteCarloFunction<InterfaceFunction>::value) {
                    if (store_interface) {
                        const std::size_t components =
                            update.integration_view().integral().data().integral_components();
                        auto begin = interfaces.begin() +
                            static_cast<std::ptrdiff_t>(hit.ordinal * components);
                        std::vector<Scalar> local(
                            begin,
                            begin + static_cast<std::ptrdiff_t>(components));
                        evaluation_point = x0 + direction * t;
                        detail::wrap_integration_evaluation_point(
                            update.integration_view(),
                            evaluation_point);
                        detail::accumulate_function_value(
                            interface_, evaluation_point, d_area, local);
                        std::copy(local.begin(), local.end(), begin);
                    }
                }
            }
        }

        const Scalar unit_ball_volume =
            std::pow(pi<Scalar>(), Scalar(dimension) / Scalar{2}) /
            std::tgamma(Scalar{1} + Scalar(dimension) / Scalar{2});
        const Scalar surface_factor = Scalar(dimension) * unit_ball_volume;
        const Scalar inv_rays = Scalar{1} /
            static_cast<Scalar>(options_.interface_rays);

        cell.set_volume(static_cast<std::decay_t<decltype(cell.volume())>>(
            volume_accumulator * unit_ball_volume * inv_rays));

        if (store_area) {
            for (auto& area : areas) {
                area = static_cast<std::decay_t<decltype(cell.volume())>>(
                    area * surface_factor * inv_rays);
            }
        }
        if constexpr (!detail::IsNoMonteCarloFunction<BulkFunction>::value) {
            if (store_bulk) {
                if (options_.bulk_samples_per_ray == 0) {
                    throw std::invalid_argument(
                        "bulk Monte-Carlo integration requires at least one bulk sample per ray");
                }
                const Scalar factor = surface_factor * inv_rays /
                    static_cast<Scalar>(options_.bulk_samples_per_ray);
                for (auto& value : bulk) {
                    value = static_cast<std::decay_t<decltype(bulk.front())>>(value * factor);
                }
            }
        }
        if constexpr (!detail::IsNoMonteCarloFunction<InterfaceFunction>::value) {
        if (store_interface) {
            const Scalar factor = surface_factor * inv_rays;
            for (auto& value : interfaces) {
                value = static_cast<std::decay_t<decltype(interfaces.front())>>(value * factor);
            }
        }
        }
    }

    template <class Pass>
    void cleanup_cell(Pass& pass, std::size_t position) const {
        auto& update = pass.writable_cell(position);
        auto& cell = update.data();
        auto& areas = cell.area();
        auto& interfaces = cell.interface_integral();
        const auto& neighbours = cell.neighbours();

        if (areas.empty()) {
            return;
        }

        // Integrator::prepare_cleanup() has re-activated this cell's extended
        // nodes before entering the algorithm-specific cleanup phase.
        auto& extended_nodes = update.extended_nodes();
        auto x0 = extended_nodes.node(update.cell());
        using Scalar = typename std::decay_t<decltype(x0)>::Scalar;
        const Scalar dimension = static_cast<Scalar>(x0.size());
        const std::size_t components =
            update.integration_view().integral().data().integral_components();

        for (std::size_t ordinal = 0; ordinal < neighbours.size(); ++ordinal) {
            const auto reciprocal =
                pass.view().reciprocal_update_interface(
                    pass,
                    position,
                    ordinal);
            if (!reciprocal || reciprocal->cell > position) {
                continue;
            }
            if (reciprocal->cell == position &&
                reciprocal->ordinal >= ordinal) {
                continue;
            }

            const std::size_t other_position = reciprocal->cell;
            auto& other_update = pass.writable_cell(other_position);
            auto& other_cell = other_update.data();
            if (other_cell.area().empty()) {
                continue;
            }
            const std::size_t other_ordinal = reciprocal->ordinal;
            if (other_ordinal >= other_cell.area().size()) {
                throw std::logic_error(
                    "Monte-Carlo cleanup reciprocal area ordinal is out of range");
            }

            const Scalar old_area = static_cast<Scalar>(areas[ordinal]);
            const Scalar neighbour_area =
                static_cast<Scalar>(other_cell.area()[other_ordinal]);
            if (!(neighbour_area > Scalar{0}) && !(old_area > Scalar{0})) {
                continue;
            }

            // Julia mcintegrator.jl: if the already-computed opposite estimate
            // is numerically absent, keep this side; otherwise average the two.
            const Scalar new_area =
                std::abs(neighbour_area) <
                        std::abs(old_area) * Scalar{1e-10}
                    ? old_area
                    : Scalar{0.5} * (old_area + neighbour_area);

            const auto xn = extended_nodes.node(neighbours[ordinal]);
            const Scalar factor = Scalar{0.5} * (xn - x0).norm() / dimension;
            adjust_volume(cell, new_area - old_area, factor);
            adjust_volume(other_cell, new_area - neighbour_area, factor);
            areas[ordinal] = static_cast<std::decay_t<decltype(areas[ordinal])>>(new_area);
            other_cell.area()[other_ordinal] =
                static_cast<std::decay_t<decltype(other_cell.area()[other_ordinal])>>(new_area);

            if (!interfaces.empty() &&
                !other_cell.interface_integral().empty() &&
                components != 0 &&
                !options_.heuristic) {
                const std::size_t begin = ordinal * components;
                const std::size_t other_begin = other_ordinal * components;
                if (begin + components > interfaces.size() ||
                    other_begin + components > other_cell.interface_integral().size()) {
                    throw std::logic_error(
                        "Monte-Carlo cleanup found misaligned interface integral data");
                }

                for (std::size_t component = 0; component < components; ++component) {
                    const auto averaged = static_cast<std::decay_t<decltype(interfaces[begin + component])>>(
                        Scalar{0.5} *
                        (static_cast<Scalar>(interfaces[begin + component]) +
                         static_cast<Scalar>(other_cell.interface_integral()[other_begin + component])));
                    interfaces[begin + component] = averaged;
                    other_cell.interface_integral()[other_begin + component] = averaged;
                }

            }
        }
    }

private:
    [[nodiscard]] constexpr bool has_bulk_function() const noexcept {
        return !detail::IsNoMonteCarloFunction<BulkFunction>::value;
    }

    [[nodiscard]] constexpr bool has_interface_function() const noexcept {
        return !detail::IsNoMonteCarloFunction<InterfaceFunction>::value;
    }

    template <typename Scalar>
    [[nodiscard]] static Scalar pi() noexcept {
        return std::acos(Scalar{-1});
    }

    template <class Point, class RNG>
    [[nodiscard]] static Point random_unit_direction(RNG& rng, int dimension) {
        using Scalar = typename Point::Scalar;
        std::normal_distribution<Scalar> normal(Scalar{0}, Scalar{1});
        Point direction;
        if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
            direction.resize(dimension);
        }
        Scalar norm2 = Scalar{0};
        do {
            norm2 = Scalar{0};
            for (int coordinate = 0; coordinate < dimension; ++coordinate) {
                direction[coordinate] = normal(rng);
                norm2 += direction[coordinate] * direction[coordinate];
            }
        } while (!(norm2 > Scalar{0}));
        direction /= std::sqrt(norm2);
        return direction;
    }

    template <class Point>
    struct RayHit {
        bool finite = false;
        std::size_t ordinal = 0;
        typename Point::Scalar t = std::numeric_limits<typename Point::Scalar>::infinity();
    };

    template <class Point, class NeighbourPoints>
    [[nodiscard]] static RayHit<Point> raycast(
        const Point& x0,
        const Point& direction,
        const NeighbourPoints& neighbour_points) {
        using Scalar = typename Point::Scalar;
        RayHit<Point> result;
        Scalar best_t = std::numeric_limits<Scalar>::infinity();

        const Scalar center_projection = x0.dot(direction);
        for (std::size_t ordinal = 0; ordinal < neighbour_points.size(); ++ordinal) {
            const Point& xn = neighbour_points[ordinal];
            if (xn.dot(direction) <= center_projection) {
                continue;
            }
            const Scalar denominator = Scalar{2} * direction.dot(xn - x0);
            if (!(denominator > Scalar{0})) {
                continue;
            }
            const Scalar numerator =
                (x0 - xn).squaredNorm();
            const Scalar t = numerator / denominator;
            if (Scalar{0} < t && t < best_t) {
                best_t = t;
                result.ordinal = ordinal;
                result.finite = true;
            }
        }
        result.t = best_t;
        return result;
    }

    template <class CellData, class Scalar>
    static void adjust_volume(CellData& cell, Scalar delta_area, Scalar factor) {
        using AreaScalar = std::decay_t<decltype(cell.volume())>;
        cell.set_volume(static_cast<AreaScalar>(
            static_cast<Scalar>(cell.volume()) + delta_area * factor));
    }

    MonteCarloOptions options_{};
    BulkFunction bulk_{};
    InterfaceFunction interface_{};
};

} // namespace highvoronoi
