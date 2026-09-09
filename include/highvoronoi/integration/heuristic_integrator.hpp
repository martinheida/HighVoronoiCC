#pragma once

/**
 * @file heuristic_integrator.hpp
 * @brief Geometry-consuming heuristic function integration.
 *
 * The Julia Heuristic_Integrator assumes that cell volumes and interface areas
 * already exist and only constructs function integrals from cell/interface
 * vertices.  In the C++ architecture the geometry source is explicit and is
 * kept separate from the target integral.  This prevents target preparation
 * from destroying the geometry values it is about to consume.
 *
 * A source algorithm is bound together with the source integral.  With
 * force_source_update=false the source must already be current and complete.
 * With force_source_update=true any inconsistent source cells are marked dirty,
 * updated through that algorithm, and validated again before the target view is
 * created.  The target VoronoiIntegral owns its normal independent mesh dirty
 * tracker, so refine/remove dirtiness is not hidden by updating the source.
 */

#include <highvoronoi/integration/integrator.hpp>
#include <highvoronoi/integration/integration_view.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

struct HeuristicOptions {
    /**
     * If false, stale/incomplete source geometry is an error.
     * If true, stale/incomplete source cells are recomputed through the bound
     * source algorithm before heuristic integration starts.
     */
    bool force_source_update = false;
};

namespace detail {

template <class T, class = void>
struct HeuristicIndexedResult : std::false_type {};

template <class T>
struct HeuristicIndexedResult<
    T,
    std::void_t<
        decltype(std::declval<const T&>().size()),
        decltype(std::declval<const T&>()[std::declval<std::size_t>()])>>
    : std::true_type {};

template <class Function, class Point, class Scalar>
void evaluate_heuristic_function(
    const Function& function,
    const Point& point,
    std::vector<Scalar>& output) {
    const auto value = function(point);
    using Value = std::decay_t<decltype(value)>;

    if constexpr (std::is_arithmetic_v<Value>) {
        if (output.size() != std::size_t{1}) {
            throw std::logic_error(
                "Scalar HeuristicAlgorithm integrand requires one component");
        }
        output[0] = static_cast<Scalar>(value);
    } else if constexpr (HeuristicIndexedResult<Value>::value) {
        if (static_cast<std::size_t>(value.size()) != output.size()) {
            throw std::logic_error(
                "HeuristicAlgorithm integrand component count mismatch");
        }
        for (std::size_t component = 0;
             component < output.size();
             ++component) {
            output[component] =
                static_cast<Scalar>(value[component]);
        }
    } else {
        static_assert(
            std::is_arithmetic_v<Value> ||
                HeuristicIndexedResult<Value>::value,
            "HeuristicAlgorithm integrand must return a scalar or indexable vector");
    }
}

template <class Point>
[[nodiscard]] Point make_heuristic_point(std::size_t dimension) {
    if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
        return Point(static_cast<Eigen::Index>(dimension));
    } else {
        if (dimension != static_cast<std::size_t>(Point::RowsAtCompileTime)) {
            throw std::invalid_argument(
                "Heuristic point dimension does not match fixed Point type");
        }
        return Point{};
    }
}

/**
 * Recycled implementation of Julia's _heuristic_integral formula.
 *
 * The old Julia code only populated all vertices on the owning side of a shared
 * interface and copied the opposite value during cleanup.  The C++ source/target
 * split has no need for that placeholder representation: when an interface must
 * be recomputed, this evaluator uses the complete current vertex set directly.
 * The quadrature formula itself is unchanged.
 */
template <class Integral, class Function>
class HeuristicCellEvaluator final {
public:
    using Mesh = typename Integral::Mesh;
    using Index = typename Integral::Index;
    using Point = typename Mesh::NodePoint;
    using IntegralScalar = typename Integral::IntegralScalar;
    using AreaScalar = typename Integral::AreaScalar;

    HeuristicCellEvaluator(
        Integral& target,
        const Function& function)
        : target_(&target),
          function_(&function),
          dimension_(static_cast<std::size_t>(target.mesh().dimension())),
          components_(target.data().integral_components()),
          center_(make_heuristic_point<Point>(dimension_)),
          cell_point_(make_heuristic_point<Point>(dimension_)),
          evaluation_point_(make_heuristic_point<Point>(dimension_)),
          function_value_(components_),
          cell_value_(components_),
          interface_value_(components_) {
        if (dimension_ == 0) {
            throw std::invalid_argument(
                "HeuristicAlgorithm requires positive dimension");
        }
        if (components_ == 0) {
            throw std::invalid_argument(
                "HeuristicAlgorithm requires at least one integral component");
        }
    }

    /**
     * Evaluate one prepared target cell using caller-supplied geometry values
     * already aligned with the target's presented neighbour order.
     */
    template <class Update, class AreaVector>
    void integrate_cell(
        Update& update,
        AreaScalar geometry_volume,
        const AreaVector& geometry_areas) {
        require_target_layout(update, geometry_areas.size());

        auto& cell = update.data();
        const auto& neighbours = update.neighbours();
        auto& target_areas = cell.area();
        auto& bulk = cell.bulk_integral();
        auto& interfaces = cell.interface_integral();

        if (!target_areas.empty()) {
            for (std::size_t ordinal = 0;
                 ordinal < geometry_areas.size();
                 ++ordinal) {
                target_areas[ordinal] = static_cast<AreaScalar>(
                    geometry_areas[ordinal]);
            }
        }
        if (target_->data().stores_volume()) {
            cell.set_volume(geometry_volume);
        }

        std::fill(bulk.begin(), bulk.end(), IntegralScalar{0});
        prepare_interface_vertices(update);

        auto& extended_nodes = update.extended_nodes();
        copy_point(extended_nodes.node(update.cell()), cell_point_);
        evaluate_for_update(update, cell_point_, cell_value_);

        const IntegralScalar dim =
            static_cast<IntegralScalar>(dimension_);
        const IntegralScalar dim_plus_one =
            static_cast<IntegralScalar>(dimension_ + 1);

        for (std::size_t ordinal = 0;
             ordinal < neighbours.size();
             ++ordinal) {
            const AreaScalar area = static_cast<AreaScalar>(
                geometry_areas[ordinal]);

            const bool can_reuse_interface =
                !interfaces.empty() &&
                !update.interface_requires_recompute(ordinal);

            if (can_reuse_interface) {
                copy_interface_from_target(interfaces, ordinal);
            } else {
                calculate_interface_value(update, ordinal, area);
                if (!interfaces.empty()) {
                    copy_interface_to_target(interfaces, ordinal);
                }
            }

            if (!bulk.empty()) {
                const auto neighbour_point =
                    extended_nodes.node(neighbours[ordinal]);
                const IntegralScalar distance =
                    IntegralScalar{0.5} *
                    static_cast<IntegralScalar>(
                        (cell_point_ - neighbour_point).norm());
                const IntegralScalar cone_factor = distance / dim;

                for (std::size_t component = 0;
                     component < components_;
                     ++component) {
                    // Julia:
                    // _y  = f(cell) * area/(d+1)
                    // _y += AREA_Int * d/(d+1)
                    // _y *= distance/d
                    const IntegralScalar inside =
                        static_cast<IntegralScalar>(area) *
                            cell_value_[component] / dim_plus_one +
                        interface_value_[component] *
                            dim / dim_plus_one;
                    bulk[component] += cone_factor * inside;
                }
            }
        }
    }

private:
    template <class Update>
    void require_target_layout(
        const Update& update,
        std::size_t geometry_area_count) const {
        const auto& data = target_->data();
        const std::size_t neighbour_count = update.neighbours().size();

        if (geometry_area_count != neighbour_count) {
            throw std::logic_error(
                "Heuristic geometry area count does not match target neighbours");
        }
        if (!data.stores_bulk_integral() &&
            !data.stores_interface_integral()) {
            throw std::logic_error(
                "Heuristic target stores no function integral payload");
        }
        if (data.stores_area() &&
            update.data().area().size() != neighbour_count) {
            throw std::logic_error(
                "Heuristic target area storage is not neighbour-aligned");
        }
        if (data.stores_bulk_integral() &&
            update.data().bulk_integral().size() != components_) {
            throw std::logic_error(
                "Heuristic target bulk-integral layout mismatch");
        }
        if (data.stores_interface_integral() &&
            update.data().interface_integral().size() !=
                neighbour_count * components_) {
            throw std::logic_error(
                "Heuristic target interface-integral layout mismatch");
        }
    }

    template <class Update>
    void prepare_interface_vertices(Update& update) {
        const auto& neighbours = update.neighbours();
        if (interface_vertices_.size() < neighbours.size()) {
            interface_vertices_.resize(neighbours.size());
        }
        for (std::size_t ordinal = 0;
             ordinal < interface_vertices_.size();
             ++ordinal) {
            interface_vertices_[ordinal].clear();
        }

        for (const auto& record : update.vertices()) {
            for (const Index generator : record.sigma) {
                if (generator == update.cell()) {
                    continue;
                }
                const auto range = std::equal_range(
                    neighbours.begin(),
                    neighbours.end(),
                    generator);
                if (range.first == range.second) {
                    continue;
                }

                // Ordinary integration currently has one occurrence per
                // geometric generator. If a future presentation exposes
                // duplicate equal indices, every occurrence represents the
                // same projected generator and receives the same vertex.
                for (auto it = range.first; it != range.second; ++it) {
                    const std::size_t ordinal = static_cast<std::size_t>(
                        std::distance(neighbours.begin(), it));
                    Point point = make_heuristic_point<Point>(dimension_);
                    copy_point(record.position, point);
                    interface_vertices_[ordinal].push_back(
                        std::move(point));
                }
            }
        }
    }

    template <class Update>
    void calculate_interface_value(
        Update& update,
        std::size_t ordinal,
        AreaScalar area) {
        const auto& vertices = interface_vertices_.at(ordinal);
        if (vertices.empty()) {
            throw std::logic_error(
                "Heuristic interface has no finite vertices");
        }

        center_.setZero();
        std::fill(
            interface_value_.begin(),
            interface_value_.end(),
            IntegralScalar{0});

        for (const Point& vertex : vertices) {
            center_ += vertex;
            evaluate_for_update(update, vertex, function_value_);
            for (std::size_t component = 0;
                 component < components_;
                 ++component) {
                interface_value_[component] +=
                    function_value_[component];
            }
        }

        const IntegralScalar count =
            static_cast<IntegralScalar>(vertices.size());
        center_ /= static_cast<typename Point::Scalar>(count);
        evaluate_for_update(update, center_, function_value_);

        const IntegralScalar dim =
            static_cast<IntegralScalar>(dimension_);
        const IntegralScalar vertex_weight =
            (dim - IntegralScalar{1}) / (dim * count);
        const IntegralScalar center_weight = IntegralScalar{1} / dim;
        const IntegralScalar measure = static_cast<IntegralScalar>(area);

        for (std::size_t component = 0;
             component < components_;
             ++component) {
            interface_value_[component] = measure *
                (vertex_weight * interface_value_[component] +
                 center_weight * function_value_[component]);
        }
    }

    void copy_interface_from_target(
        const std::vector<IntegralScalar>& values,
        std::size_t ordinal) {
        const std::size_t begin = ordinal * components_;
        std::copy_n(
            values.begin() + static_cast<std::ptrdiff_t>(begin),
            components_,
            interface_value_.begin());
    }

    void copy_interface_to_target(
        std::vector<IntegralScalar>& values,
        std::size_t ordinal) const {
        const std::size_t begin = ordinal * components_;
        std::copy_n(
            interface_value_.begin(),
            components_,
            values.begin() + static_cast<std::ptrdiff_t>(begin));
    }

    template <class SourcePoint>
    void copy_point(const SourcePoint& source, Point& target) const {
        if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
            target.resize(static_cast<Eigen::Index>(dimension_));
        }
        for (std::size_t coordinate = 0;
             coordinate < dimension_;
             ++coordinate) {
            target[static_cast<Eigen::Index>(coordinate)] =
                static_cast<typename Point::Scalar>(
                    source[static_cast<Eigen::Index>(coordinate)]);
        }
    }

    template <class Update>
    void evaluate_for_update(
        Update& update,
        const Point& point,
        std::vector<IntegralScalar>& output) {
        evaluation_point_ = point;
        detail::wrap_integration_evaluation_point(
            update.integration_view(),
            evaluation_point_);
        evaluate_heuristic_function(
            *function_,
            evaluation_point_,
            output);
    }

    Integral* target_ = nullptr;
    const Function* function_ = nullptr;
    std::size_t dimension_ = 0;
    std::size_t components_ = 0;

    std::vector<std::vector<Point>> interface_vertices_;
    Point center_;
    Point cell_point_;
    Point evaluation_point_;
    std::vector<IntegralScalar> function_value_;
    std::vector<IntegralScalar> cell_value_;
    std::vector<IntegralScalar> interface_value_;
};

} // namespace detail

/**
 * @brief Heuristic function integrator consuming geometry from another integral.
 *
 * Source and target must be distinct VoronoiIntegral objects attached to the
 * exact same mesh. Source volume/area records are treated as authoritative once
 * currentness validation succeeds.
 */
template <class TargetIntegralT,
          class SourceIntegralT,
          class SourceAlgorithmT,
          class FunctionT>
class HeuristicAlgorithm final {
public:
    using TargetIntegral = TargetIntegralT;
    using SourceIntegral = SourceIntegralT;
    using SourceAlgorithm = SourceAlgorithmT;
    using Function = FunctionT;
    using Index = typename SourceIntegral::Index;
    using SourceCellData = typename SourceIntegral::Data::CellData;
    using AreaScalar = typename TargetIntegral::AreaScalar;
    using Evaluator = detail::HeuristicCellEvaluator<TargetIntegral, Function>;

    HeuristicAlgorithm(
        TargetIntegral& target,
        SourceIntegral& source,
        SourceAlgorithm& source_algorithm,
        Function function,
        HeuristicOptions options = {})
        : target_(&target),
          source_(&source),
          source_algorithm_(&source_algorithm),
          function_(std::move(function)),
          options_(options),
          evaluator_(target, function_) {
        if (static_cast<const void*>(target_) ==
            static_cast<const void*>(source_)) {
            throw std::invalid_argument(
                "Heuristic source and target integral must be distinct");
        }
    }

    void prepare_integration(TargetIntegral& target) {
        prepare_integration_impl(
            target,
            SerialIntegrationExecution{});
    }

    /**
     * Execution-aware preflight used by the generic integration driver.
     *
     * When force_source_update is active, a parallel-capable source algorithm
     * inherits the target execution policy. A source algorithm without worker
     * support falls back to its serial driver rather than disabling parallelism
     * of the heuristic target itself.
     */
    template <class Execution>
    void prepare_integration(
        TargetIntegral& target,
        Execution execution) {
        prepare_integration_impl(target, execution);
    }

    template <class Update>
    void integrate_cell(Update& update) {
        integrate_cell_with_scratch(
            update,
            source_cell_,
            aligned_areas_,
            evaluator_);
    }

    /** Worker-local heuristic scratch over one immutable geometry source. */
    class ParallelWorker final {
    public:
        explicit ParallelWorker(const HeuristicAlgorithm& parent)
            : parent_(&parent),
              evaluator_(*parent.target_, parent.function_) {}

        template <class Update>
        void integrate_cell(Update& update) {
            parent_->integrate_cell_with_scratch(
                update,
                source_cell_,
                aligned_areas_,
                evaluator_);
        }

    private:
        const HeuristicAlgorithm* parent_ = nullptr;
        Evaluator evaluator_;
        SourceCellData source_cell_;
        std::vector<AreaScalar> aligned_areas_;
    };

    [[nodiscard]] ParallelWorker make_worker_algorithm(std::size_t) const {
        return ParallelWorker(*this);
    }

    [[nodiscard]] const HeuristicOptions& options() const noexcept {
        return options_;
    }

    [[nodiscard]] const SourceIntegral& source_integral() const noexcept {
        return *source_;
    }

private:
    template <class Execution>
    void prepare_integration_impl(
        TargetIntegral& target,
        Execution execution) {
        if (&target != target_) {
            throw std::logic_error(
                "HeuristicAlgorithm used with a different target integral");
        }
        require_compatible_integrals();

        source_->synchronize_size();
        target_->synchronize_size();
        ensure_tracker_sizes();

        collect_inconsistent_source_cells();
        if (source_inconsistent_.empty()) {
            return;
        }

        if (!options_.force_source_update) {
            throw std::logic_error(
                "Heuristic geometry source is stale or incomplete; "
                "set force_source_update=true or update the source integral first");
        }

        // Force exactly the source cells that failed validation into the source
        // update prefix. Mark the same stable cells dirty in the target so a
        // changed source geometry can never be hidden from heuristic output.
        for (const Index stable : source_inconsistent_) {
            source_->dirty_tracker()->set_dirty(
                static_cast<std::size_t>(stable), true);
            target_->dirty_tracker()->set_dirty(
                static_cast<std::size_t>(stable), true);
        }

        update_source(execution);

        collect_inconsistent_source_cells();
        if (!source_inconsistent_.empty()) {
            throw std::logic_error(
                "Heuristic geometry source remains stale/incomplete after forced update");
        }
    }

    template <class Execution>
    void update_source(Execution execution) {
        if constexpr (detail::HasMakeWorkerAlgorithm<SourceAlgorithm>::value) {
            // Reuse the caller's execution policy. With serial execution this
            // still produces one worker; with parallel execution the source
            // gets the same requested worker count as the heuristic target.
            (void)highvoronoi::integrate(
                *source_,
                *source_algorithm_,
                execution);
        } else {
            // Source validity is independent of target execution. A source
            // provider that intentionally exposes no worker state remains a
            // valid serial provider for a parallel heuristic target.
            (void)highvoronoi::integrate(
                *source_,
                *source_algorithm_);
        }
    }

    template <class Update>
    void integrate_cell_with_scratch(
        Update& update,
        SourceCellData& source_cell,
        std::vector<AreaScalar>& aligned_areas,
        Evaluator& evaluator) const {
        if (!source_->data().read_geometry_cell(
                update.stable_cell(),
                source_cell)) {
            throw std::logic_error(
                "Heuristic could not read complete source geometry for target cell");
        }

        if (source_cell.neighbours() != update.new_internal_neighbours()) {
            throw std::logic_error(
                "Heuristic source neighbour topology changed after source validation");
        }

        const auto& source_areas = source_cell.area();
        aligned_areas.resize(update.neighbours().size());
        for (std::size_t ordinal = 0;
             ordinal < update.neighbours().size();
             ++ordinal) {
            const std::size_t raw_ordinal =
                update.raw_neighbour_ordinal(ordinal);
            if (raw_ordinal >= source_areas.size()) {
                throw std::logic_error(
                    "Heuristic source area ordinal is outside source geometry record");
            }
            aligned_areas[ordinal] = static_cast<AreaScalar>(
                source_areas[raw_ordinal]);
        }

        evaluator.integrate_cell(
            update,
            static_cast<AreaScalar>(source_cell.volume()),
            aligned_areas);
    }

    void require_compatible_integrals() const {
        if (static_cast<const void*>(&target_->mesh()) !=
            static_cast<const void*>(&source_->mesh())) {
            throw std::invalid_argument(
                "Heuristic source and target must reference the same mesh object");
        }
        if (!source_->data().stores_volume() ||
            !source_->data().stores_area()) {
            throw std::invalid_argument(
                "Heuristic geometry source must store volume and area");
        }
        if (!target_->data().stores_bulk_integral() &&
            !target_->data().stores_interface_integral()) {
            throw std::invalid_argument(
                "Heuristic target must store bulk and/or interface integrals");
        }
    }

    void ensure_tracker_sizes() {
        const std::size_t target_size = target_->stable_size();
        const std::size_t source_size = source_->stable_size();
        if (source_->dirty_tracker()->size() != source_size) {
            source_->dirty_tracker()->resize(source_size);
        }
        if (target_->dirty_tracker()->size() != target_size) {
            target_->dirty_tracker()->resize(target_size);
        }
    }

    void collect_inconsistent_source_cells() {
        source_inconsistent_.clear();
        auto& mesh = source_->mesh();

        for (std::size_t public_position = 0;
             public_position < static_cast<std::size_t>(mesh.size());
             ++public_position) {
            const Index public_cell = static_cast<Index>(public_position);
            const Index stable =
                mesh.index_mapping().public_to_internal(public_cell);
            const std::size_t stable_position =
                static_cast<std::size_t>(stable);

            bool inconsistent = false;
            const auto source_address =
                source_->data().neighbour_address(stable);
            const auto current_address =
                mesh.neighbour_address(public_cell);

            inconsistent = inconsistent ||
                source_->dirty_tracker()->dirty(stable_position);
            inconsistent = inconsistent || mesh.dirty(public_cell);
            inconsistent = inconsistent || source_address == 0;
            inconsistent = inconsistent || current_address == 0;
            inconsistent = inconsistent ||
                (source_address != 0 &&
                 current_address != 0 &&
                 source_address != current_address);

            if (!inconsistent &&
                !source_->data().read_geometry_cell(stable, validation_cell_)) {
                inconsistent = true;
            }

            if (inconsistent) {
                source_inconsistent_.push_back(stable);
            }
        }
    }

    TargetIntegral* target_ = nullptr;
    SourceIntegral* source_ = nullptr;
    SourceAlgorithm* source_algorithm_ = nullptr;
    Function function_;
    HeuristicOptions options_{};
    Evaluator evaluator_;

    SourceCellData source_cell_;
    SourceCellData validation_cell_;
    std::vector<Index> source_inconsistent_;
    std::vector<AreaScalar> aligned_areas_;
};

template <class TargetIntegral,
          class SourceIntegral,
          class SourceAlgorithm,
          class Function>
[[nodiscard]] auto make_heuristic_algorithm(
    TargetIntegral& target,
    SourceIntegral& source,
    SourceAlgorithm& source_algorithm,
    Function&& function,
    HeuristicOptions options = {}) {
    using StoredFunction = std::decay_t<Function>;
    return HeuristicAlgorithm<
        TargetIntegral,
        SourceIntegral,
        SourceAlgorithm,
        StoredFunction>(
            target,
            source,
            source_algorithm,
            std::forward<Function>(function),
            options);
}

} // namespace highvoronoi
