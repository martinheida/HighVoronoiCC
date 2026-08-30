#pragma once

/**
 * @file raycaster.hpp
 * @brief CRTP ray-casting algorithms used by HighVoronoi mesh construction.
 *
 * A RayCaster owns one safe copy of a SearchTree. The SearchTree remains tied
 * to the Mesh from which it was built, but its safe copy shares only immutable
 * or otherwise safely shareable backend state (for example the nanoflann KD
 * index) and owns independent ExtendedVoronoiNodes. Consequently each
 * RayCaster can activate one cell exactly once and then be used independently
 * by WalkRay/Descent and, later, by one worker thread.
 *
 * Two concrete algorithms are provided:
 *
 * - ClassicRayCaster: iterative nearest-neighbour ray cast for vertices in
 *   general position, corresponding to the Julia Raycast_Original path.
 * - InRangeRayCaster: Julia's Raycast_Non_General_HP / HPUnion path for
 *   potentially degenerate endpoint vertices.
 *
 * The public cast interface is deliberately identical for both classes. The
 * old Julia arguments `xs` and `searcher` are owned by the RayCaster/SearchTree,
 * the unused `old` and development-only `debug` arguments are removed, and the
 * Julia RCType is represented by the concrete C++ RayCaster class. The one
 * remaining runtime distinction, Raycast_By_Descend versus Raycast_By_Walkray,
 * is represented by RayCastUsage.
 *
 * Vertex correction reuses QRNormalSolver and falls back to
 * ExtendedQRNormalSolver only when the ordinary correction is not accurate
 * enough.
 */

#include <highvoronoi/geometry/normal_solver.hpp>
#include <highvoronoi/geometry/point.hpp>
#include <highvoronoi/parameters.hpp>

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/** @brief Calling context of one ray cast. */
enum class RayCastUsage {
    Descent,
    WalkRay
};

/** @brief Outcome of a ray cast. */
enum class RayCastStatus {
    Finite,
    Infinite
};

template <class PointT, typename IndexT, typename ScalarT>
struct RayCastResult {
    using Point = PointT;
    using Index = IndexT;
    using Scalar = ScalarT;

    RayCastStatus status = RayCastStatus::Infinite;
    Index generator{};
    Scalar t = infinity();
    Point position{};
    bool corrected = false;

private:
    [[nodiscard]] static constexpr Scalar infinity() noexcept {
        if constexpr (std::numeric_limits<Scalar>::has_infinity) {
            return std::numeric_limits<Scalar>::infinity();
        } else {
            return std::numeric_limits<Scalar>::max();
        }
    }
};

namespace detail {

template <typename Scalar, int Dim>
using RayPoint = std::conditional_t<
    Dim == Dynamic,
    DynamicPoint<Scalar>,
    StaticPoint<Scalar, Dim>>;

template <typename Scalar>
using DynamicRayMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <class Container, class Index>
[[nodiscard]] bool contains_index(
    const Container& container,
    Index index) {
    return std::find(container.begin(), container.end(), index) !=
           container.end();
}

template <class Container>
void sort_unique(Container& container) {
    std::sort(container.begin(), container.end());
    container.erase(
        std::unique(container.begin(), container.end()),
        container.end());
}

} // namespace detail

/**
 * @brief Shared static-polymorphic infrastructure for concrete RayCasters.
 *
 * Derived must implement
 *
 * @code{.cpp}
 * template<class Sigma, class Edge, class Origin>
 * Result cast_impl(Sigma&, const Point&, const Point&, const Edge&,
 *                  const Origin&, RayCastUsage, Scalar);
 *
 * Derived safe_copy_impl() const;
 * @endcode
 */
template <class DerivedT,
          class SearchTreeT,
          class ParametersT>
class RayCasterBase {
public:
    using Derived = DerivedT;
    using SearchTree = SearchTreeT;
    using Parameters = ParametersT;
    using Mesh = typename SearchTree::Mesh;
    using Scalar = typename SearchTree::Scalar;
    using Index = typename SearchTree::MeshIndex;
    using ExtendedNodes = typename SearchTree::ExtendedNodes;
    using SearchData = typename SearchTree::SearchData;

    static constexpr int DimensionAtCompileTime =
        SearchTree::DimensionAtCompileTime;

    using Point = detail::RayPoint<
        Scalar,
        DimensionAtCompileTime>;
    using Result = RayCastResult<Point, Index, Scalar>;

    RayCasterBase(const RayCasterBase&) = delete;
    RayCasterBase& operator=(const RayCasterBase&) = delete;
    RayCasterBase(RayCasterBase&&) = delete;
    RayCasterBase& operator=(RayCasterBase&&) = delete;

    /**
     * @brief Activate the independent ExtendedNodes owned by this RayCaster.
     *
     * Call once at the beginning of a cell search. No ray-cast operation
     * changes the active cell afterwards.
     */
    template <class NeighbourContainer>
    void activate_cell(
        Index cell,
        const NeighbourContainer& neighbours) {
        tree_.extended_nodes().activate_cell(cell, neighbours);
    }

    [[nodiscard]] SearchTree& tree() noexcept {
        return tree_;
    }

    [[nodiscard]] const SearchTree& tree() const noexcept {
        return tree_;
    }

    [[nodiscard]] ExtendedNodes& extended_nodes() noexcept {
        return tree_.extended_nodes();
    }

    [[nodiscard]] const ExtendedNodes& extended_nodes() const noexcept {
        return tree_.extended_nodes();
    }

    [[nodiscard]] const Parameters& parameters() const noexcept {
        return parameters_;
    }

    /**
     * @brief Cast one edge with the concrete algorithm.
     *
     * @param sigma Mutable current/full edge signature. Concrete algorithms may
     *        append newly detected endpoint generators, exactly as in Julia.
     * @param start Old vertex or general starting point r.
     * @param direction Edge direction u. It must be normalized.
     * @param edge Minimal supporting edge. Its first entry supplies x0; for
     *        correction it must contain at least d generators.
     * @param origin Complete signature of the vertex from which the ray leaves.
     *        The non-general algorithm uses it to reject zero-length returns.
     * @param usage Descent or WalkRay context.
     * @param direction_error Estimated error du of the edge direction.
     */
    template <class Sigma, class Edge, class Origin>
    [[nodiscard]] Result cast(
        Sigma& sigma,
        const Point& start,
        const Point& direction,
        const Edge& edge,
        const Origin& origin,
        RayCastUsage usage,
        Scalar direction_error = Scalar{0}) {
        validate_cast_input(sigma, start, direction, edge);
        return derived().cast_impl(
            sigma,
            start,
            direction,
            edge,
            origin,
            usage,
            direction_error);
    }

    /** @brief Create a worker-safe clone with fresh local mutable state. */
    [[nodiscard]] Derived safe_copy() const {
        return derived().safe_copy_impl();
    }

    /**
     * @brief Improve the RayCaster candidate instead of recomputing the vertex.
     *
     * The ordinary path factorizes the row-normalized local equal-distance
     * system once with ColPivHouseholderQR and iterates
     *
     *   A delta = c(r),  r <- r + delta.
     *
     * If the pivot ratio indicates poor conditioning, or if the relative
     * correction |delta| / |r_initial-p_0| does not converge to the requested
     * tolerance, the complete correction is repeated directly in Float128.
     * There is intentionally no platform-dependent intermediate precision.
     */
    template <class Support>
    [[nodiscard]] bool correct_vertex(
        const Support& support,
        Point& output) {
        if (static_cast<std::size_t>(support.size()) != dimension_size() + 1) {
            return false;
        }

        const ExtendedNodes& nodes = tree_.extended_nodes();
        const Point initial = output;

        const auto ordinary =
            vertex_solver_.correct_vertex(
                nodes,
                support,
                initial,
                output,
                parameters_.vertex_condition_tolerance,
                parameters_.vertex_correction_relative_tolerance,
                parameters_.vertex_correction_max_iterations,
                true);

        if (ordinary.converged) {
            return true;
        }

        // The ordinary corrector leaves output untouched when it does not
        // converge, so the Float128 path starts from the original RayCaster
        // candidate rather than from a failed double correction.
        output = initial;
        const auto extended =
            extended_vertex_solver_.correct_vertex(
                nodes,
                support,
                initial,
                output,
                Scalar{0},
                parameters_.vertex_correction_relative_tolerance,
                parameters_.vertex_correction_max_iterations,
                false);

        if (!extended.converged) {
            output = initial;
            return false;
        }
        return true;
    }

    /**
     * @brief Verify the geometric and nearest-neighbour conditions of a vertex.
     *
     * This is intentionally semantic rather than a literal transcription of
     * Julia's debugging routine: all supplied generators must lie on one sphere,
     * no generator may lie meaningfully inside that sphere, every generator on
     * the sphere within tolerance must be represented by sigma, and the
     * generator differences must span the full ambient dimension.
     */
    template <class Sigma>
    [[nodiscard]] bool verify_vertex(
        const Sigma& sigma,
        const Point& position) {
        const std::size_t dim = dimension_size();
        if (sigma.size() < dim + 1) {
            return false;
        }

        load_node(sigma[0], reference_node_);
        const Scalar radius = (position - reference_node_).norm();
        const Scalar scale = std::max(Scalar{1}, radius);
        const Scalar sphere_tolerance =
            std::max(parameters_.verification_absolute_tolerance,
                     parameters_.verification_relative_tolerance * scale);

        for (const Index index : sigma) {
            load_node(index, candidate_node_);
            if (std::abs((position - candidate_node_).norm() - radius) >
                sphere_tolerance) {
                return false;
            }
        }

        // The nearest node must be one of the vertex generators.
        tree_.write_point(search_data_, position);
        const auto never_skip = [](Index) noexcept { return false; };
        const auto nearest = tree_.nn(search_data_, never_skip);
        if (!nearest || !detail::contains_index(sigma, nearest->index)) {
            return false;
        }

        // Search slightly beyond the sphere because inrange() is strict.
        const Scalar search_radius =
            radius + Scalar{2} * sphere_tolerance;
        tree_.write_point(search_data_, position);
        tree_.inrange(search_data_, search_radius, never_skip);

        for (const auto& entry : search_data_.list.entries) {
            load_node(entry.index, candidate_node_);
            const Scalar distance = (position - candidate_node_).norm();

            if (distance < radius - sphere_tolerance) {
                return false;
            }

            if (std::abs(distance - radius) <= sphere_tolerance &&
                !detail::contains_index(sigma, entry.index)) {
                return false;
            }
        }

        // Full-dimensional rank is required for an isolated Voronoi vertex.
        rank_matrix_.resize(
            static_cast<Eigen::Index>(sigma.size() - 1),
            static_cast<Eigen::Index>(dim));
        load_node(sigma.back(), reference_node_);
        for (std::size_t row = 0; row + 1 < sigma.size(); ++row) {
            load_node(sigma[row], candidate_node_);
            rank_matrix_.row(static_cast<Eigen::Index>(row)) =
                (candidate_node_ - reference_node_).transpose();
        }

        Eigen::ColPivHouseholderQR<detail::DynamicRayMatrix<Scalar>> rank_qr;
        rank_qr.setThreshold(parameters_.rank_tolerance);
        rank_qr.compute(rank_matrix_);
        return static_cast<std::size_t>(rank_qr.rank()) == dim;
    }

    /** @brief Julia-compatible scale-free equal-distance defect. */
    template <class Sigma>
    [[nodiscard]] Scalar vertex_variance(
        const Sigma& sigma,
        const Point& position) {
        if (sigma.empty()) {
            return infinity();
        }

        distance_buffer_.clear();
        distance_buffer_.reserve(sigma.size());

        Scalar mean = Scalar{0};
        for (const Index index : sigma) {
            load_node(index, candidate_node_);
            const Scalar squared_distance =
                (candidate_node_ - position).squaredNorm();
            distance_buffer_.push_back(squared_distance);
            mean += squared_distance;
        }
        mean /= static_cast<Scalar>(sigma.size());

        if (mean == Scalar{0}) {
            return Scalar{0};
        }

        Scalar variance = Scalar{0};
        for (const Scalar value : distance_buffer_) {
            const Scalar delta = value - mean;
            variance += delta * delta;
        }
        return variance / (mean * mean);
    }

protected:
    explicit RayCasterBase(
        const SearchTree& source_tree,
        Parameters parameters)
        : tree_(source_tree.safe_copy()),
          search_data_(tree_.make_backend_data()),
          parameters_(std::move(parameters)),
          reference_node_(make_point()),
          candidate_node_(make_point()),
          query_point_(make_point()),
          corrected_point_(make_point()),
          delta_(make_point()),
          scratch_point_(make_point()),
          bounds_min_(make_point()),
          bounds_max_(make_point()),
          vertex_solver_(dimension_size()),
          extended_vertex_solver_(dimension_size()) {
        const std::size_t reserve_count =
            std::max<std::size_t>(8, std::size_t{2} * dimension_size() + 4);
        candidate_indices_.reserve(reserve_count);
        parameter_buffer_.reserve(reserve_count);
        support_buffer_.reserve(dimension_size() + 1);
        distance_buffer_.reserve(reserve_count);
        search_data_.reserve(reserve_count);

        auto bounds = tree_.extended_nodes().boundary().bounding_box(
            tree_.extended_nodes().inner_nodes());
        bounds_min_ = std::move(bounds.first);
        bounds_max_ = std::move(bounds.second);
    }

    [[nodiscard]] static constexpr Scalar infinity() noexcept {
        if constexpr (std::numeric_limits<Scalar>::has_infinity) {
            return std::numeric_limits<Scalar>::infinity();
        } else {
            return std::numeric_limits<Scalar>::max();
        }
    }

    [[nodiscard]] std::size_t dimension_size() const noexcept {
        return static_cast<std::size_t>(tree_.dimension());
    }

    [[nodiscard]] Point make_point() const {
        if constexpr (DimensionAtCompileTime == Dynamic) {
            return Point(static_cast<Eigen::Index>(tree_.dimension()));
        } else {
            return Point{};
        }
    }

    void load_node(Index index, Point& target) const {
        tree_.extended_nodes().copy_node(index, target.data());
    }

    [[nodiscard]] bool in_bounds(const Point& point) const noexcept {
        for (std::size_t c = 0; c < dimension_size(); ++c) {
            const Eigen::Index ec = static_cast<Eigen::Index>(c);
            if (point[ec] < bounds_min_[ec] || point[ec] > bounds_max_[ec]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] Scalar node_dot(
        Index index,
        const Point& direction) const {
        Scalar value = Scalar{0};
        const ExtendedNodes& nodes = tree_.extended_nodes();
        for (Index coordinate = Index{0};
             coordinate < tree_.dimension();
             ++coordinate) {
            value += nodes.get_data(index, coordinate) *
                     direction[static_cast<Eigen::Index>(coordinate)];
        }
        return value;
    }

    template <class Sigma>
    [[nodiscard]] Scalar maximum_projection(
        const Sigma& sigma,
        const Point& direction) const {
        if (sigma.empty()) {
            throw std::invalid_argument(
                "Ray cast requires a non-empty signature.");
        }

        Scalar maximum = node_dot(sigma[0], direction);
        for (std::size_t i = 1; i < sigma.size(); ++i) {
            maximum = std::max(
                maximum,
                node_dot(sigma[i], direction));
        }
        return maximum;
    }

    [[nodiscard]] std::pair<Scalar, Scalar> get_t_impl(
        const Point& start,
        const Point& direction,
        const Point& x0,
        const Point& new_node,
        Scalar direction_error,
        bool with_error) {
        delta_ = new_node - x0;
        scratch_point_ = x0 + new_node - Scalar{2} * start;

        const Scalar denominator = direction.dot(delta_);
        const Scalar value =
            delta_.dot(scratch_point_) / (Scalar{2} * denominator);

        if (!with_error) {
            return {value, Scalar{0}};
        }

        const Scalar normalized_denominator = denominator / delta_.norm();
        const Scalar error =
            (std::abs(value) * direction_error + start.norm() * Scalar{1e-15}) /
            normalized_denominator;
        return {value, error};
    }

    [[nodiscard]] Scalar get_t(
        const Point& start,
        const Point& direction,
        const Point& x0,
        const Point& new_node) {
        return get_t_impl(
            start, direction, x0, new_node, Scalar{0}, false).first;
    }

    [[nodiscard]] std::pair<Scalar, Scalar> get_t_with_error(
        const Point& start,
        const Point& direction,
        const Point& x0,
        const Point& new_node,
        Scalar direction_error) {
        return get_t_impl(
            start, direction, x0, new_node, direction_error, true);
    }

    [[nodiscard]] Scalar get_scale(
        const Point& direction,
        const Point& x0,
        const Point& position) {
        delta_ = position - x0;
        const Scalar reference = delta_.squaredNorm();
        const Scalar projection = direction.dot(delta_);
        const Scalar vertical = projection * projection;
        const Scalar horizontal =
            vertical > reference ? Scalar{0} : reference - vertical;
        return std::sqrt(horizontal / reference);
    }

    void raycast_start_heuristic(
        const Point& x0,
        const Point& start,
        const Point& direction,
        Point& result) {
        delta_ = x0 - start;
        result = start + direction * direction.dot(delta_);

        const std::size_t dim = dimension_size();
        if (dim > 1) {
            delta_ = result - x0;
            const Scalar radius = delta_.norm();
            const Scalar denominator = std::sqrt(
                static_cast<Scalar>((dim + 1) * (dim - 1)));
            result += (radius / denominator) * direction;
        }
    }

    template <class Edge>
    [[nodiscard]] bool correction_support(
        const Edge& edge,
        Index generator) {
        const std::size_t dim = dimension_size();
        if (edge.size() < dim) {
            return false;
        }

        support_buffer_.clear();
        for (const Index index : edge) {
            if (!detail::contains_index(support_buffer_, index)) {
                support_buffer_.push_back(index);
                if (support_buffer_.size() == dim) {
                    break;
                }
            }
        }
        if (support_buffer_.size() != dim ||
            detail::contains_index(support_buffer_, generator)) {
            return false;
        }
        support_buffer_.push_back(generator);
        return true;
    }

    template <class Edge>
    [[nodiscard]] bool correct_cast(
        const Point& raw,
        const Edge& edge,
        Index generator,
        RayCastUsage usage,
        Point& output) {
        output = raw;
        if (usage == RayCastUsage::Descent) {
            return false;
        }
        if (!correction_support(edge, generator)) {
            return false;
        }
        return correct_vertex(support_buffer_, output);
    }

    template <class Sigma, class Edge>
    void validate_cast_input(
        const Sigma& sigma,
        const Point& start,
        const Point& direction,
        const Edge& edge) const {
        const std::size_t dim = dimension_size();
        if (sigma.empty() || edge.empty()) {
            throw std::invalid_argument(
                "Ray cast requires non-empty sigma and edge.");
        }
        if (static_cast<std::size_t>(start.size()) != dim ||
            static_cast<std::size_t>(direction.size()) != dim) {
            throw std::invalid_argument(
                "Ray cast point dimension does not match SearchTree dimension.");
        }
        const Scalar direction_norm = direction.norm();
        if (!(direction_norm > Scalar{0}) || !std::isfinite(direction_norm)) {
            throw std::invalid_argument(
                "Ray cast direction must be finite and non-zero.");
        }
    }

    [[nodiscard]] Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    [[nodiscard]] const Derived& derived() const noexcept {
        return static_cast<const Derived&>(*this);
    }

    SearchTree tree_;
    SearchData search_data_;
    Parameters parameters_;

    Point reference_node_;
    Point candidate_node_;
    Point query_point_;
    Point corrected_point_;
    Point delta_;
    Point scratch_point_;
    Point bounds_min_;
    Point bounds_max_;

    QRNormalSolver<Scalar, DimensionAtCompileTime> vertex_solver_;
    ExtendedQRNormalSolver<Scalar, DimensionAtCompileTime> extended_vertex_solver_;

    std::vector<Index> candidate_indices_;
    std::vector<Scalar> parameter_buffer_;
    std::vector<Index> support_buffer_;
    std::vector<Scalar> distance_buffer_;
    detail::DynamicRayMatrix<Scalar> rank_matrix_;
};

/**
 * @brief Classical iterative nearest-neighbour ray cast.
 */
template <class SearchTreeT>
class ClassicRayCaster final
    : public RayCasterBase<
          ClassicRayCaster<SearchTreeT>,
          SearchTreeT,
          RaycastParameters<ClassicRaycast, typename SearchTreeT::Scalar>> {
public:
    using Parameters =
        RaycastParameters<ClassicRaycast, typename SearchTreeT::Scalar>;
    using Base = RayCasterBase<
        ClassicRayCaster<SearchTreeT>,
        SearchTreeT,
        Parameters>;
    using typename Base::Index;
    using typename Base::Point;
    using typename Base::Result;
    using typename Base::Scalar;

    explicit ClassicRayCaster(
        const SearchTreeT& tree,
        Parameters parameters = Parameters{})
        : Base(tree, std::move(parameters)) {}

    [[nodiscard]] ClassicRayCaster safe_copy_impl() const {
        return ClassicRayCaster(this->tree_, this->parameters_);
    }

    /** @brief Recreate this ray-cast policy on another compatible mesh. */
    template <class NewMesh>
    [[nodiscard]] auto rebind(NewMesh& mesh) const {
        auto tree = this->tree_.rebind(mesh);
        return ClassicRayCaster<decltype(tree)>(tree, this->parameters_);
    }

    template <class Sigma, class Edge, class Origin>
    [[nodiscard]] Result cast_impl(
        Sigma& sigma,
        const Point& start,
        const Point& direction,
        const Edge& edge,
        const Origin& /* origin */,
        RayCastUsage usage,
        Scalar direction_error) {
        this->load_node(edge[0], this->reference_node_);
        const Point& x0 = this->reference_node_;

        const Scalar c1 = this->maximum_projection(sigma, direction);
        const auto skip_initial = [this, &sigma, &direction, c1](Index index) {
            return std::binary_search(sigma.begin(), sigma.end(), index) ||
                   this->node_dot(index, direction) <= c1;
        };

        this->raycast_start_heuristic(
            x0, start, direction, this->query_point_);
        this->tree_.write_point(this->search_data_, this->query_point_);
        auto nearest = this->tree_.nn(this->search_data_, skip_initial);
        if (!nearest) {
            return infinite_result(start);
        }

        Index current_generator = nearest->index;
        Index best_generator = current_generator;
        this->load_node(current_generator, this->candidate_node_);

        const auto [first_t, full_error] = this->get_t_with_error(
            start,
            direction,
            x0,
            this->candidate_node_,
            direction_error);
        if (!std::isfinite(first_t)) {
            return infinite_result(start);
        }

        this->query_point_ = start + first_t * direction;
        const Scalar scale =
            this->get_scale(direction, x0, this->query_point_);
        const Scalar relative_error =
            full_error / this->query_point_.norm();
        const bool full_mode =
            relative_error > this->parameters_.classic_relative_error_trigger ||
            full_error >
                this->parameters_.classic_absolute_error_trigger /
                    std::max(scale, Scalar{1e-4});

        Scalar current_t = Base::infinity();
        const auto never_skip = [](Index) noexcept { return false; };

        while (true) {
            this->load_node(current_generator, this->candidate_node_);
            const Scalar t = this->get_t(
                start, direction, x0, this->candidate_node_);
            if (!std::isfinite(t) || t >= current_t) {
                break;
            }
            current_t = t;

            this->query_point_ = start + t * direction;
            if (full_mode && usage == RayCastUsage::WalkRay &&
                this->correct_cast(
                    this->query_point_, edge, current_generator,
                    usage, this->corrected_point_)) {
                this->query_point_ = this->corrected_point_;
            }

            this->tree_.write_point(this->search_data_, this->query_point_);
            nearest = this->tree_.nn(this->search_data_, never_skip);
            if (!nearest ||
                nearest->index == best_generator ||
                detail::contains_index(sigma, nearest->index)) {
                break;
            }

            current_generator = nearest->index;
            best_generator = current_generator;
        }

        this->load_node(best_generator, this->candidate_node_);
        const Scalar t = this->get_t(
            start, direction, x0, this->candidate_node_);
        if (!std::isfinite(t)) {
            return infinite_result(start);
        }

        this->query_point_ = start + t * direction;
        bool corrected = false;
        if (full_mode && usage == RayCastUsage::WalkRay) {
            corrected = this->correct_cast(
                this->query_point_, edge, best_generator,
                usage, this->corrected_point_);
            if (corrected) {
                this->query_point_ = this->corrected_point_;
            }
        }

        sigma.push_back(best_generator);
        detail::sort_unique(sigma);

        return Result{
            RayCastStatus::Finite,
            best_generator,
            t,
            this->query_point_,
            corrected};
    }

private:
    [[nodiscard]] Result infinite_result(const Point& start) const {
        return Result{
            RayCastStatus::Infinite,
            Index{},
            Base::infinity(),
            start,
            false};
    }
};

/**
 * @brief Ray cast for potentially degenerate endpoint vertices.
 *
 * The implementation follows the Julia Raycast_Non_General path: NN provides
 * an initial forward estimate, an in-range search gathers nearby candidates,
 * ray parameters determine the first admissible endpoint plane, and all nodes
 * lying on the resulting endpoint sphere are appended to sigma.
 */
template <class SearchTreeT>
class InRangeRayCaster final
    : public RayCasterBase<
          InRangeRayCaster<SearchTreeT>,
          SearchTreeT,
          RaycastParameters<InRangeRaycast, typename SearchTreeT::Scalar>> {
public:
    using Parameters =
        RaycastParameters<InRangeRaycast, typename SearchTreeT::Scalar>;
    using Base = RayCasterBase<
        InRangeRayCaster<SearchTreeT>,
        SearchTreeT,
        Parameters>;
    using typename Base::Index;
    using typename Base::Point;
    using typename Base::Result;
    using typename Base::Scalar;

    explicit InRangeRayCaster(
        const SearchTreeT& tree,
        Parameters parameters = Parameters{})
        : Base(tree, std::move(parameters)) {}

    [[nodiscard]] InRangeRayCaster safe_copy_impl() const {
        return InRangeRayCaster(this->tree_, this->parameters_);
    }

    /** @brief Recreate this ray-cast policy on another compatible mesh. */
    template <class NewMesh>
    [[nodiscard]] auto rebind(NewMesh& mesh) const {
        auto tree = this->tree_.rebind(mesh);
        return InRangeRayCaster<decltype(tree)>(tree, this->parameters_);
    }

    template <class Sigma, class Edge, class Origin>
    [[nodiscard]] Result cast_impl(
        Sigma& sigma,
        const Point& start,
        const Point& direction,
        const Edge& edge,
        const Origin& origin,
        RayCastUsage usage,
        Scalar direction_error) {
        const std::size_t dim = this->dimension_size();
        const Index invalid = invalid_index();

        this->load_node(edge[0], this->reference_node_);
        const Point& x0 = this->reference_node_;
        bool full_mode = usage == RayCastUsage::WalkRay;

        const Scalar c1 = this->maximum_projection(sigma, direction);
        const Scalar c = c1 + std::abs(c1) * this->parameters_.plane_tolerance;
        const auto skip = [this, &direction, c](Index index) {
            return this->node_dot(index, direction) <= c;
        };

        // Julia: vvv -> first NN -> t/full_error.
        this->delta_ = x0 - start;
        this->query_point_ = start + direction * direction.dot(this->delta_);
        this->tree_.write_point(this->search_data_, this->query_point_);
        auto nearest = this->tree_.nn(this->search_data_, skip);
        if (!nearest) {
            return infinite_result(start);
        }

        Index current = nearest->index;
        this->load_node(current, this->candidate_node_);
        auto [t, full_error] = this->get_t_with_error(
            start, direction, x0, this->candidate_node_, direction_error);
        const Scalar first_t = t;

        this->query_point_ = start + t * direction;
        const Scalar scale = this->in_bounds(this->query_point_)
            ? Scalar{1}
            : this->get_scale(direction, x0, this->query_point_);
        this->delta_ = this->query_point_ - start;
        const Scalar relative_error = full_error / this->delta_.norm();
        full_mode &=
            relative_error > this->parameters_.classic_relative_error_trigger ||
            full_error > this->parameters_.classic_absolute_error_trigger /
                std::max(scale, Scalar{1e-4});

        t += this->get_t_with_error(
            this->query_point_, direction, x0,
            this->candidate_node_, direction_error).first;
        this->query_point_ += t * direction;
        if (full_mode && this->correct_cast(
                this->query_point_, edge, current, usage, this->corrected_point_)) {
            this->query_point_ = this->corrected_point_;
        }

        // Julia get__r. The refined point remains in query_point_.
        auto [returned_t, new_generator] = refine_nn(
            start, direction, x0, edge, skip, usage, full_mode, first_t);
        if (new_generator == invalid) {
            return infinite_result(start);
        }
        t = returned_t;

        // Julia: one inrange search, then keep the first admissible t-plane.
        Scalar measure = Scalar{0};
        for (const Index index : edge) {
            this->load_node(index, this->candidate_node_);
            this->delta_ = this->candidate_node_ - this->query_point_;
            measure = std::max(measure, this->delta_.norm());
        }

        Scalar upper_t = t * Scalar{1.0000000001};
        this->tree_.write_point(this->search_data_, this->query_point_);
        const auto never_skip = [](Index) noexcept { return false; };
        this->tree_.inrange(
            this->search_data_,
            (Scalar{1} + std::max(
                relative_error,
                this->parameters_.boundary_node_tolerance * Scalar{10} * scale)) * measure,
            never_skip);
        if (this->search_data_.list.empty()) {
            return infinite_result(start);
        }

        this->candidate_indices_.clear();
        this->parameter_buffer_.clear();
        for (const auto& entry : this->search_data_.list.entries) {
            this->candidate_indices_.push_back(entry.index);
            if (std::binary_search(origin.begin(), origin.end(), entry.index)) {
                this->parameter_buffer_.push_back(Scalar{0});
            } else {
                this->load_node(entry.index, this->candidate_node_);
                this->parameter_buffer_.push_back(
                    this->get_t(start, direction, x0, this->candidate_node_));
            }
        }

        for (std::size_t k = 0; k < this->candidate_indices_.size(); ++k) {
            if (std::binary_search(
                    origin.begin(), origin.end(), this->candidate_indices_[k])) {
                this->candidate_indices_[k] = invalid;
                continue;
            }
            Scalar& candidate_t = this->parameter_buffer_[k];
            if (candidate_t < this->parameters_.plane_tolerance || candidate_t > upper_t) {
                candidate_t = Scalar{0};
            } else if (candidate_t < upper_t) {
                upper_t = candidate_t;
            }
        }

        upper_t += std::min(
            Scalar{1e-7}, static_cast<Scalar>(10 + dim) * full_error);

        Scalar maximum_forward = Scalar{0};
        const Scalar x0_projection = this->node_dot(edge[0], direction);
        Index generator = invalid;
        for (std::size_t k = 0; k < this->candidate_indices_.size(); ++k) {
            Scalar& candidate_t = this->parameter_buffer_[k];
            if (candidate_t > upper_t) {
                candidate_t = Scalar{0};
            } else if (candidate_t > Scalar{0}) {
                const Index index = this->candidate_indices_[k];
                candidate_t =
                    this->node_dot(index, direction) - x0_projection;
                if (candidate_t > maximum_forward) {
                    maximum_forward = candidate_t;
                    generator = index;
                }
            }
        }
        if (generator == invalid) {
            return infinite_result(start);
        }

        // Julia: pre_r2 -> optional correction -> final sphere filter.
        this->load_node(generator, this->candidate_node_);
        t = this->get_t(start, direction, x0, this->candidate_node_);
        this->query_point_ = start + t * direction;

        this->support_buffer_.clear();
        const std::size_t variance_count =
            std::min(dim + 1, static_cast<std::size_t>(edge.size()));
        for (std::size_t i = 0; i < variance_count; ++i) {
            this->support_buffer_.push_back(edge[i]);
        }
        const bool variance_requests_correction =
            this->parameters_.variance_tolerance * Scalar{1e-5} <
            this->vertex_variance(this->support_buffer_, this->query_point_);

        const bool corrected =
            usage == RayCastUsage::WalkRay &&
            (full_mode || variance_requests_correction) &&
            this->correct_cast(
                this->query_point_, edge, generator,
                usage, this->corrected_point_);
        if (corrected) {
            this->query_point_ = this->corrected_point_;
        }

        this->load_node(generator, this->candidate_node_);
        this->delta_ = this->candidate_node_ - this->query_point_;
        Scalar min_measure = this->delta_.norm();
        Scalar max_measure = min_measure;
        for (const Index index : sigma) {
            this->load_node(index, this->candidate_node_);
            this->delta_ = this->candidate_node_ - this->query_point_;
            const Scalar value = this->delta_.norm();
            min_measure = std::min(min_measure, value);
            max_measure = std::max(max_measure, value);
        }
        const Scalar measure2 =
            max_measure + Scalar{10} * static_cast<Scalar>(dim) * scale *
                std::max(max_measure - min_measure, full_error);

        for (Index& index : this->candidate_indices_) {
            if (index == invalid) {
                continue;
            }
            this->load_node(index, this->candidate_node_);
            this->delta_ = this->candidate_node_ - this->query_point_;
            if (this->delta_.norm() > measure2) {
                index = invalid;
            }
        }
        for (const Index index : this->candidate_indices_) {
            if (index != invalid) {
                sigma.push_back(index);
            }
        }
        detail::sort_unique(sigma);

        return Result{
            RayCastStatus::Finite,
            generator,
            t,
            this->query_point_,
            corrected};
    }

private:
    template <class Edge, class Skip>
    [[nodiscard]] std::pair<Scalar, Index> refine_nn(
        const Point& start,
        const Point& direction,
        const Point& x0,
        const Edge& edge,
        const Skip& skip,
        RayCastUsage usage,
        bool full_mode,
        Scalar first_t) {
        const Index invalid = invalid_index();
        Scalar returned_t = first_t;
        this->delta_ = x0 - this->query_point_;
        Scalar best_distance = this->delta_.norm();
        Index generator = invalid;

        for (int iteration = 0; iteration < 2; ++iteration) {
            this->tree_.write_point(this->search_data_, this->query_point_);
            const auto nearest = this->tree_.nn(this->search_data_, skip);
            if (!nearest || nearest->index == edge[0]) {
                break;
            }

            generator = nearest->index;
            this->load_node(generator, this->candidate_node_);
            this->delta_ = this->candidate_node_ - this->query_point_;
            if (this->delta_.norm() >= best_distance) {
                break;
            }

            Scalar t = this->get_t(
                start, direction, x0, this->candidate_node_);
            returned_t = t;
            this->scratch_point_ = start + t * direction;
            t += this->get_t(
                this->scratch_point_, direction, x0, this->candidate_node_);
            this->query_point_ = start + t * direction;

            if (full_mode && this->correct_cast(
                    this->query_point_, edge, generator,
                    usage, this->corrected_point_)) {
                this->query_point_ = this->corrected_point_;
            }

            this->delta_ = this->candidate_node_ - this->query_point_;
            const Scalar distance = this->delta_.norm();
            if (distance >= best_distance) {
                break;
            }
            best_distance = distance;
        }

        return {returned_t, generator};
    }

    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return std::numeric_limits<Index>::max();
    }

    [[nodiscard]] Result infinite_result(const Point& start) const {
        return Result{
            RayCastStatus::Infinite,
            Index{},
            Base::infinity(),
            start,
            false};
    }
};

// ============================================================================
// Factory: public RaycastParameters select a concrete implementation
// ============================================================================

template <class SearchTree>
[[nodiscard]] ClassicRayCaster<SearchTree> make_raycaster(
    const SearchTree& tree,
    RaycastParameters<ClassicRaycast, typename SearchTree::Scalar> parameters) {
    return ClassicRayCaster<SearchTree>(tree, std::move(parameters));
}

template <class SearchTree>
[[nodiscard]] InRangeRayCaster<SearchTree> make_raycaster(
    const SearchTree& tree,
    RaycastParameters<InRangeRaycast, typename SearchTree::Scalar> parameters) {
    return InRangeRayCaster<SearchTree>(tree, std::move(parameters));
}

} // namespace highvoronoi
