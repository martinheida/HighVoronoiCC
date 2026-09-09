#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <highvoronoi/core/point.hpp>

namespace highvoronoi {

/**
 * @file boundary.hpp
 * @brief Owning planar boundary primitives for fixed or runtime dimension.
 *
 * Every base point and every normal vector owns its storage independently.
 * Fixed-dimensional classes use `StaticPoint`, while runtime-dimensional
 * classes use `DynamicPoint`.
 *
 * The normal of a plane points out of the domain. Therefore, a point `x` lies
 * in the closed half-space represented by a plane when
 *
 * @code{.cpp}
 * (x - plane.base()).dot(plane.normal()) <= tolerance
 * @endcode
 *
 * Periodic partner indices are zero-based, as usual in C++.
 */

/** @brief Boundary condition attached to a plane. */
enum class BoundaryCondition {
    Dirichlet,
    Neumann,
    Periodic
};

template <int Dim = Dynamic,
          typename Scalar = double,
          typename Index = std::size_t>
class Boundary;

namespace detail {

template <int Dim, typename Scalar>
using BoundaryPoint = std::conditional_t<
    Dim == Dynamic,
    DynamicPoint<Scalar>,
    StaticPoint<Scalar, Dim>>;

template <int Dim, typename Scalar, typename Derived>
[[nodiscard]] BoundaryPoint<Dim, Scalar>
make_owned_point(const Eigen::MatrixBase<Derived>& point) {
    static_assert(Derived::IsVectorAtCompileTime,
                  "Boundary coordinates must be vector expressions.");

    if constexpr (Dim == Dynamic) {
        return BoundaryPoint<Dim, Scalar>(point);
    } else {
        if (point.size() != Dim) {
            throw std::invalid_argument(
                "Point dimension does not match fixed boundary dimension.");
        }
        return BoundaryPoint<Dim, Scalar>(point);
    }
}

template <int Dim, typename Scalar>
[[nodiscard]] BoundaryPoint<Dim, Scalar>
zero_point(Eigen::Index dimension) {
    if constexpr (Dim == Dynamic) {
        return BoundaryPoint<Dim, Scalar>::Zero(dimension);
    } else {
        (void)dimension;
        return BoundaryPoint<Dim, Scalar>::Zero();
    }
}

} // namespace detail

/**
 * @brief Represents an oriented plane together with its boundary condition.
 *
 * A plane is defined by a base point and an outward-pointing normal vector.
 * The normal vector is normalized when the plane is constructed or when the
 * normal is replaced.
 *
 * The plane divides space into an inner and an outer half-space. Points for
 * which
 *
 * @code
 * (point - base) dot normal <= 0
 * @endcode
 *
 * lie inside the associated half-space.
 *
 * The class provides the following geometric operations:
 *
 * - `halfspace_value(point)` evaluates the oriented position of a point
 *   relative to the plane. Negative values indicate the inner half-space,
 *   positive values indicate the outer half-space, and zero denotes the plane.
 *
 * - `signed_distance(point)` returns the signed Euclidean distance from a
 *   point to the plane.
 *
 * - `contains(point, tolerance)` tests whether a point lies inside the closed
 *   half-space, optionally enlarged by a tolerance.
 *
 * - `project(point)` returns the orthogonal projection of a point onto the
 *   plane.
 *
 * - `reflect(point)` returns the mirror image of a point with respect to the
 *   plane. This is a purely geometric reflection and does not perform a
 *   periodic boundary mapping.
 *
 * - `intersection_parameter(origin, direction)` computes the parameter of the
 *   intersection between the plane and the line
 *   `origin + parameter * direction`. No value is returned for a line parallel
 *   to the plane.
 *
 * - `translate(displacement)` moves the plane without changing its orientation.
 *
 * - `set_base(base)` replaces the base point.
 *
 * - `set_normal(normal)` replaces and normalizes the normal vector.
 *
 * A plane may be marked as Dirichlet, Neumann, or periodic. For a periodic
 * plane, the class stores the index of its partner plane. The actual mapping
 * between periodic partner planes is not performed by `reflect()` or
 * `project()` and must be handled by the surrounding boundary structure.
 */
 template <int Dim = Dynamic,
          typename ScalarT = double,
          typename IndexT = std::size_t>
class Plane {
    template <int, typename, typename>
    friend class Boundary;

public:
    using Scalar = ScalarT;
    using Index = IndexT;
    using Point = detail::BoundaryPoint<Dim, Scalar>;

    static_assert(Dim == Dynamic || Dim > 0,
                  "Plane dimension must be positive or highvoronoi::Dynamic.");
    static_assert(std::is_integral_v<Index> && std::is_unsigned_v<Index>,
                  "Plane index type must be an unsigned integer type.");

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr bool IsDynamic = (Dim == Dynamic);

    template <typename BaseDerived, typename NormalDerived>
    Plane(const Eigen::MatrixBase<BaseDerived>& base,
          const Eigen::MatrixBase<NormalDerived>& normal,
          BoundaryCondition condition,
          std::optional<Index> periodic_partner = std::nullopt)
        : base_(detail::make_owned_point<Dim, Scalar>(base)),
          normal_(detail::make_owned_point<Dim, Scalar>(normal)),
          condition_(condition),
          periodic_partner_(periodic_partner) {
        if constexpr (IsDynamic) {
            if (base_.size() == 0) {
                throw std::invalid_argument("Plane dimension must be positive.");
            }
            if (normal_.size() != base_.size()) {
                throw std::invalid_argument(
                    "Plane base and normal must have the same dimension.");
            }
        }
        if (normal_.squaredNorm() == Scalar{0}) {
            throw std::invalid_argument("Plane normal must be non-zero.");
        }
        normal_.normalize();
        validate_boundary_condition();
    }

    template <typename BaseDerived, typename NormalDerived>
    [[nodiscard]] static Plane periodic(
        const Eigen::MatrixBase<BaseDerived>& base,
        const Eigen::MatrixBase<NormalDerived>& normal,
        Index partner) {
        return Plane(base, normal, BoundaryCondition::Periodic, partner);
    }

    [[nodiscard]] const Point& base() const noexcept {
        return base_;
    }

    [[nodiscard]] const Point& normal() const noexcept {
        return normal_;
    }

    [[nodiscard]] std::size_t dimension() const noexcept {
        return static_cast<std::size_t>(base_.size());
    }

    [[nodiscard]] BoundaryCondition condition() const noexcept {
        return condition_;
    }

    [[nodiscard]] bool is_dirichlet() const noexcept {
        return condition_ == BoundaryCondition::Dirichlet;
    }

    [[nodiscard]] bool is_neumann() const noexcept {
        return condition_ == BoundaryCondition::Neumann;
    }

    [[nodiscard]] bool is_periodic() const noexcept {
        return condition_ == BoundaryCondition::Periodic;
    }

    [[nodiscard]] std::optional<Index> periodic_partner() const noexcept {
        return periodic_partner_;
    }

    template <typename Derived>
    void set_base(const Eigen::MatrixBase<Derived>& new_base) {
        require_dimension(new_base);
        base_ = new_base;
    }

    template <typename Derived>
    void set_normal(const Eigen::MatrixBase<Derived>& new_normal) {
        require_dimension(new_normal);
        if (new_normal.squaredNorm() == Scalar{0}) {
            throw std::invalid_argument("Plane normal must be non-zero.");
        }
        normal_ = new_normal;
        normal_.normalize();
    }

    template <typename Derived>
    void translate(const Eigen::MatrixBase<Derived>& displacement) {
        require_dimension(displacement);
        base_ += displacement;
    }

    /**
     * @brief Return `(point - base).dot(normal)`.
     *
     * Negative values lie inside the half-space and positive values outside.
     * Unlike `signed_distance()`, this operation evaluates no square root.
     */
    template <typename Derived>
    [[nodiscard]] Scalar halfspace_value(
        const Eigen::MatrixBase<Derived>& point) const {
        require_dimension(point);
        return halfspace_value_unchecked(point);
    }

    /** @brief Return the signed Euclidean distance to the plane. */
    template <typename Derived>
    [[nodiscard]] Scalar signed_distance(
        const Eigen::MatrixBase<Derived>& point) const {
        require_dimension(point);
        return halfspace_value_unchecked(point);
    }

    /** @brief Test membership in the plane's closed half-space. */
    template <typename Derived>
    [[nodiscard]] bool contains(
        const Eigen::MatrixBase<Derived>& point,
        Scalar tolerance = Scalar{0}) const {
        require_dimension(point);
        return halfspace_value_unchecked(point) <= tolerance;
    }

    /**
     * @brief Reflect a point at this plane.
     *
     * A fixed-dimensional plane returns `StaticPoint<Scalar, Dim>`. A dynamic
     * plane returns `DynamicPoint<Scalar>`.
     */
    template <typename Derived>
    [[nodiscard]] Point reflect(
        const Eigen::MatrixBase<Derived>& point) const {
        require_dimension(point);
        Point result(point);
        reflect_in_place_unchecked(result);
        return result;
    }

    /** @brief Orthogonally project a point onto this plane. */
    template <typename Derived>
    [[nodiscard]] Point project(
        const Eigen::MatrixBase<Derived>& point) const {
        require_dimension(point);
        Point result(point);
        project_in_place_unchecked(result);
        return result;
    }

    /**
     * @brief Compute `t` for the line-plane intersection
     *        `origin + t * direction`.
     *
     * `std::nullopt` is returned when the direction is parallel to the plane.
     * The default uses an exact zero test, matching the direct Julia formula.
     * A non-zero tolerance may be supplied explicitly when desired.
     */
    template <typename OriginDerived, typename DirectionDerived>
    [[nodiscard]] std::optional<Scalar> intersection_parameter(
        const Eigen::MatrixBase<OriginDerived>& origin,
        const Eigen::MatrixBase<DirectionDerived>& direction,
        Scalar parallel_tolerance = Scalar{0}) const {
        require_dimension(origin);
        require_dimension(direction);
        return intersection_parameter_unchecked(
            origin, direction, parallel_tolerance);
    }

    [[nodiscard]] Plane<Dynamic, Scalar, Index> to_dynamic() const {
        return Plane<Dynamic, Scalar, Index>(
            base_, normal_, condition_, periodic_partner_);
    }

    template <int TargetDim>
    [[nodiscard]] Plane<TargetDim, Scalar, Index> to_static() const {
        static_assert(TargetDim > 0,
                      "Target plane dimension must be positive.");
        if (base_.size() != TargetDim) {
            throw std::invalid_argument(
                "Cannot convert Plane to requested static dimension.");
        }
        return Plane<TargetDim, Scalar, Index>(
            base_, normal_, condition_, periodic_partner_);
    }

private:
    void validate_boundary_condition() const {
        if (is_periodic()) {
            if (!periodic_partner_) {
                throw std::invalid_argument(
                    "Periodic plane requires a partner index.");
            }
        } else if (periodic_partner_) {
            throw std::invalid_argument(
                "Only periodic planes may have a partner index.");
        }
    }

    template <typename Derived>
    void require_dimension(const Eigen::MatrixBase<Derived>& point) const {
        static_assert(Derived::IsVectorAtCompileTime,
                      "Boundary coordinates must be vector expressions.");
        if (point.size() != base_.size()) {
            throw std::invalid_argument(
                "Point dimension does not match plane dimension.");
        }
    }

    template <typename Derived>
    [[nodiscard]] Scalar halfspace_value_unchecked(
        const Eigen::MatrixBase<Derived>& point) const {
        return (point - base_).dot(normal_);
    }

    template <typename Derived>
    void reflect_in_place_unchecked(Eigen::MatrixBase<Derived>& point) const {
        point.derived() -=
            (Scalar{2} * halfspace_value_unchecked(point)) * normal_;
    }

    template <typename Derived>
    void project_in_place_unchecked(Eigen::MatrixBase<Derived>& point) const {
        point.derived() -=
            (halfspace_value_unchecked(point)) * normal_;
    }

    template <typename OriginDerived, typename DirectionDerived>
    [[nodiscard]] std::optional<Scalar> intersection_parameter_unchecked(
        const Eigen::MatrixBase<OriginDerived>& origin,
        const Eigen::MatrixBase<DirectionDerived>& direction,
        Scalar parallel_tolerance) const {
        const Scalar denominator = direction.dot(normal_);
        using std::abs;
        if (abs(denominator) <= parallel_tolerance) {
            return std::nullopt;
        }
        return (base_ - origin).dot(normal_) / denominator;
    }

    Point base_;
    Point normal_;
    BoundaryCondition condition_;
    std::optional<Index> periodic_partner_;
};

/** @brief Plane with Dirichlet boundary condition. */
template <int Dim = Dynamic,
          typename ScalarT = double,
          typename IndexT = std::size_t>
class Dirichlet final : public Plane<Dim, ScalarT, IndexT> {
public:
    using Base = Plane<Dim, ScalarT, IndexT>;
    using Scalar = ScalarT;
    using Index = IndexT;

    static constexpr int DimensionAtCompileTime = Dim;

    template <typename BaseDerived, typename NormalDerived>
    Dirichlet(const Eigen::MatrixBase<BaseDerived>& base,
              const Eigen::MatrixBase<NormalDerived>& normal)
        : Base(base, normal, BoundaryCondition::Dirichlet) {}

    [[nodiscard]] Dirichlet<Dynamic, Scalar, Index> to_dynamic() const {
        return Dirichlet<Dynamic, Scalar, Index>(
            this->base(), this->normal());
    }

    template <int TargetDim>
    [[nodiscard]] Dirichlet<TargetDim, Scalar, Index> to_static() const {
        static_assert(TargetDim > 0,
                      "Target Dirichlet dimension must be positive.");
        if (this->base().size() != TargetDim) {
            throw std::invalid_argument(
                "Cannot convert Dirichlet to requested static dimension.");
        }
        return Dirichlet<TargetDim, Scalar, Index>(
            this->base(), this->normal());
    }
};

/** @brief Plane with Neumann boundary condition. */
template <int Dim = Dynamic,
          typename ScalarT = double,
          typename IndexT = std::size_t>
class Neumann final : public Plane<Dim, ScalarT, IndexT> {
public:
    using Base = Plane<Dim, ScalarT, IndexT>;
    using Scalar = ScalarT;
    using Index = IndexT;

    static constexpr int DimensionAtCompileTime = Dim;

    template <typename BaseDerived, typename NormalDerived>
    Neumann(const Eigen::MatrixBase<BaseDerived>& base,
            const Eigen::MatrixBase<NormalDerived>& normal)
        : Base(base, normal, BoundaryCondition::Neumann) {}

    [[nodiscard]] Neumann<Dynamic, Scalar, Index> to_dynamic() const {
        return Neumann<Dynamic, Scalar, Index>(
            this->base(), this->normal());
    }

    template <int TargetDim>
    [[nodiscard]] Neumann<TargetDim, Scalar, Index> to_static() const {
        static_assert(TargetDim > 0,
                      "Target Neumann dimension must be positive.");
        if (this->base().size() != TargetDim) {
            throw std::invalid_argument(
                "Cannot convert Neumann to requested static dimension.");
        }
        return Neumann<TargetDim, Scalar, Index>(
            this->base(), this->normal());
    }
};

/**
 * @brief Pair of corresponding periodic planes.
 *
 * The pair stores one corresponding base point on each periodic face together
 * with the two outward-pointing normals. Translating the first face by
 *
 * @code
 * second_base - first_base
 * @endcode
 *
 * maps it onto the second face. The translation may contain tangential
 * components and is therefore not generally parallel to either normal.
 *
 * The two base points must represent corresponding points of the periodic
 * faces. Arbitrary points on the two supporting planes are insufficient to
 * determine a unique periodic translation.
 *
 * Reciprocal partner indices are assigned only when the pair is inserted into
 * a `Boundary`.
 */
 template <int Dim = Dynamic,
          typename ScalarT = double,
          typename IndexT = std::size_t>
class Periodic {
public:
    using Scalar = ScalarT;
    using Index = IndexT;
    using Point = detail::BoundaryPoint<Dim, Scalar>;

    static_assert(Dim == Dynamic || Dim > 0,
                  "Periodic dimension must be positive or Dynamic.");
    static_assert(std::is_integral_v<Index> && std::is_unsigned_v<Index>,
                  "Periodic index type must be an unsigned integer type.");

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr bool IsDynamic = (Dim == Dynamic);

    /**
     * @brief Construct from two bases and the outward normal of the first plane.
     *
     * The second normal is the negation of the first normal, as in the Julia
     * `BC_Periodic(base1, base2, normal1)` constructor.
     */
    template <typename FirstBaseDerived,
              typename SecondBaseDerived,
              typename FirstNormalDerived>
    Periodic(const Eigen::MatrixBase<FirstBaseDerived>& first_base,
             const Eigen::MatrixBase<SecondBaseDerived>& second_base,
             const Eigen::MatrixBase<FirstNormalDerived>& first_normal)
        : first_base_(detail::make_owned_point<Dim, Scalar>(first_base)),
          second_base_(detail::make_owned_point<Dim, Scalar>(second_base)),
          first_normal_(detail::make_owned_point<Dim, Scalar>(first_normal)),
          second_normal_(-first_normal_) {
        validate_geometry();
    }

    /** @brief Construct from two complete plane geometries. */
    template <typename FirstBaseDerived,
              typename FirstNormalDerived,
              typename SecondBaseDerived,
              typename SecondNormalDerived>
    Periodic(const Eigen::MatrixBase<FirstBaseDerived>& first_base,
             const Eigen::MatrixBase<FirstNormalDerived>& first_normal,
             const Eigen::MatrixBase<SecondBaseDerived>& second_base,
             const Eigen::MatrixBase<SecondNormalDerived>& second_normal)
        : first_base_(detail::make_owned_point<Dim, Scalar>(first_base)),
          second_base_(detail::make_owned_point<Dim, Scalar>(second_base)),
          first_normal_(detail::make_owned_point<Dim, Scalar>(first_normal)),
          second_normal_(detail::make_owned_point<Dim, Scalar>(second_normal)) {
        validate_geometry();
    }

    Periodic(const Plane<Dim, Scalar, Index>& first,
             const Plane<Dim, Scalar, Index>& second)
        : Periodic(first.base(),
                   first.normal(),
                   second.base(),
                   second.normal()) {}

    [[nodiscard]] const Point& first_base() const noexcept {
        return first_base_;
    }

    [[nodiscard]] const Point& first_normal() const noexcept {
        return first_normal_;
    }

    [[nodiscard]] const Point& second_base() const noexcept {
        return second_base_;
    }

    [[nodiscard]] const Point& second_normal() const noexcept {
        return second_normal_;
    }

    [[nodiscard]] std::size_t dimension() const noexcept {
        return static_cast<std::size_t>(first_base_.size());
    }

    [[nodiscard]] Periodic<Dynamic, Scalar, Index> to_dynamic() const {
        return Periodic<Dynamic, Scalar, Index>(
            first_base_, first_normal_, second_base_, second_normal_);
    }

    template <int TargetDim>
    [[nodiscard]] Periodic<TargetDim, Scalar, Index> to_static() const {
        static_assert(TargetDim > 0,
                      "Target Periodic dimension must be positive.");
        if (first_base_.size() != TargetDim) {
            throw std::invalid_argument(
                "Cannot convert Periodic to requested static dimension.");
        }
        return Periodic<TargetDim, Scalar, Index>(
            first_base_, first_normal_, second_base_, second_normal_);
    }

private:
    void validate_geometry() const {
        if constexpr (IsDynamic) {
            if (first_base_.size() == 0) {
                throw std::invalid_argument("Periodic dimension must be positive.");
            }
            if (second_base_.size() != first_base_.size() ||
                first_normal_.size() != first_base_.size() ||
                second_normal_.size() != first_base_.size()) {
                throw std::invalid_argument(
                    "All periodic vectors must have the same dimension.");
            }
        }
        if (first_normal_.squaredNorm() == Scalar{0} ||
            second_normal_.squaredNorm() == Scalar{0}) {
            throw std::invalid_argument(
                "Periodic plane normals must be non-zero.");
        }
    }

    Point first_base_;
    Point second_base_;
    Point first_normal_;
    Point second_normal_;
};

/** @brief Result of the first positive boundary intersection. */
template <typename ScalarT, typename IndexT>
struct BoundaryIntersection {
    IndexT plane;
    ScalarT parameter;
};

/** @brief Plane indices grouped by boundary condition. */
template <typename IndexT>
struct BoundaryIndexGroups {
    std::vector<IndexT> periodic;
    std::vector<IndexT> neumann;
    std::vector<IndexT> dirichlet;
};

/**
 * @brief Represents a domain boundary as an intersection of oriented
 *        half-spaces.
 *
 * A boundary consists of a collection of oriented planes. Each plane normal
 * points out of the represented domain. A point lies inside the boundary if it
 * lies inside the closed half-space of every plane:
 *
 * @code
 * (point - plane.base()) dot plane.normal() <= tolerance
 * @endcode
 *
 * Consequently, the represented domain is
 *
 * @f[
 *     \Omega
 *     =
 *     \bigcap_i
 *     \left\{
 *         x :
 *         (x-b_i)\cdot n_i \le 0
 *     \right\}.
 * @f]
 *
 * An empty boundary contains every point. All planes belonging to one boundary
 * must have the same spatial dimension. Periodic planes occur in reciprocal
 * pairs and store the index of their respective partner plane.
 *
 * The class provides the following operations:
 *
 * - `size()` returns the number of planes.
 *
 * - `dimension()` returns the spatial dimension of the boundary.
 *
 * - `empty()` tests whether the boundary contains no planes.
 *
 * - `planes()` provides read-only access to the complete plane collection.
 *
 * - `bounding_box(nodes)` returns the coordinate-wise bounds of all supplied
 *   nodes and their reflections at every boundary plane.
 *
 * - `operator[](index)` provides read-only access to one plane.
 *
 * - `add(plane)` appends a Dirichlet, Neumann, or already configured periodic
 *   plane.
 *
 * - `add(periodic)` appends both planes of a periodic pair and assigns their
 *   reciprocal partner indices.
 *
 * - `contains(point, tolerance)` tests whether a point lies inside every
 *   half-space and therefore inside the represented domain.
 *
 * - `project_inside(point, tolerance, maximum_sweeps)` moves a point into the
 *   represented domain by projecting it onto violated planes. For a convex
 *   boundary, repeated sweeps are performed until the point satisfies all
 *   half-space constraints or the maximum number of sweeps is reached.
 *
 * - `first_intersection(origin, direction, tolerance)` determines the first
 *   boundary plane reached by the ray
 *   `origin + parameter * direction`. It returns the plane index together with
 *   the corresponding non-negative intersection parameter. No result is
 *   returned when the ray does not intersect the boundary in forward
 *   direction.
 *
 * - `check_nodes(nodes, tolerance)` verifies that every supplied node lies
 *   inside the represented domain and throws if at least one node lies outside.
 *
 * - `split_indices()` groups the zero-based plane indices according to their
 *   Dirichlet, Neumann, or periodic boundary condition.
 *
 * - `to_dynamic()` creates an equivalent boundary with runtime dimension.
 *
 * - `to_static()` creates an equivalent fixed-dimensional boundary after
 *   verifying that the dimensions agree.
 *
 * - `cuboid(...)` constructs an axis-aligned cuboid. Opposing faces may be
 *   periodic, while the remaining lower and upper faces may independently be
 *   marked as Neumann or Dirichlet.
 *
 * - `centered_cube(...)` constructs an axis-aligned cube centered at the
 *   origin.
 *
 * - `to_string()` returns a textual description of all planes and their
 *   boundary conditions.
 *
 * The `convex` property describes whether the planes are intended to represent
 * a convex domain. The half-space membership test itself always evaluates all
 * stored planes. Projection by repeated plane projections is principally
 * intended for convex intersections.
 *
 * Periodic partner indices are zero-based. Periodicity only records the
 * correspondence between opposing planes. Geometric reflection and projection
 * at an individual plane do not themselves perform a periodic mapping.
 */
template <int Dim,
          typename ScalarT,
          typename IndexT>
class Boundary {
public:
    using Scalar = ScalarT;
    using Index = IndexT;
    using Point = detail::BoundaryPoint<Dim, Scalar>;
    using PlaneType = Plane<Dim, Scalar, Index>;
    using DirichletType = Dirichlet<Dim, Scalar, Index>;
    using NeumannType = Neumann<Dim, Scalar, Index>;
    using PeriodicType = Periodic<Dim, Scalar, Index>;
    using Intersection = BoundaryIntersection<Scalar, Index>;
    using IndexGroups = BoundaryIndexGroups<Index>;
    /**
    * @brief Translation connecting one periodic plane with its partner.
    *
    * Translating a point on `plane` by `shift` maps it to the corresponding
    * point on `partner`. Each periodic pair occurs only once in the result of
    * `periodic_shifts()`.
    */
    struct PeriodicShiftData {
        Index plane;
        Index partner;
        Point shift;
    };

    static_assert(Dim == Dynamic || Dim > 0,
                  "Boundary dimension must be positive or Dynamic.");
    static_assert(std::is_integral_v<Index> && std::is_unsigned_v<Index>,
                  "Boundary index type must be an unsigned integer type.");

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr bool IsDynamic = (Dim == Dynamic);

    Boundary()
        : dimension_(IsDynamic ? Eigen::Index{0} : Eigen::Index{Dim}) {}

    /**
     * @brief Construct an empty boundary with an explicit runtime dimension.
     *
     * This is primarily useful for an empty dynamic boundary. Fixed-dimensional
     * boundaries require the supplied dimension to equal `Dim`.
     */
    explicit Boundary(std::size_t dimension, bool convex = true)
        : dimension_(static_cast<Eigen::Index>(dimension)),
          convex_(convex) {
        if (dimension_ == 0) {
            throw std::invalid_argument("Boundary dimension must be positive.");
        }
        if constexpr (!IsDynamic) {
            if (dimension_ != Dim) {
                throw std::invalid_argument(
                    "Explicit dimension does not match fixed Boundary dimension.");
            }
        }
    }

    explicit Boundary(std::vector<PlaneType> planes, bool convex = true)
        : planes_(std::move(planes)),
          dimension_(planes_.empty()
                         ? (IsDynamic ? Eigen::Index{0} : Eigen::Index{Dim})
                         : planes_.front().base().size()),
          convex_(convex) {
        validate_planes();
    }

    [[nodiscard]] std::size_t dimension() const noexcept {
        return static_cast<std::size_t>(dimension_);
    }

    [[nodiscard]] Index size() const noexcept {
        return static_cast<Index>(planes_.size());
    }

    [[nodiscard]] Index length() const noexcept {
        return size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return planes_.empty();
    }

    [[nodiscard]] bool convex() const noexcept {
        return convex_;
    }

    void convex(bool value) noexcept {
        convex_ = value;
    }

    [[nodiscard]] const std::vector<PlaneType>& planes() const noexcept {
        return planes_;
    }

    /**
     * @brief Bounding box of nodes and all single-plane reflections.
     *
     * This is the C++ equivalent of Julia `bounding_box(boundary, xs)`.
     * Reflections are always generated from the original node, not recursively
     * from already reflected points.
     */
    template <class Nodes>
    [[nodiscard]] std::pair<Point, Point> bounding_box(
        const Nodes& nodes) const {
        static_assert(
            std::is_same_v<typename Nodes::Scalar, Scalar>,
            "Boundary and nodes must use the same scalar type.");

        if (nodes.empty()) {
            throw std::invalid_argument(
                "Boundary bounding_box requires at least one node.");
        }

        const std::size_t node_dimension =
            static_cast<std::size_t>(nodes.dimension());
        if (dimension_ != Eigen::Index{0} &&
            node_dimension != static_cast<std::size_t>(dimension_)) {
            throw std::invalid_argument(
                "Node dimension does not match Boundary dimension.");
        }

        Point mins = detail::zero_point<Dim, Scalar>(
            static_cast<Eigen::Index>(node_dimension));
        Point maxs = mins;
        Point point = mins;
        Point reflected = mins;

        using NodeIndex = typename Nodes::Index;
        nodes.copy_node(NodeIndex{0}, mins.data());
        maxs = mins;

        for (NodeIndex node = NodeIndex{0}; node < nodes.size(); ++node) {
            nodes.copy_node(node, point.data());

            for (Eigen::Index c = 0; c < point.size(); ++c) {
                mins[c] = std::min(mins[c], point[c]);
                maxs[c] = std::max(maxs[c], point[c]);
            }

            for (const PlaneType& plane : planes_) {
                reflected = point;
                plane.reflect_in_place_unchecked(reflected);
                for (Eigen::Index c = 0; c < reflected.size(); ++c) {
                    mins[c] = std::min(mins[c], reflected[c]);
                    maxs[c] = std::max(maxs[c], reflected[c]);
                }
            }
        }

        return {std::move(mins), std::move(maxs)};
    }

    [[nodiscard]] const PlaneType& operator[](Index index) const noexcept {
        return planes_[index];
    }

    [[nodiscard]] const PlaneType& at(Index index) const {
        return planes_.at(index);
    }

    void add(PlaneType plane) {
        if (plane.is_periodic()) {
            throw std::invalid_argument(
                "Add periodic planes as a Periodic pair.");
        }
        adopt_or_check_dimension(plane.base().size());
        planes_.push_back(std::move(plane));
    }


    /** @brief Add a periodic pair and assign reciprocal zero-based indices. */
    void add(const PeriodicType& periodic) {
        adopt_or_check_dimension(periodic.first_base().size());

        const Index first_index = static_cast<Index>(planes_.size());
        const Index second_index = first_index + Index{1};

        PlaneType first = PlaneType::periodic(
            periodic.first_base(), periodic.first_normal(), second_index);
        PlaneType second = PlaneType::periodic(
            periodic.second_base(), periodic.second_normal(), first_index);

        planes_.reserve(planes_.size() + 2);
        planes_.push_back(std::move(first));
        planes_.push_back(std::move(second));
    }

    template <typename Derived>
    [[nodiscard]] bool contains(
        const Eigen::MatrixBase<Derived>& point,
        Scalar tolerance = Scalar{0}) const {
        require_dimension(point);
        return contains_unchecked(point, tolerance);
    }

    template <typename Derived>
    [[nodiscard]] Point reflect(
        const Eigen::MatrixBase<Derived>& point,
        Index plane_index) const {
        require_dimension(point);
        Point result(point);
        at(plane_index).reflect_in_place_unchecked(result);
        return result;
    }

    template <typename Derived, typename Indices>
    [[nodiscard]] Point reflect(
        const Eigen::MatrixBase<Derived>& point,
        Index local_plane_index,
        const Indices& plane_indices) const {
        return reflect(point, plane_indices[local_plane_index]);
    }

    template <typename OriginDerived, typename DirectionDerived>
    [[nodiscard]] std::optional<Intersection> first_intersection(
        const Eigen::MatrixBase<OriginDerived>& origin,
        const Eigen::MatrixBase<DirectionDerived>& direction,
        Scalar minimum_t = Scalar{0},
        Scalar parallel_tolerance = Scalar{0}) const {
        return first_intersection_if(
            origin,
            direction,
            [](Index, const PlaneType&) { return true; },
            minimum_t,
            parallel_tolerance);
    }

    template <typename OriginDerived,
              typename DirectionDerived,
              typename Predicate>
    [[nodiscard]] std::optional<Intersection> first_intersection_if(
        const Eigen::MatrixBase<OriginDerived>& origin,
        const Eigen::MatrixBase<DirectionDerived>& direction,
        Predicate&& condition,
        Scalar minimum_t = Scalar{0},
        Scalar parallel_tolerance = Scalar{0}) const {
        require_dimension(origin);
        require_dimension(direction);

        std::optional<Intersection> result;
        const Index count = size();
        for (Index index = 0; index < count; ++index) {
            const PlaneType& plane = planes_[index];
            if (!condition(index, plane)) {
                continue;
            }

            const std::optional<Scalar> parameter =
                plane.intersection_parameter_unchecked(
                    origin, direction, parallel_tolerance);
            if (!parameter || *parameter <= minimum_t) {
                continue;
            }

            if (!result || *parameter < result->parameter) {
                result = Intersection{index, *parameter};
            }
        }
        return result;
    }

    template <typename OriginDerived, typename DirectionDerived>
    [[nodiscard]] std::optional<Point> intersection_point(
        const Eigen::MatrixBase<OriginDerived>& origin,
        const Eigen::MatrixBase<DirectionDerived>& direction,
        Scalar minimum_t = Scalar{0},
        Scalar parallel_tolerance = Scalar{0}) const {
        const std::optional<Intersection> hit = first_intersection(
            origin, direction, minimum_t, parallel_tolerance);
        if (!hit) {
            return std::nullopt;
        }

        Point point(origin);
        point += hit->parameter * direction;
        return point;
    }

    template <typename OriginDerived, typename DirectionDerived>
    [[nodiscard]] bool intersection_exists(
        const Eigen::MatrixBase<OriginDerived>& origin,
        const Eigen::MatrixBase<DirectionDerived>& direction,
        Scalar containment_tolerance = Scalar{1e-5},
        Scalar parallel_tolerance = Scalar{0}) const {
        const std::optional<Point> point = intersection_point(
            origin, direction, Scalar{0}, parallel_tolerance);
        return point && contains(*point, containment_tolerance);
    }

    /**
     * @brief Iteratively project a point into a convex boundary.
     *
     * @note This is an intentional C++ extension of the Julia implementation.
     *       The Julia `project(x, B)` performs one sweep. This method performs
     *       cyclic projections onto every currently violated half-space and may
     *       repeat the sweep up to `maximum_sweeps` times. A sweep stops early
     *       when no plane changes the point. For an intersection of convex
     *       half-spaces this is more robust than a single sweep, especially when
     *       projecting onto one plane makes another constraint active again.
     *       It is not intended for non-convex boundaries.
     */
    template <typename Derived>
    [[nodiscard]] Point project_inside(
        const Eigen::MatrixBase<Derived>& point,
        std::size_t maximum_sweeps = 16,
        Scalar tolerance = Scalar{0}) const {
        require_dimension(point);
        Point result(point);

        for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
            bool changed = false;
            for (const PlaneType& plane : planes_) {
                if (plane.halfspace_value_unchecked(result) > tolerance) {
                    plane.project_in_place_unchecked(result);
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }
        return result;
    }

    template <typename Nodes>
    void check_nodes(
        const Nodes& nodes,
        Scalar tolerance = Scalar{0}) const {
        if (dimension_ != 0 &&
            static_cast<Eigen::Index>(nodes.dimension()) != dimension_) {
            throw std::invalid_argument(
                "Node-container dimension does not match Boundary dimension.");
        }

        using NodeIndex = typename Nodes::Index;
        for (NodeIndex index = NodeIndex{0}; index < nodes.size(); ++index) {
            const auto point = nodes.node(index);
            if (!contains_unchecked(point, tolerance)) {
                throw std::domain_error(
                    "Voronoi node lies outside the represented boundary.");
            }
        }
    }

    [[nodiscard]] IndexGroups split_indices() const {
        IndexGroups groups;
        groups.periodic.reserve(planes_.size());
        groups.neumann.reserve(planes_.size());
        groups.dirichlet.reserve(planes_.size());

        const Index count = size();
        for (Index index = 0; index < count; ++index) {
            const PlaneType& plane = planes_[index];
            if (plane.is_periodic()) {
                groups.periodic.push_back(index);
            } else if (plane.is_neumann()) {
                groups.neumann.push_back(index);
            } else {
                groups.dirichlet.push_back(index);
            }
        }
        return groups;
    }

    [[nodiscard]] Boundary<Dynamic, Scalar, Index> to_dynamic() const {
        using DynamicPlane = Plane<Dynamic, Scalar, Index>;
        std::vector<DynamicPlane> converted;
        converted.reserve(planes_.size());
        for (const PlaneType& plane : planes_) {
            converted.push_back(plane.to_dynamic());
        }

        if (converted.empty()) {
            if (dimension_ == 0) {
                return Boundary<Dynamic, Scalar, Index>();
            }
            return Boundary<Dynamic, Scalar, Index>(dimension(), convex_);
        }
        return Boundary<Dynamic, Scalar, Index>(
            std::move(converted), convex_);
    }

    template <int TargetDim>
    [[nodiscard]] Boundary<TargetDim, Scalar, Index> to_static() const {
        static_assert(TargetDim > 0,
                      "Target Boundary dimension must be positive.");
        if (dimension_ != TargetDim) {
            throw std::invalid_argument(
                "Cannot convert Boundary to requested static dimension.");
        }

        using StaticPlane = Plane<TargetDim, Scalar, Index>;
        std::vector<StaticPlane> converted;
        converted.reserve(planes_.size());
        for (const PlaneType& plane : planes_) {
            converted.push_back(plane.template to_static<TargetDim>());
        }
        return Boundary<TargetDim, Scalar, Index>(
            std::move(converted), convex_);
    }

    /**
     * @brief Construct an axis-aligned cuboid.
     *
     * Axes are zero-based. `periodic_axes == std::nullopt` means all axes are
     * periodic, matching the Julia default. An empty vector means no periodic
     * axes. Remaining faces are Dirichlet unless listed as Neumann.
     */
    [[nodiscard]] static Boundary cuboid(
        const Point& dimensions,
        const Point& offset,
        std::optional<std::vector<Index>> periodic_axes = std::nullopt,
        const std::vector<Index>& neumann_lower_axes = {},
        const std::vector<Index>& neumann_upper_axes = {}) {
        if (dimensions.size() == 0 || dimensions.size() != offset.size()) {
            throw std::invalid_argument("Cuboid dimensions are inconsistent.");
        }
        if ((dimensions.array() <= Scalar{0}).any()) {
            throw std::invalid_argument(
                "Cuboid edge lengths must be positive.");
        }

        const Index dimension = static_cast<Index>(dimensions.size());
        Boundary result;
        if constexpr (IsDynamic) {
            result.dimension_ = dimensions.size();
        }

        if (!periodic_axes) {
            periodic_axes.emplace();
            periodic_axes->reserve(dimension);
            for (Index axis = 0; axis < dimension; ++axis) {
                periodic_axes->push_back(axis);
            }
        }

        validate_axes(*periodic_axes, dimension, "periodic axis");
        validate_axes(neumann_lower_axes, dimension, "lower Neumann axis");
        validate_axes(neumann_upper_axes, dimension, "upper Neumann axis");

        for (Eigen::Index coordinate = 0;
             coordinate < dimensions.size();
             ++coordinate) {
            const Index axis = static_cast<Index>(coordinate);
            Point unit = detail::zero_point<Dim, Scalar>(dimensions.size());
            unit[coordinate] = Scalar{1};

            Point upper_base(offset);
            upper_base[coordinate] += dimensions[coordinate];

            if (has_axis(*periodic_axes, axis)) {
                result.add(PeriodicType(upper_base, offset, unit));
                continue;
            }

            if (has_axis(neumann_upper_axes, axis)) {
                result.add(NeumannType(upper_base, unit));
            } else {
                result.add(DirichletType(upper_base, unit));
            }

            if (has_axis(neumann_lower_axes, axis)) {
                result.add(NeumannType(offset, -unit));
            } else {
                result.add(DirichletType(offset, -unit));
            }
        }
        return result;
    }

    [[nodiscard]] static Boundary cuboid(
        const Point& dimensions,
        std::optional<std::vector<Index>> periodic_axes = std::nullopt,
        const std::vector<Index>& neumann_lower_axes = {},
        const std::vector<Index>& neumann_upper_axes = {}) {
        return cuboid(
            dimensions,
            detail::zero_point<Dim, Scalar>(dimensions.size()),
            std::move(periodic_axes),
            neumann_lower_axes,
            neumann_upper_axes);
    }

    [[nodiscard]] static Boundary centered_cube(
        std::size_t dimension,
        Scalar size) {
        if (size <= Scalar{0}) {
            throw std::invalid_argument("Cube size must be positive.");
        }
        if constexpr (!IsDynamic) {
            if (dimension != static_cast<std::size_t>(Dim)) {
                throw std::invalid_argument(
                    "Cube dimension does not match fixed Boundary dimension.");
            }
        }

        Point dimensions;
        Point offset;
        if constexpr (IsDynamic) {
            const Eigen::Index eigen_dimension =
                static_cast<Eigen::Index>(dimension);
            dimensions = Point::Zero(eigen_dimension);
            offset = Point::Zero(eigen_dimension);
        } else {
            dimensions.setZero();
            offset.setZero();
        }

        dimensions.setConstant(size);
        offset.setConstant(-size / Scalar{2});
        return cuboid(dimensions, offset);
    }

    template <int D = Dim, std::enable_if_t<D != Dynamic, int> = 0>
    [[nodiscard]] static Boundary centered_cube(Scalar size) {
        Point dimensions = Point::Constant(size);
        Point offset = Point::Constant(-size / Scalar{2});
        return cuboid(dimensions, offset);
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream stream;
        stream << "BOUNDARY in " << dimension() << " dimensions with "
               << planes_.size() << " planes:\n";

        for (Index index = 0; index < size(); ++index) {
            const PlaneType& plane = planes_[index];
            stream << "    " << index << ": base=["
                   << plane.base().transpose() << "], normal=["
                   << plane.normal().transpose() << "]; ";

            if (plane.is_dirichlet()) {
                stream << "Dirichlet";
            } else if (plane.is_neumann()) {
                stream << "Neumann";
            } else {
                stream << "periodic with partner "
                       << *plane.periodic_partner();
            }
            stream << '\n';
        }
        return stream.str();
    }
    /**
    * @brief Return the translation from a periodic plane to its partner.
    *
    * The base points of paired periodic planes are interpreted as corresponding
    * points. Therefore, the returned translation is
    *
    * @code
    * partner.base() - plane.base()
    * @endcode
    *
    * The translation is not generally parallel to either plane normal.
    *
    * @throws std::invalid_argument if the selected plane is not periodic.
    */
    [[nodiscard]] Point periodic_shift(Index plane_index) const {
        const PlaneType& plane = at(plane_index);

        if (!plane.is_periodic()) {
            throw std::invalid_argument(
                "Periodic shift requested for a non-periodic plane.");
        }

        const PlaneType& partner = at(*plane.periodic_partner());
        return Point(partner.base() - plane.base());
    }
    /**
    * @brief Return all periodic plane pairs and their translations.
    *
    * Every periodic pair is returned exactly once. The entry contains the index
    * of the first plane, the index of its partner, and the translation from the
    * first plane to the partner plane.
    *
    * The reverse translation is the negative of the stored shift.
    */
    [[nodiscard]] std::vector<PeriodicShiftData>
    periodic_shifts() const {
        std::vector<PeriodicShiftData> result;
        result.reserve(planes_.size() / 2);

        const Index count = size();
        for (Index plane_index = 0; plane_index < count; ++plane_index) {
            const PlaneType& plane = planes_[plane_index];

            if (!plane.is_periodic()) {
                continue;
            }

            const Index partner_index = *plane.periodic_partner();

            // Return each reciprocal pair only once.
            if (plane_index > partner_index) {
                continue;
            }

            result.push_back(PeriodicShiftData{
                plane_index,
                partner_index,
                periodic_shift(plane_index)
            });
        }

        return result;
    }
private:
    template <typename Derived>
    void require_dimension(const Eigen::MatrixBase<Derived>& point) const {
        static_assert(Derived::IsVectorAtCompileTime,
                      "Boundary coordinates must be vector expressions.");
        if (dimension_ != 0 && point.size() != dimension_) {
            throw std::invalid_argument(
                "Point dimension does not match Boundary dimension.");
        }
    }

    template <typename Derived>
    [[nodiscard]] bool contains_unchecked(
        const Eigen::MatrixBase<Derived>& point,
        Scalar tolerance) const {
        for (const PlaneType& plane : planes_) {
            if (plane.halfspace_value_unchecked(point) > tolerance) {
                return false;
            }
        }
        return true;
    }

    void adopt_or_check_dimension(Eigen::Index candidate) {
        if constexpr (IsDynamic) {
            if (dimension_ == 0) {
                dimension_ = candidate;
            } else if (candidate != dimension_) {
                throw std::invalid_argument(
                    "Plane dimension does not match Boundary dimension.");
            }
        } else {
            (void)candidate;
        }
    }

    void validate_planes() const {
        if (planes_.empty()) {
            return;
        }

        const Index count = size();
        for (Index index = 0; index < count; ++index) {
            const PlaneType& plane = planes_[index];
            if constexpr (IsDynamic) {
                if (plane.base().size() != dimension_) {
                    throw std::invalid_argument(
                        "Plane dimension does not match Boundary dimension.");
                }
            }
            if (!plane.is_periodic()) {
                continue;
            }

            const Index partner = *plane.periodic_partner();
            if (partner >= count || partner == index) {
                throw std::invalid_argument(
                    "Invalid periodic partner index.");
            }

            const PlaneType& partner_plane = planes_[partner];
            if (!partner_plane.is_periodic() ||
                partner_plane.periodic_partner() != index) {
                throw std::invalid_argument(
                    "Periodic partner relation must be reciprocal.");
            }
        }
    }

    [[nodiscard]] static bool has_axis(
        const std::vector<Index>& axes,
        Index axis) {
        return std::find(axes.begin(), axes.end(), axis) != axes.end();
    }

    static void validate_axes(
        const std::vector<Index>& axes,
        Index dimension,
        const char* description) {
        for (Index axis : axes) {
            if (axis >= dimension) {
                throw std::invalid_argument(
                    std::string(description) +
                    " is outside cuboid dimension.");
            }
        }
    }

    std::vector<PlaneType> planes_;
    Eigen::Index dimension_;
    bool convex_ = true;
};

template <int Dim, typename Scalar, typename Index>
std::ostream& operator<<(
    std::ostream& stream,
    const Boundary<Dim, Scalar, Index>& boundary) {
    return stream << boundary.to_string();
}

// C++17 deduction guides preserve compile-time dimension for static inputs.
template <typename BaseDerived, typename NormalDerived>
Plane(const Eigen::MatrixBase<BaseDerived>&,
      const Eigen::MatrixBase<NormalDerived>&,
      BoundaryCondition)
    -> Plane<BaseDerived::SizeAtCompileTime,
             typename BaseDerived::Scalar,
             std::size_t>;

template <typename BaseDerived, typename NormalDerived>
Dirichlet(const Eigen::MatrixBase<BaseDerived>&,
          const Eigen::MatrixBase<NormalDerived>&)
    -> Dirichlet<BaseDerived::SizeAtCompileTime,
                 typename BaseDerived::Scalar,
                 std::size_t>;

template <typename BaseDerived, typename NormalDerived>
Neumann(const Eigen::MatrixBase<BaseDerived>&,
        const Eigen::MatrixBase<NormalDerived>&)
    -> Neumann<BaseDerived::SizeAtCompileTime,
               typename BaseDerived::Scalar,
               std::size_t>;

template <typename FirstBaseDerived,
          typename SecondBaseDerived,
          typename FirstNormalDerived>
Periodic(const Eigen::MatrixBase<FirstBaseDerived>&,
         const Eigen::MatrixBase<SecondBaseDerived>&,
         const Eigen::MatrixBase<FirstNormalDerived>&)
    -> Periodic<FirstBaseDerived::SizeAtCompileTime,
                typename FirstBaseDerived::Scalar,
                std::size_t>;

} // namespace highvoronoi


