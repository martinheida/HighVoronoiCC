
#pragma once

#include <Eigen/Core>

#include <type_traits>

namespace highvoronoi {

/**
 * @file point.hpp
 * @brief Eigen-based point types used by HighVoronoi.
 */

/**
 * @brief HighVoronoi marker for a runtime-defined dimension.
 *
 * This deliberately hides Eigen's `Eigen::Dynamic` from the public
 * HighVoronoi API. Users can write, for example,
 *
 * @code{.cpp}
 * highvoronoi::VoronoiNodes<double, highvoronoi::Dynamic> nodes(1000, 5);
 * @endcode
 *
 * without referring to the Eigen namespace.
 */
inline constexpr int Dynamic = Eigen::Dynamic;

// ============================================================================
// Owning point types
// ============================================================================

/**
 * @brief Owning point with compile-time dimension.
 *
 * @tparam Scalar Scalar coordinate type, for example `double` or `float`.
 * @tparam Dim Number of coordinates known at compile time.
 *
 * @par Example
 * @code{.cpp}
 * highvoronoi::StaticPoint<double, 3> p;
 * p << 1.0, 2.0, 3.0;
 * @endcode
 */
template <typename Scalar, int Dim>
using StaticPoint = Eigen::Matrix<Scalar, Dim, 1>;

/**
 * @brief Owning point with runtime dimension.
 *
 * @tparam Scalar Scalar coordinate type.
 *
 * @par Example
 * @code{.cpp}
 * highvoronoi::DynamicPoint<double> p(4);
 * p << 1.0, 2.0, 3.0, 4.0;
 * @endcode
 */
template <typename Scalar>
using DynamicPoint = Eigen::Matrix<Scalar, Dynamic, 1>;

/**
 * @brief Owning HighVoronoi point selected only by scalar and dimension.
 *
 * Fixed dimensions use StaticPoint; highvoronoi::Dynamic uses DynamicPoint.
 * NodePoint and VertexPoint aliases throughout the mesh layer are defined from
 * this one rule.
 */
template <typename Scalar, int Dim>
using PointType = std::conditional_t<
    Dim == Dynamic,
    DynamicPoint<Scalar>,
    StaticPoint<Scalar, Dim>>;

// ============================================================================
// Read-only views of raw memory
// ============================================================================

/**
 * @brief Non-owning read-only view of `Dim` contiguous scalar values.
 *
 * The view stores no copy of the coordinates and never frees the referenced
 * memory. The pointed-to storage must remain valid while the view is used.
 *
 * @par Example
 * @code{.cpp}
 * double raw[3] = {1.0, 2.0, 3.0};
 * highvoronoi::StaticPointView<double, 3> p(raw);
 *
 * raw[1] = 42.0; // p now observes 42.0 as its second coordinate.
 * @endcode
 */
template <typename Scalar, int Dim>
using StaticPointView = Eigen::Map<const StaticPoint<Scalar, Dim>>;

/**
 * @brief Non-owning read-only view of a runtime-sized contiguous scalar array.
 *
 * @par Example
 * @code{.cpp}
 * double* raw = get_coordinates();
 * std::size_t dim = get_dimension();
 * highvoronoi::DynamicPointView<double> p(raw, dim);
 * @endcode
 */
template <typename Scalar>
using DynamicPointView = Eigen::Map<const DynamicPoint<Scalar>>;

// ============================================================================
// Read-only references to existing Eigen vectors / expressions
// ============================================================================

/** @brief Read-only reference to an existing fixed-size Eigen-compatible point. */
template <typename Scalar, int Dim>
using StaticPointRef = Eigen::Ref<const StaticPoint<Scalar, Dim>>;

/** @brief Read-only reference to an existing runtime-sized Eigen-compatible point. */
template <typename Scalar>
using DynamicPointRef = Eigen::Ref<const DynamicPoint<Scalar>>;

} // namespace highvoronoi

