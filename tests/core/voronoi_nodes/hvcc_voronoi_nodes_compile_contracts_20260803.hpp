#pragma once

#include <type_traits>
#include <utility>

#include "hvcc_voronoi_nodes_fixtures_20260803.hpp"

namespace voronoi_nodes_test::type_checks {

/** Return type of operator[] for a const node provider. */
template <typename Nodes>
using ReadResult =
    decltype(std::declval<const Nodes&>()[Index{0}]);

// Every provider has exactly the same owning point-returning convenience API.
static_assert(std::is_same_v<ReadResult<StoredNodes>, Point2>);
static_assert(std::is_same_v<ReadResult<ComputedGridNodes>, Point2>);
static_assert(std::is_same_v<ReadResult<MixedExampleNodes>, Point2>);

static_assert(std::is_same_v<ReadResult<StoredExtendedNodes>, Point2>);
static_assert(std::is_same_v<ReadResult<ComputedExtendedNodes>, Point2>);
static_assert(std::is_same_v<ReadResult<MixedExtendedNodes>, Point2>);

static_assert(std::is_same_v<ReadResult<StoredPrecomputedNodes>, Point2>);
static_assert(std::is_same_v<ReadResult<ComputedPrecomputedNodes>, Point2>);
static_assert(std::is_same_v<ReadResult<MixedPrecomputedNodes>, Point2>);

// The polymorphic contract itself also exposes the same point type.
static_assert(std::is_same_v<typename AbstractNodes<2>::Point, Point2>);
static_assert(std::is_same_v<
    typename highvoronoi::AbstractVoronoiNodes<
        Scalar,
        highvoronoi::Dynamic,
        Index>::Point,
    DynamicPoint>);

} // namespace voronoi_nodes_test::type_checks
