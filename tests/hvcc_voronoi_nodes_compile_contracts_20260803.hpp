#pragma once

#include <type_traits>
#include <utility>

#include "hvcc_voronoi_nodes_fixtures_20260803.hpp"

namespace voronoi_nodes_test::type_checks {

/** Return type of operator[] for a const node container. */
template <typename Nodes>
using ReadResult =
    decltype(std::declval<const Nodes&>()[Index{0}]);

// ============================================================================
// Ordinary node containers
// ============================================================================

static_assert(
    StoredNodes::AccessMode == highvoronoi::NodeAccessMode::Stored,
    "StoredNodes must use stored access.");

static_assert(
    std::is_same_v<ReadResult<StoredNodes>, PointView<2>>,
    "StoredNodes::operator[] must return a zero-copy point view.");

static_assert(
    ComputedGridNodes::AccessMode == highvoronoi::NodeAccessMode::Computed,
    "ComputedGridNodes must use computed access.");

static_assert(
    std::is_same_v<ReadResult<ComputedGridNodes>, Point2>,
    "ComputedGridNodes::operator[] must return an owning point.");

static_assert(
    HybridExampleNodes::AccessMode == highvoronoi::NodeAccessMode::Hybrid,
    "HybridExampleNodes must use hybrid access.");

static_assert(
    std::is_same_v<ReadResult<HybridExampleNodes>, Handle2>,
    "HybridExampleNodes::operator[] must return NodeHandle.");

// ============================================================================
// Extended node containers
// ============================================================================

static_assert(
    StoredExtendedNodes::AccessMode == highvoronoi::NodeAccessMode::Stored,
    "Extending stored nodes must preserve stored access.");

static_assert(
    std::is_same_v<ReadResult<StoredExtendedNodes>, PointView<2>>,
    "Stored extended nodes must return point views.");

static_assert(
    ComputedExtendedNodes::AccessMode == highvoronoi::NodeAccessMode::Hybrid,
    "Extending computed nodes must produce hybrid access: inner nodes are "
    "computed, mirror slots are stored.");

static_assert(
    std::is_same_v<ReadResult<ComputedExtendedNodes>, Handle2>,
    "Computed extended nodes must return NodeHandle.");

static_assert(
    HybridExtendedNodes::AccessMode == highvoronoi::NodeAccessMode::Hybrid,
    "Extending hybrid nodes must preserve hybrid access.");

static_assert(
    std::is_same_v<ReadResult<HybridExtendedNodes>, Handle2>,
    "Hybrid extended nodes must return NodeHandle.");

// The precomputed variants expose the same access modes as the corresponding
// active-cell variants; only their mirror-storage strategy differs.
static_assert(
    StoredPrecomputedNodes::AccessMode == highvoronoi::NodeAccessMode::Stored);
static_assert(
    ComputedPrecomputedNodes::AccessMode == highvoronoi::NodeAccessMode::Hybrid);
static_assert(
    HybridPrecomputedNodes::AccessMode == highvoronoi::NodeAccessMode::Hybrid);

} // namespace voronoi_nodes_test::type_checks
