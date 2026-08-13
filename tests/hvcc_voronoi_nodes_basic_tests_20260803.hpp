#pragma once

#include <iostream>

#include "hvcc_test_support_voronoi_nodes_20260803.hpp"
#include "hvcc_voronoi_nodes_fixtures_20260803.hpp"

namespace voronoi_nodes_test {

using hvtest::check;
using hvtest::check_point_2d;
using hvtest::close;

// ============================================================================
// Ordinary node-access branches
// ============================================================================

/**
 * Verify the stored-access branch.
 *
 * This test is useful because stored nodes are expected to expose their
 * existing memory without copying, while the common base interface must still
 * be able to materialize an independent point when requested.
 */
inline void test_stored_branch() {
    std::cout << "  Read stored points as zero-copy views and through the "
                 "common copy interface.\n";

    StoredNodes nodes = make_stored_nodes();
    const auto point = nodes[Index{1}];

    check_point_2d(
        point,
        2.0,
        3.0,
        "operator[] reads the stored node at index 1");

    check(
        point.data() == nodes.stable_node_data(Index{1}),
        "the returned view points into the original node storage");

    Point2 target;
    nodes.copy_node(Index{2}, target);
    check_point_2d(
        target,
        3.0,
        4.0,
        "copy_node writes index 2 into caller-owned storage");

    AbstractNodes<2>& abstract_nodes = nodes;
    const auto owning = abstract_nodes.node(Index{3});
    check_point_2d(
        owning,
        4.0,
        5.0,
        "the common base interface materializes an owning point");
}

/**
 * Verify stored nodes with runtime dimension.
 *
 * This test is useful because dynamic points must preserve their runtime size
 * both when returned as views and when copied into an initially empty target.
 */
inline void test_dynamic_stored_branch() {
    std::cout << "  Preserve the runtime dimension in dynamic views and "
                 "copied targets.\n";

    DynamicStoredNodes nodes(Index{2}, Index{3});

    DynamicPoint point(3);
    point << 1.0, 2.0, 3.0;
    nodes.set(Index{0}, point);

    point << 4.0, 5.0, 6.0;
    nodes.set(Index{1}, point);

    check(
        nodes[Index{0}].size() == 3,
        "operator[] returns a three-dimensional dynamic view");

    check(
        close(nodes[Index{1}][Index{2}], 6.0),
        "the dynamic view reads the final coordinate correctly");

    DynamicPoint target;
    nodes.copy_node(Index{0}, target);

    check(
        target.size() == 3,
        "copy_node resizes a dynamic target to the node dimension");

    check(
        close(target[Index{1}], 2.0),
        "copy_node preserves dynamic-node coordinates");
}

/**
 * Verify the computed-access branch.
 *
 * This test is useful because each operator[] result must own its coordinates.
 * A later node computation must therefore not overwrite a previously returned
 * point, which would happen with an unsafe shared scratch buffer.
 */
inline void test_computed_branch() {
    std::cout << "  Compute points on demand and verify independent return "
                 "values.\n";

    ComputedGridNodes nodes = make_computed_nodes();
    const auto first = nodes[Index{0}];
    const auto last = nodes[Index{3}];

    check_point_2d(
        first,
        1.0,
        2.0,
        "index 0 is computed as (1, 2)");

    check_point_2d(
        last,
        2.0,
        3.0,
        "index 3 is computed as (2, 3)");

    check_point_2d(
        first,
        1.0,
        2.0,
        "the first returned value survives a later computation");

    Point2 target;
    nodes.copy_node(Index{2}, target.data());
    check_point_2d(
        target,
        1.0,
        3.0,
        "copy_node computes directly into caller-owned storage");
}

/**
 * Verify the hybrid-access branch.
 *
 * This test is useful because a single container must distinguish stable
 * stored points from points calculated on demand. NodeHandle must borrow in
 * the first case and own its coordinates in the second case.
 */
inline void test_hybrid_branch() {
    std::cout << "  Read one stored and one computed node through the same "
                 "NodeHandle interface.\n";

    HybridExampleNodes nodes = make_hybrid_nodes();
    const Handle2 stored = nodes[Index{0}];
    const Handle2 computed = nodes[Index{1}];

    check(
        !stored.owns_storage(),
        "an explicitly stored node produces a borrowing handle");

    check(
        computed.owns_storage(),
        "an on-demand node produces an owning handle");

    check_point_2d(
        stored,
        1.0,
        2.0,
        "the borrowing handle reads the stored node");

    check_point_2d(
        computed,
        2.0,
        3.0,
        "the owning handle contains the computed node");

    const auto computed_view = computed.view();
    check_point_2d(
        computed_view,
        2.0,
        3.0,
        "NodeHandle::view exposes the same coordinates");

    Point2 target;
    stored.copy_to(target.data());
    check_point_2d(
        target,
        1.0,
        2.0,
        "NodeHandle::copy_to writes into caller-owned storage");
}

// ============================================================================
// Common-interface integration and input validation
// ============================================================================

/**
 * Verify Boundary::check_nodes with non-stored node containers.
 *
 * This test is useful because Boundary must use the universal node()/copy
 * interface rather than assuming that every node container exposes raw
 * pointers.
 */
inline void test_boundary_check_uses_common_interface() {
    std::cout << "  Validate computed and hybrid nodes through Boundary's "
                 "common node interface.\n";

    const Boundary<2> boundary = make_boundary();

    bool computed_was_accepted = true;
    try {
        boundary.check_nodes(make_computed_nodes());
    } catch (...) {
        computed_was_accepted = false;
    }

    check(
        computed_was_accepted,
        "Boundary::check_nodes accepts a computed node container");

    bool hybrid_was_accepted = true;
    try {
        boundary.check_nodes(make_hybrid_nodes());
    } catch (...) {
        hybrid_was_accepted = false;
    }

    check(
        hybrid_was_accepted,
        "Boundary::check_nodes accepts a hybrid node container");
}

} // namespace voronoi_nodes_test
