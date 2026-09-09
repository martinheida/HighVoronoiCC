#pragma once

#include <iostream>

#include "hvcc_test_support_voronoi_nodes_20260803.hpp"
#include "hvcc_voronoi_nodes_fixtures_20260803.hpp"

namespace voronoi_nodes_test {

using hvtest::check;
using hvtest::check_point_2d;
using hvtest::close;

// ============================================================================
// Ordinary node providers
// ============================================================================

/** Verify explicitly stored nodes through the unified owning-point interface. */
inline void test_stored_branch() {
    std::cout << "  Read stored points through the common owning-point and "
                 "caller-owned copy interfaces.\n";

    StoredNodes nodes = make_stored_nodes();
    const Point2 point = nodes[Index{1}];

    check_point_2d(
        point,
        2.0,
        3.0,
        "operator[] reads the stored node at index 1");

    const Scalar* stable = nodes.stable_node_data(Index{1});
    check(
        close(stable[0], 2.0) && close(stable[1], 3.0),
        "stable_node_data remains available as a concrete stored-node optimization");

    check(
        close(nodes.get_data(Index{1}, Index{0}), 2.0) &&
            close(nodes.get_data(Index{1}, Index{1}), 3.0),
        "get_data reads stored coordinates without materializing a point");

    Point2 target;
    nodes.copy_node(Index{2}, target);
    check_point_2d(
        target,
        3.0,
        4.0,
        "copy_node writes index 2 into caller-owned storage");

    AbstractNodes<2>& abstract_nodes = nodes;
    const Point2 owning = abstract_nodes.node(Index{3});
    check_point_2d(
        owning,
        4.0,
        5.0,
        "the common base interface returns the same owning point type");
}

/** Verify stored nodes with runtime dimension. */
inline void test_dynamic_stored_branch() {
    std::cout << "  Preserve the runtime dimension in owning results and "
                 "caller-owned targets.\n";

    DynamicStoredNodes nodes(Index{2}, Index{3});

    DynamicPoint point(3);
    point << 1.0, 2.0, 3.0;
    nodes.set(Index{0}, point);

    point << 4.0, 5.0, 6.0;
    nodes.set(Index{1}, point);

    check(
        nodes[Index{0}].size() == 3,
        "operator[] returns a three-dimensional dynamic owning point");

    check(
        close(nodes[Index{1}][Index{2}], 6.0),
        "the dynamic owning point reads the final coordinate correctly");

    DynamicPoint target;
    nodes.copy_node(Index{0}, target);

    check(
        target.size() == 3,
        "copy_node resizes a dynamic target to the node dimension");

    check(
        close(target[Index{1}], 2.0),
        "copy_node preserves dynamic-node coordinates");
}

/** Verify a provider computing all coordinates on demand. */
inline void test_computed_branch() {
    std::cout << "  Compute points on demand and verify independent owning "
                 "return values.\n";

    ComputedGridNodes nodes = make_computed_nodes();
    const Point2 first = nodes[Index{0}];
    const Point2 last = nodes[Index{3}];

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

    check(
        close(nodes.get_data(Index{2}, Index{0}), 1.0) &&
            close(nodes.get_data(Index{2}, Index{1}), 3.0),
        "get_data computes one coordinate directly");

    Point2 target;
    nodes.copy_node(Index{2}, target.data());
    check_point_2d(
        target,
        1.0,
        3.0,
        "copy_node computes directly into caller-owned storage");
}

/**
 * Verify a provider whose implementation internally mixes stored and computed
 * nodes. The generic interface deliberately exposes no such distinction.
 */
inline void test_mixed_branch() {
    std::cout << "  Read internally stored and computed nodes through one "
                 "identical owning-point interface.\n";

    MixedExampleNodes nodes = make_mixed_nodes();
    const Point2 stored = nodes[Index{0}];
    const Point2 computed = nodes[Index{1}];

    check_point_2d(
        stored,
        1.0,
        2.0,
        "operator[] reads the internally stored node");

    check_point_2d(
        computed,
        2.0,
        3.0,
        "operator[] reads the internally computed node");

    check_point_2d(
        stored,
        1.0,
        2.0,
        "both results are independent owning points");

    Point2 target;
    nodes.copy_node(Index{0}, target.data());
    check_point_2d(
        target,
        1.0,
        2.0,
        "copy_node uses the same caller-owned path for mixed providers");

    check(
        close(nodes.get_data(Index{1}, Index{1}), 3.0),
        "get_data is independent of the provider's internal storage strategy");
}

// ============================================================================
// Common-interface integration and input validation
// ============================================================================

/** Verify Boundary::check_nodes with non-stored and internally mixed providers. */
inline void test_boundary_check_uses_common_interface() {
    std::cout << "  Validate computed and internally mixed nodes through "
                 "Boundary's common node interface.\n";

    const Boundary<2> boundary = make_boundary();

    bool computed_was_accepted = true;
    try {
        boundary.check_nodes(make_computed_nodes());
    } catch (...) {
        computed_was_accepted = false;
    }

    check(
        computed_was_accepted,
        "Boundary::check_nodes accepts a computed node provider");

    bool mixed_was_accepted = true;
    try {
        boundary.check_nodes(make_mixed_nodes());
    } catch (...) {
        mixed_was_accepted = false;
    }

    check(
        mixed_was_accepted,
        "Boundary::check_nodes accepts an internally mixed node provider");
}

} // namespace voronoi_nodes_test
