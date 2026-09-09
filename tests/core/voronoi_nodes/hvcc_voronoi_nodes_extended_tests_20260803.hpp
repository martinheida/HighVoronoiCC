#pragma once

#include <iostream>
#include <stdexcept>

#include "hvcc_test_support_voronoi_nodes_20260803.hpp"
#include "hvcc_voronoi_nodes_fixtures_20260803.hpp"

namespace voronoi_nodes_test {

using hvtest::check;
using hvtest::check_point_2d;
using hvtest::check_throws;

// ============================================================================
// Extended nodes
// ============================================================================

template <typename Extended>
IndexVector all_mirror_indices(const Extended& nodes) {
    IndexVector result;
    result.reserve(nodes.mirror_count());

    for (Index plane = Index{0}; plane < nodes.mirror_count(); ++plane) {
        result.push_back(nodes.mirror_index(plane));
    }

    return result;
}

/** Verify ExtendedNodes built from stored base nodes. */
inline void test_extended_stored_branch() {
    std::cout << "  Extend four stored nodes by two mirror slots, activate "
                 "cell 1 and read both reflections.\n";

    StoredExtendedNodes nodes(make_stored_nodes(), make_boundary());

    check(
        nodes.inner_size() == Index{4},
        "inner_size reports the four original nodes");

    check(
        nodes.mirror_count() == Index{2},
        "mirror_count reports one slot per boundary plane");

    check(
        nodes.size() == Index{6},
        "size equals four inner nodes plus two mirror nodes");

    const Point2 inner = nodes[Index{1}];
    check_point_2d(
        inner,
        2.0,
        3.0,
        "operator[] reads inner node 1 before activation");

    const Scalar* stable_inner =
        nodes.inner_nodes().stable_node_data(Index{1});
    check(
        hvtest::close(stable_inner[0], 2.0) &&
            hvtest::close(stable_inner[1], 3.0),
        "the concrete stored inner provider still exposes stable_node_data");

    const IndexVector neighbours = all_mirror_indices(nodes);
    check(
        neighbours.size() == nodes.mirror_count(),
        "activate_cell receives exactly one global index per mirror slot");

    nodes.activate_cell(Index{1}, neighbours);

    const Index x_mirror = nodes.mirror_index(Index{0});
    const Index y_mirror = nodes.mirror_index(Index{1});

    const Point2 reflected_x = nodes[x_mirror];
    const Point2 reflected_y = nodes[y_mirror];

    check_point_2d(
        reflected_x,
        -2.0,
        3.0,
        "operator[] returns the reflection of cell 1 at x = 0");

    check_point_2d(
        reflected_y,
        2.0,
        -3.0,
        "operator[] returns the reflection of cell 1 at y = 0");

    check_point_2d(
        nodes[Index{1}],
        2.0,
        3.0,
        "activating mirror nodes does not alter the original cell node");

    Point2 copied_mirror;
    nodes.copy_node(y_mirror, copied_mirror);
    check_point_2d(
        copied_mirror,
        2.0,
        -3.0,
        "copy_node reads an activated mirror into caller-owned storage");
}

/** Verify ExtendedNodes built from computed base nodes. */
inline void test_extended_computed_base() {
    std::cout << "  Extend computed nodes while keeping the same owning-point "
                 "interface for inner and mirror nodes.\n";

    ComputedExtendedNodes nodes(make_computed_nodes(), make_boundary());

    check(
        nodes.inner_size() == Index{4} &&
            nodes.mirror_count() == Index{2} &&
            nodes.size() == Index{6},
        "computed-base ExtendedNodes reports four inner and two mirror nodes");

    const Point2 original = nodes[Index{3}];
    check_point_2d(
        original,
        2.0,
        3.0,
        "operator[] computes inner node 3 as (2, 3)");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{3}, neighbours);

    const Point2 reflected_x = nodes[nodes.mirror_index(Index{0})];
    const Point2 reflected_y = nodes[nodes.mirror_index(Index{1})];

    check_point_2d(
        reflected_x,
        -2.0,
        3.0,
        "operator[] returns the x-reflection of computed cell 3");

    check_point_2d(
        reflected_y,
        2.0,
        -3.0,
        "operator[] returns the y-reflection of computed cell 3");

    check_point_2d(
        original,
        2.0,
        3.0,
        "the owning inner-node result remains valid after mirror access");
}

/** Verify ExtendedNodes built from an internally mixed base provider. */
inline void test_extended_mixed_base() {
    std::cout << "  Extend an internally mixed provider without exposing its "
                 "storage strategy through the type system.\n";

    MixedExtendedNodes nodes(make_mixed_nodes(), make_boundary());

    const Point2 stored_inner = nodes[Index{0}];
    const Point2 computed_inner = nodes[Index{1}];

    check_point_2d(
        stored_inner,
        1.0,
        2.0,
        "operator[] reads the internally stored inner node");

    check_point_2d(
        computed_inner,
        2.0,
        3.0,
        "operator[] reads the internally computed inner node");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{1}, neighbours);

    const Point2 reflected_x = nodes[nodes.mirror_index(Index{0})];
    const Point2 reflected_y = nodes[nodes.mirror_index(Index{1})];

    check_point_2d(
        reflected_x,
        -2.0,
        3.0,
        "operator[] returns the correct mixed-provider x-reflection");

    check_point_2d(
        reflected_y,
        2.0,
        -3.0,
        "operator[] returns the correct mixed-provider y-reflection");
}

// ============================================================================
// Precomputed extended nodes
// ============================================================================

/** Verify precomputed reflections for stored base nodes. */
inline void test_precomputed_stored_base() {
    std::cout << "  Precompute 4 x 2 stored reflections and activate the two "
                 "reflections of cell 3.\n";

    StoredPrecomputedNodes nodes(make_stored_nodes(), make_boundary());

    check(
        nodes.precomputed_nodes().size() == Index{8},
        "four cells times two planes produce eight precomputed reflections");

    check_point_2d(
        nodes.reflected_node(Index{3}, Index{1}),
        4.0,
        -5.0,
        "reflected_node(3, 1) addresses the expected precomputed value");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{3}, neighbours);

    check_point_2d(
        nodes[nodes.mirror_index(Index{0})],
        -4.0,
        5.0,
        "operator[] exposes the activated precomputed x-reflection");

    check_point_2d(
        nodes[nodes.mirror_index(Index{1})],
        4.0,
        -5.0,
        "operator[] exposes the activated precomputed y-reflection");
}

/** Verify precomputed reflections for computed base nodes. */
inline void test_precomputed_computed_base() {
    std::cout << "  Precompute reflections of computed nodes and verify the "
                 "unified owning-point access semantics.\n";

    ComputedPrecomputedNodes nodes(make_computed_nodes(), make_boundary());

    const Point2 inner = nodes[Index{3}];
    check_point_2d(
        inner,
        2.0,
        3.0,
        "a computed inner node uses the common owning point type");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{3}, neighbours);

    const Point2 mirror = nodes[nodes.mirror_index(Index{0})];
    check_point_2d(
        mirror,
        -2.0,
        3.0,
        "the precomputed x-reflection of computed cell 3 is correct");
}

/** Verify precomputed reflections for an internally mixed base provider. */
inline void test_precomputed_mixed_base() {
    std::cout << "  Precompute reflections of internally mixed nodes without "
                 "exposing separate access modes.\n";

    MixedPrecomputedNodes nodes(make_mixed_nodes(), make_boundary());

    const Point2 stored_inner = nodes[Index{0}];
    const Point2 computed_inner = nodes[Index{1}];

    check_point_2d(
        stored_inner,
        1.0,
        2.0,
        "the internally stored node uses the common owning point type");

    check_point_2d(
        computed_inner,
        2.0,
        3.0,
        "the internally computed node uses the common owning point type");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{1}, neighbours);

    const Point2 mirror = nodes[nodes.mirror_index(Index{0})];
    check_point_2d(
        mirror,
        -2.0,
        3.0,
        "the precomputed mixed-provider x-reflection is correct");
}

/** Verify rejection of a global mirror index outside the extended range. */
inline void test_invalid_mirror_index_is_rejected() {
    std::cout << "  Pass an invalid global mirror index and require "
                 "std::out_of_range.\n";

    StoredExtendedNodes nodes(make_stored_nodes(), make_boundary());
    const IndexVector invalid_neighbours{nodes.size() + Index{1}};

    check_throws<std::out_of_range>(
        [&nodes, &invalid_neighbours] {
            nodes.activate_cell(Index{0}, invalid_neighbours);
        },
        "activate_cell rejects a mirror index beyond the extended range");
}

} // namespace voronoi_nodes_test
