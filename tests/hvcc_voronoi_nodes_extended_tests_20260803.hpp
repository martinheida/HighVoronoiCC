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

/**
 * Build the neighbour list used by the current activate_cell interface.
 *
 * This helper belongs next to the ExtendedNodes tests because ordinary node
 * tests do not need global mirror indices. It returns exactly one mirror index
 * per boundary plane. For the shared two-plane fixture the list is [4, 5].
 * It contains neither an ordinary node index nor duplicate mirror indices.
 */
template <typename Extended>
IndexVector all_mirror_indices(const Extended& nodes) {
    IndexVector result;
    result.reserve(nodes.mirror_count());

    for (Index plane = Index{0}; plane < nodes.mirror_count(); ++plane) {
        result.push_back(nodes.mirror_index(plane));
    }

    return result;
}

/**
 * Verify ExtendedNodes built from stored base nodes.
 *
 * This is a pure stored branch: original nodes and activated mirror nodes both
 * have stable memory. The test therefore checks lengths, ordinary-node access,
 * activation of both boundary reflections and zero-copy operator[] access.
 */
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

    const auto inner = nodes[Index{1}];
    check_point_2d(
        inner,
        2.0,
        3.0,
        "operator[] reads inner node 1 before activation");

    check(
        inner.data() == nodes.inner_nodes().stable_node_data(Index{1}),
        "inner-node operator[] access remains zero-copy");

    const IndexVector neighbours = all_mirror_indices(nodes);
    check(
        neighbours.size() == nodes.mirror_count(),
        "activate_cell receives exactly one global index per mirror slot");

    nodes.activate_cell(Index{1}, neighbours);

    const Index x_mirror = nodes.mirror_index(Index{0});
    const Index y_mirror = nodes.mirror_index(Index{1});

    const auto reflected_x = nodes[x_mirror];
    const auto reflected_y = nodes[y_mirror];

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

    check(
        reflected_x.data() == nodes.stable_node_data(x_mirror),
        "a stored mirror node is returned as a zero-copy view");

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

/**
 * Verify ExtendedNodes built from computed base nodes.
 *
 * The resulting class is hybrid, not computed: ordinary nodes are still
 * calculated on demand and therefore produce owning handles, while activated
 * mirror nodes live in stable slots and therefore produce borrowing handles.
 */
inline void test_extended_computed_base() {
    std::cout << "  Extend computed nodes: inner results must own their data, "
                 "mirror results must borrow stable slots.\n";

    ComputedExtendedNodes nodes(make_computed_nodes(), make_boundary());

    check(
        nodes.inner_size() == Index{4} &&
            nodes.mirror_count() == Index{2} &&
            nodes.size() == Index{6},
        "computed-base ExtendedNodes reports four inner and two mirror nodes");

    const Handle2 original = nodes[Index{3}];
    check(
        original.owns_storage(),
        "a computed inner node produces an owning handle");

    check_point_2d(
        original,
        2.0,
        3.0,
        "operator[] computes inner node 3 as (2, 3)");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{3}, neighbours);

    const Handle2 reflected_x = nodes[nodes.mirror_index(Index{0})];
    const Handle2 reflected_y = nodes[nodes.mirror_index(Index{1})];

    check(
        !reflected_x.owns_storage() && !reflected_y.owns_storage(),
        "both activated mirror handles borrow stable mirror storage");

    check_point_2d(
        reflected_x,
        -2.0,
        3.0,
        "operator[] returns the stored x-reflection of computed cell 3");

    check_point_2d(
        reflected_y,
        2.0,
        -3.0,
        "operator[] returns the stored y-reflection of computed cell 3");

    check_point_2d(
        original,
        2.0,
        3.0,
        "the owning inner-node result remains valid after mirror access");
}

/**
 * Verify ExtendedNodes built from hybrid base nodes.
 *
 * This test covers all three storage situations visible through one extended
 * container: a stored inner node borrows, a computed inner node owns, and each
 * activated mirror node borrows the extended container's stable mirror slots.
 */
inline void test_extended_hybrid_base() {
    std::cout << "  Extend hybrid nodes and distinguish stored inner, computed "
                 "inner and stored mirror access.\n";

    HybridExtendedNodes nodes(make_hybrid_nodes(), make_boundary());

    const Handle2 stored_inner = nodes[Index{0}];
    const Handle2 computed_inner = nodes[Index{1}];

    check(
        !stored_inner.owns_storage(),
        "stored inner node 0 produces a borrowing handle");

    check(
        computed_inner.owns_storage(),
        "computed inner node 1 produces an owning handle");

    check_point_2d(
        stored_inner,
        1.0,
        2.0,
        "operator[] reads the stored inner node");

    check_point_2d(
        computed_inner,
        2.0,
        3.0,
        "operator[] computes the on-demand inner node");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{1}, neighbours);

    const Handle2 reflected_x = nodes[nodes.mirror_index(Index{0})];
    const Handle2 reflected_y = nodes[nodes.mirror_index(Index{1})];

    check(
        !reflected_x.owns_storage() && !reflected_y.owns_storage(),
        "activated hybrid mirror nodes borrow stable mirror storage");

    check_point_2d(
        reflected_x,
        -2.0,
        3.0,
        "operator[] returns the correct hybrid x-reflection");

    check_point_2d(
        reflected_y,
        2.0,
        -3.0,
        "operator[] returns the correct hybrid y-reflection");
}

// ============================================================================
// Precomputed extended nodes
// ============================================================================

/**
 * Verify precomputed reflections for stored base nodes.
 *
 * Four cells and two boundary planes require eight stored precomputed points.
 * Activating one cell must copy its two precomputed reflections into the two
 * public mirror slots without changing stored operator[] semantics.
 */
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

/**
 * Verify precomputed reflections for computed base nodes.
 *
 * The complete extended class is hybrid: inner nodes are calculated and own
 * their returned values, while activated precomputed mirrors borrow stable
 * mirror slots.
 */
inline void test_precomputed_computed_base() {
    std::cout << "  Precompute reflections of computed nodes and verify the "
                 "resulting hybrid access semantics.\n";

    ComputedPrecomputedNodes nodes(make_computed_nodes(), make_boundary());

    const Handle2 inner = nodes[Index{3}];
    check(
        inner.owns_storage(),
        "a computed inner node still produces an owning handle");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{3}, neighbours);

    const Handle2 mirror = nodes[nodes.mirror_index(Index{0})];
    check(
        !mirror.owns_storage(),
        "an activated precomputed mirror borrows stable storage");

    check_point_2d(
        mirror,
        -2.0,
        3.0,
        "the precomputed x-reflection of computed cell 3 is correct");
}

/**
 * Verify precomputed reflections for hybrid base nodes.
 *
 * The test confirms that precomputation changes only where reflection values
 * are obtained. It does not change the hybrid ownership rules exposed by
 * operator[].
 */
inline void test_precomputed_hybrid_base() {
    std::cout << "  Precompute reflections of hybrid nodes while preserving "
                 "borrowed and owned handle semantics.\n";

    HybridPrecomputedNodes nodes(make_hybrid_nodes(), make_boundary());

    const Handle2 stored_inner = nodes[Index{0}];
    const Handle2 computed_inner = nodes[Index{1}];

    check(
        !stored_inner.owns_storage() && computed_inner.owns_storage(),
        "stored and computed inner nodes retain their hybrid ownership rules");

    const IndexVector neighbours = all_mirror_indices(nodes);
    nodes.activate_cell(Index{1}, neighbours);

    const Handle2 mirror = nodes[nodes.mirror_index(Index{0})];
    check(
        !mirror.owns_storage(),
        "an activated precomputed hybrid mirror borrows stable storage");

    check_point_2d(
        mirror,
        -2.0,
        3.0,
        "the precomputed hybrid x-reflection is correct");
}

/**
 * Verify rejection of a global mirror index outside the extended range.
 *
 * The valid global indices are 0 through 5 for the shared fixture. Passing 7
 * cannot identify either an inner node or one of the two boundary planes and
 * must therefore raise std::out_of_range.
 */
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
