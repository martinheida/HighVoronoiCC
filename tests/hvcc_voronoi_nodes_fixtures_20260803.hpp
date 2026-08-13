#pragma once

#include <vector>

#include <highvoronoi/geometry/voronoi_nodes.hpp>

#include "hvcc_test_support_voronoi_nodes_20260803.hpp"

namespace voronoi_nodes_test {

using hvtest::Index;
using hvtest::IndexVector;
using hvtest::Scalar;

// ============================================================================
// Short names used consistently by all VoronoiNodes tests
// ============================================================================

template <int Dim>
using Point = highvoronoi::StaticPoint<Scalar, Dim>;

template <int Dim>
using PointView = highvoronoi::StaticPointView<Scalar, Dim>;

template <int Dim>
using Boundary = highvoronoi::Boundary<Dim, Scalar, Index>;

template <int Dim>
using VoronoiNodes = highvoronoi::VoronoiNodes<Scalar, Dim, Index>;

template <int Dim>
using AbstractNodes =
    highvoronoi::AbstractVoronoiNodes<Scalar, Dim, Index>;

template <int Dim>
using ComputedAccess =
    highvoronoi::ComputedNodeAccess<Scalar, Dim, Index>;

template <int Dim>
using HybridAccess =
    highvoronoi::HybridNodeAccess<Scalar, Dim, Index>;

template <int Dim>
using NodeHandle = highvoronoi::NodeHandle<Scalar, Dim, Index>;

template <typename BaseNodes>
using ExtendedNodes = highvoronoi::ExtendedVoronoiNodes<BaseNodes>;

template <typename BaseNodes>
using PrecomputedExtendedNodes =
    highvoronoi::PrecomputedExtendedVoronoiNodes<BaseNodes>;

using Point2 = Point<2>;
using StoredNodes = VoronoiNodes<2>;
using DynamicStoredNodes =
    highvoronoi::VoronoiNodes<Scalar, highvoronoi::Dynamic, Index>;
using DynamicPoint = highvoronoi::DynamicPoint<Scalar>;
using Handle2 = NodeHandle<2>;

// ============================================================================
// Shared geometric fixture
// ============================================================================

/**
 * Build the two-plane boundary used by all reflection tests.
 *
 * Plane 0 is x = 0 and maps (x, y) to (-x, y).
 * Plane 1 is y = 0 and maps (x, y) to (x, -y).
 */
inline Boundary<2> make_boundary() {
    Boundary<2> boundary;

    Point2 origin;
    origin << 0.0, 0.0;

    Point2 x_normal;
    x_normal << -1.0, 0.0;
    boundary.add(highvoronoi::Plane<2, Scalar, Index>(
        origin,
        x_normal,
        highvoronoi::BoundaryCondition::Dirichlet));

    Point2 y_normal;
    y_normal << 0.0, -1.0;
    boundary.add(highvoronoi::Plane<2, Scalar, Index>(
        origin,
        y_normal,
        highvoronoi::BoundaryCondition::Dirichlet));

    return boundary;
}

// ============================================================================
// Stored, computed and hybrid example containers
// ============================================================================

/**
 * Build four explicitly stored points:
 * (1,2), (2,3), (3,4), (4,5).
 */
inline StoredNodes make_stored_nodes() {
    StoredNodes nodes(Index{4});

    Point2 point;
    point << 1.0, 2.0;
    nodes.set(Index{0}, point);

    point << 2.0, 3.0;
    nodes.set(Index{1}, point);

    point << 3.0, 4.0;
    nodes.set(Index{2}, point);

    point << 4.0, 5.0;
    nodes.set(Index{3}, point);

    return nodes;
}

/**
 * Small computed 2D grid used to verify value-returning node access.
 *
 * A 2x2 grid yields (1,2), (2,2), (1,3), (2,3).
 */
class ComputedGridNodes : public ComputedAccess<2> {
public:
    using Base = ComputedAccess<2>;

    ComputedGridNodes(Index width, Index height)
        : Base(Index{2}),
          width_(width),
          height_(height) {}

    [[nodiscard]] Index size() const noexcept override {
        return width_ * height_;
    }

protected:
    void compute_node(Index index, Scalar* target) const override {
        const Index x = index % width_;
        const Index y = index / width_;

        target[0] = Scalar{1} + static_cast<Scalar>(x);
        target[1] = Scalar{2} + static_cast<Scalar>(y);
    }

private:
    Index width_;
    Index height_;
};

/** Return the standard 2x2 computed-node fixture. */
inline ComputedGridNodes make_computed_nodes() {
    return ComputedGridNodes(Index{2}, Index{2});
}

/**
 * Four-node hybrid fixture.
 *
 * Even indices borrow stable storage. Odd indices are calculated on demand.
 * The visible points are again (1,2), (2,3), (3,4), (4,5).
 */
class HybridExampleNodes : public HybridAccess<2> {
public:
    using Base = HybridAccess<2>;

    HybridExampleNodes()
        : Base(Index{2}),
          stored_{
              Scalar{1}, Scalar{2},
              Scalar{0}, Scalar{0},
              Scalar{3}, Scalar{4},
              Scalar{0}, Scalar{0}} {}

    [[nodiscard]] Index size() const noexcept override {
        return Index{4};
    }

protected:
    [[nodiscard]] const Scalar*
    try_get_stored_node_pointer(Index index) const override {
        if ((index % Index{2}) == Index{0}) {
            return stored_.data() + index * Index{2};
        }
        return nullptr;
    }

    void compute_node(Index index, Scalar* target) const override {
        target[0] = Scalar{1} + static_cast<Scalar>(index);
        target[1] = Scalar{2} + static_cast<Scalar>(index);
    }

private:
    std::vector<Scalar> stored_;
};

/** Return the standard hybrid-node fixture. */
inline HybridExampleNodes make_hybrid_nodes() {
    return HybridExampleNodes{};
}

// ============================================================================
// Concrete extended-node types used by the tests
// ============================================================================

using StoredExtendedNodes = ExtendedNodes<StoredNodes>;
using ComputedExtendedNodes = ExtendedNodes<ComputedGridNodes>;
using HybridExtendedNodes = ExtendedNodes<HybridExampleNodes>;

using StoredPrecomputedNodes = PrecomputedExtendedNodes<StoredNodes>;
using ComputedPrecomputedNodes =
    PrecomputedExtendedNodes<ComputedGridNodes>;
using HybridPrecomputedNodes =
    PrecomputedExtendedNodes<HybridExampleNodes>;

} // namespace voronoi_nodes_test
