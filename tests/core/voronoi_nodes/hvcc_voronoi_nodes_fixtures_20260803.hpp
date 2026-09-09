#pragma once

#include <vector>

#include <highvoronoi/core/voronoi_nodes.hpp>

#include "hvcc_test_support_voronoi_nodes_20260803.hpp"

namespace voronoi_nodes_test {

using hvtest::Index;
using hvtest::IndexVector;
using hvtest::Scalar;

// ============================================================================
// Short names used consistently by all VoronoiNodes tests
// ============================================================================

template <int Dim>
using Point = highvoronoi::PointType<Scalar, Dim>;

template <int Dim>
using Boundary = highvoronoi::Boundary<Dim, Scalar, Index>;

template <int Dim>
using VoronoiNodes = highvoronoi::VoronoiNodes<Scalar, Dim, Index>;

template <int Dim>
using AbstractNodes =
    highvoronoi::AbstractVoronoiNodes<Scalar, Dim, Index>;

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
// Stored, computed and internally mixed example providers
// ============================================================================

/** Build four explicitly stored points: (1,2), (2,3), (3,4), (4,5). */
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

/** Small computed 2D grid implementing the same universal node interface. */
class ComputedGridNodes : public AbstractNodes<2> {
public:
    using Base = AbstractNodes<2>;

    ComputedGridNodes(Index width, Index height)
        : Base(Index{2}),
          width_(width),
          height_(height) {}

    [[nodiscard]] Index size() const noexcept override {
        return width_ * height_;
    }

    [[nodiscard]] Scalar get_data(
        Index index,
        Index coordinate) const override {
        const Index x = index % width_;
        const Index y = index / width_;
        return coordinate == Index{0}
            ? Scalar{1} + static_cast<Scalar>(x)
            : Scalar{2} + static_cast<Scalar>(y);
    }

protected:
    void copy_node_impl(Index index, Scalar* target) const override {
        const Index x = index % width_;
        const Index y = index / width_;

        target[0] = Scalar{1} + static_cast<Scalar>(x);
        target[1] = Scalar{2} + static_cast<Scalar>(y);
    }

private:
    Index width_;
    Index height_;
};

inline ComputedGridNodes make_computed_nodes() {
    return ComputedGridNodes(Index{2}, Index{2});
}

/**
 * Four-node provider whose implementation internally mixes stored and computed
 * coordinates. The generic type system deliberately does not expose that fact.
 */
class MixedExampleNodes : public AbstractNodes<2> {
public:
    using Base = AbstractNodes<2>;

    MixedExampleNodes()
        : Base(Index{2}),
          stored_{
              Scalar{1}, Scalar{2},
              Scalar{0}, Scalar{0},
              Scalar{3}, Scalar{4},
              Scalar{0}, Scalar{0}} {}

    [[nodiscard]] Index size() const noexcept override {
        return Index{4};
    }

    [[nodiscard]] Scalar get_data(
        Index index,
        Index coordinate) const override {
        if ((index % Index{2}) == Index{0}) {
            return stored_[
                static_cast<std::size_t>(index * Index{2} + coordinate)];
        }
        return Scalar{1} + static_cast<Scalar>(index + coordinate);
    }

protected:
    void copy_node_impl(Index index, Scalar* target) const override {
        if ((index % Index{2}) == Index{0}) {
            const Scalar* source =
                stored_.data() + static_cast<std::size_t>(index * Index{2});
            target[0] = source[0];
            target[1] = source[1];
            return;
        }

        target[0] = Scalar{1} + static_cast<Scalar>(index);
        target[1] = Scalar{2} + static_cast<Scalar>(index);
    }

private:
    std::vector<Scalar> stored_;
};

inline MixedExampleNodes make_mixed_nodes() {
    return MixedExampleNodes{};
}

// ============================================================================
// Concrete extended-node types used by the tests
// ============================================================================

using StoredExtendedNodes = ExtendedNodes<StoredNodes>;
using ComputedExtendedNodes = ExtendedNodes<ComputedGridNodes>;
using MixedExtendedNodes = ExtendedNodes<MixedExampleNodes>;

using StoredPrecomputedNodes = PrecomputedExtendedNodes<StoredNodes>;
using ComputedPrecomputedNodes =
    PrecomputedExtendedNodes<ComputedGridNodes>;
using MixedPrecomputedNodes =
    PrecomputedExtendedNodes<MixedExampleNodes>;

} // namespace voronoi_nodes_test
