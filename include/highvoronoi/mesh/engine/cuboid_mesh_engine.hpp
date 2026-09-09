#pragma once

/**
 * @file cuboid_mesh_engine.hpp
 * @brief Analytic Cartesian-grid ComputeMeshEngine for an unbounded setting.
 *
 * Nodes are
 *     start + (k_0 d_0, ..., k_{d-1} d_{d-1}),
 * with 0 <= k_i < repetitions_i. Dimension 0 is the fastest flattened index.
 *
 * Every elementary grid box q, 0 <= q_i < repetitions_i-1, contributes one
 * finite Voronoi vertex at
 *     start + (q_i + 1/2) d_i.
 * Its 2^d generators are the corners q + {0,1}^d. The lower corner q is the
 * unique primary owner; all other box corners see the vertex as secondary.
 *
 * The engine deliberately has no boundary/infinite-edge implementation yet.
 */

#include <highvoronoi/mesh/engine/compute_mesh_engine.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace highvoronoi {

template <typename NodeScalarT,
          typename VertexScalarT,
          typename IndexT,
          int Dim>
class CuboidMeshEngine final
    : public ComputeMeshEngine<
          NodeScalarT,
          VertexScalarT,
          IndexT,
          Dim> {
    static_assert(Dim > 0,
                  "CuboidMeshEngine currently requires a fixed positive dimension.");

public:
    using Base = ComputeMeshEngine<
        NodeScalarT,
        VertexScalarT,
        IndexT,
        Dim>;
    using NodeScalar = typename Base::NodeScalar;
    using VertexScalar = typename Base::VertexScalar;
    using Index = typename Base::Index;
    using Address = typename Base::Address;
    using Sigma = typename Base::Sigma;
    using NodePoint = typename Base::NodePoint;
    using VertexPoint = typename Base::VertexPoint;
    using CountPoint = StaticPoint<Index, Dim>;

    static constexpr int DimensionAtCompileTime = Dim;

    CuboidMeshEngine(
        NodePoint start,
        NodePoint spacing,
        CountPoint repetitions)
        : Base(static_cast<Index>(Dim)),
          start_(std::move(start)),
          spacing_(std::move(spacing)),
          repetitions_(std::move(repetitions)) {
        initialize_strides_and_counts();
        active_vertices_.assign(vertex_count_, std::uint8_t{1});
    }

    [[nodiscard]] Index node_count() const noexcept override {
        return node_count_;
    }

    /**
     * Cartesian neighbour topology is immutable and analytic. These records are
     * therefore safe to expose as persistent virtual HybridNeighbourDatabase
     * addresses even if individual computed vertices are later invalidated by
     * embedding/refinement; affected host cells are marked dirty and may publish
     * a newer stored neighbour record without changing this historical one.
     */
    [[nodiscard]] bool provides_neighbours() const noexcept override {
        return true;
    }

    [[nodiscard]] bool read_neighbours(
        Index local_node,
        Sigma& neighbours) const override {
        require_node(local_node);
        neighbours.clear();
        neighbours.reserve(static_cast<std::size_t>(2 * Dim));

        CountPoint coordinate;
        unflatten_node(local_node, coordinate);
        for (int axis = 0; axis < Dim; ++axis) {
            const Index stride = node_stride_[axis];
            if (coordinate[axis] > Index{0}) {
                neighbours.push_back(static_cast<Index>(local_node - stride));
            }
            if (coordinate[axis] + Index{1} < repetitions_[axis]) {
                neighbours.push_back(static_cast<Index>(local_node + stride));
            }
        }
        std::sort(neighbours.begin(), neighbours.end());
        return true;
    }

    [[nodiscard]] NodeScalar get_data(
        Index local_node,
        Index coordinate) const override {
        require_node(local_node);
        if (coordinate >= static_cast<Index>(Dim)) {
            throw std::out_of_range(
                "CuboidMeshEngine node coordinate out of range.");
        }

        const Index grid_coordinate = static_cast<Index>(
            (local_node / node_stride_[static_cast<int>(coordinate)]) %
            repetitions_[static_cast<int>(coordinate)]);
        return static_cast<NodeScalar>(
            start_[static_cast<int>(coordinate)] +
            spacing_[static_cast<int>(coordinate)] *
                static_cast<NodeScalar>(grid_coordinate));
    }

    [[nodiscard]] Address
    vertex_address_capacity() const noexcept override {
        return vertex_count_;
    }

    [[nodiscard]] const NodePoint& start() const noexcept {
        return start_;
    }

    [[nodiscard]] const NodePoint& spacing() const noexcept {
        return spacing_;
    }

    [[nodiscard]] const CountPoint& repetitions() const noexcept {
        return repetitions_;
    }

    void copy_node(Index local_node, NodeScalar* target) const override {
        require_node(local_node);
        CountPoint coordinate;
        unflatten_node(local_node, coordinate);
        for (int axis = 0; axis < Dim; ++axis) {
            target[axis] = static_cast<NodeScalar>(
                start_[axis] +
                spacing_[axis] * static_cast<NodeScalar>(coordinate[axis]));
        }
    }

    void read_vertex(
        Address local_address,
        VertexPoint& position,
        Sigma& sigma) const override {
        require_vertex(local_address);
        sigma.clear();
        if (!active_vertices_[local_address]) {
            return;
        }

        CountPoint lower;
        unflatten_vertex(local_address, lower);

        for (int axis = 0; axis < Dim; ++axis) {
            position[axis] = static_cast<VertexScalar>(start_[axis]) +
                (static_cast<VertexScalar>(lower[axis]) + VertexScalar{0.5}) *
                    static_cast<VertexScalar>(spacing_[axis]);
        }

        constexpr std::size_t corner_count = std::size_t{1} << Dim;
        sigma.resize(corner_count);
        for (std::size_t mask = 0; mask < corner_count; ++mask) {
            CountPoint corner = lower;
            for (int axis = 0; axis < Dim; ++axis) {
                if ((mask & (std::size_t{1} << axis)) != 0) {
                    ++corner[axis];
                }
            }
            sigma[mask] = flatten_node(corner);
        }
        std::sort(sigma.begin(), sigma.end());
    }

    [[nodiscard]] bool contains_vertex_signature(
        const Sigma& sigma) const override {
        constexpr std::size_t corner_count = std::size_t{1} << Dim;
        if (sigma.size() != corner_count || sigma.empty() ||
            sigma.front() >= node_count_) {
            return false;
        }

        CountPoint lower;
        unflatten_node(sigma.front(), lower);
        for (int axis = 0; axis < Dim; ++axis) {
            if (lower[axis] + Index{1} >= repetitions_[axis]) {
                return false;
            }
        }

        const Address local_address = flatten_vertex(lower);
        if (local_address >= active_vertices_.size() ||
            !active_vertices_[local_address]) {
            return false;
        }

        std::array<Index, corner_count> expected{};
        for (std::size_t mask = 0; mask < corner_count; ++mask) {
            CountPoint corner = lower;
            for (int axis = 0; axis < Dim; ++axis) {
                if ((mask & (std::size_t{1} << axis)) != 0) {
                    ++corner[axis];
                }
            }
            expected[mask] = flatten_node(corner);
        }
        std::sort(expected.begin(), expected.end());
        return std::equal(expected.begin(), expected.end(), sigma.begin());
    }

    [[nodiscard]] bool erase_vertex(Address local_address) override {
        require_vertex(local_address);
        auto& active = active_vertices_[local_address];
        if (!active) {
            return false;
        }
        active = std::uint8_t{0};
        return true;
    }

    [[nodiscard]] std::size_t
    primary_vertex_count(Index local_node) const override {
        require_node(local_node);
        CountPoint node_coordinate;
        unflatten_node(local_node, node_coordinate);
        if (!is_vertex_lower_corner(node_coordinate)) {
            return 0;
        }
        const Address address = flatten_vertex(node_coordinate);
        return active_vertices_[address] ? 1u : 0u;
    }

    [[nodiscard]] Address primary_vertex_address(
        Index local_node,
        std::size_t ordinal) const override {
        if (ordinal != 0 || primary_vertex_count(local_node) != 1) {
            throw std::out_of_range(
                "CuboidMeshEngine primary vertex ordinal out of range.");
        }
        CountPoint coordinate;
        unflatten_node(local_node, coordinate);
        return flatten_vertex(coordinate);
    }

    [[nodiscard]] std::size_t
    secondary_vertex_count(Index local_node) const override {
        require_node(local_node);
        std::size_t count = 0;
        enumerate_secondary(local_node, [&](Address) { ++count; });
        return count;
    }

    [[nodiscard]] Address secondary_vertex_address(
        Index local_node,
        std::size_t ordinal) const override {
        require_node(local_node);
        std::size_t current = 0;
        Address result = 0;
        bool found = false;
        enumerate_secondary(local_node, [&](Address address) {
            if (!found && current++ == ordinal) {
                result = address;
                found = true;
            }
        });
        if (!found) {
            throw std::out_of_range(
                "CuboidMeshEngine secondary vertex ordinal out of range.");
        }
        return result;
    }

    [[nodiscard]] bool next_active_vertex(
        Address& cursor,
        Address& local_address,
        VertexPoint& position,
        Sigma& sigma) const override {
        while (cursor < vertex_count_) {
            const Address candidate = cursor++;
            if (!active_vertices_[candidate]) {
                continue;
            }
            local_address = candidate;
            read_vertex(candidate, position, sigma);
            return true;
        }
        return false;
    }

private:
    void initialize_strides_and_counts() {
        node_stride_[0] = Index{1};
        vertex_stride_[0] = Address{1};

        std::size_t nodes = 1;
        std::size_t vertices = 1;
        bool no_vertices = false;

        for (int axis = 0; axis < Dim; ++axis) {
            if (repetitions_[axis] == Index{0}) {
                throw std::invalid_argument(
                    "CuboidMeshEngine repetitions must be positive.");
            }
            if (!(spacing_[axis] > NodeScalar{0})) {
                throw std::invalid_argument(
                    "CuboidMeshEngine spacing must be strictly positive.");
            }

            if (axis > 0) {
                node_stride_[axis] = checked_index(nodes);
                vertex_stride_[axis] = vertices;
            }

            nodes = checked_product(
                nodes,
                static_cast<std::size_t>(repetitions_[axis]));

            if (repetitions_[axis] <= Index{1}) {
                no_vertices = true;
            }
            if (!no_vertices) {
                vertices = checked_product(
                    vertices,
                    static_cast<std::size_t>(repetitions_[axis] - Index{1}));
            }
        }

        node_count_ = checked_index(nodes);
        vertex_count_ = no_vertices ? Address{0} : vertices;
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left,
        std::size_t right) {
        if (right != 0 &&
            left > (std::numeric_limits<std::size_t>::max)() / right) {
            throw std::overflow_error("CuboidMeshEngine grid size overflow.");
        }
        return left * right;
    }

    [[nodiscard]] static Index checked_index(std::size_t value) {
        if (value > static_cast<std::size_t>((std::numeric_limits<Index>::max)())) {
            throw std::overflow_error(
                "CuboidMeshEngine node count exceeds Index capacity.");
        }
        return static_cast<Index>(value);
    }

    void require_node(Index node) const {
        if (node >= node_count_) {
            throw std::out_of_range("CuboidMeshEngine node index out of range.");
        }
    }

    void require_vertex(Address address) const {
        if (address >= vertex_count_) {
            throw std::out_of_range("CuboidMeshEngine vertex address out of range.");
        }
    }

    void unflatten_node(Index flat, CountPoint& coordinate) const {
        std::size_t value = static_cast<std::size_t>(flat);
        for (int axis = Dim - 1; axis >= 0; --axis) {
            const std::size_t stride = static_cast<std::size_t>(node_stride_[axis]);
            coordinate[axis] = static_cast<Index>(value / stride);
            value %= stride;
        }
    }

    void unflatten_vertex(Address flat, CountPoint& coordinate) const {
        std::size_t value = flat;
        for (int axis = Dim - 1; axis >= 0; --axis) {
            const std::size_t stride = vertex_stride_[axis];
            coordinate[axis] = static_cast<Index>(value / stride);
            value %= stride;
        }
    }

    [[nodiscard]] Index flatten_node(const CountPoint& coordinate) const {
        std::size_t value = 0;
        for (int axis = 0; axis < Dim; ++axis) {
            if (coordinate[axis] >= repetitions_[axis]) {
                throw std::out_of_range("CuboidMeshEngine node coordinate out of range.");
            }
            value += static_cast<std::size_t>(coordinate[axis]) *
                     static_cast<std::size_t>(node_stride_[axis]);
        }
        return checked_index(value);
    }

    [[nodiscard]] Address flatten_vertex(const CountPoint& lower) const {
        std::size_t value = 0;
        for (int axis = 0; axis < Dim; ++axis) {
            if (lower[axis] + Index{1} >= repetitions_[axis]) {
                throw std::out_of_range("CuboidMeshEngine vertex coordinate out of range.");
            }
            value += static_cast<std::size_t>(lower[axis]) * vertex_stride_[axis];
        }
        return value;
    }

    [[nodiscard]] bool
    is_vertex_lower_corner(const CountPoint& coordinate) const noexcept {
        for (int axis = 0; axis < Dim; ++axis) {
            if (coordinate[axis] + Index{1} >= repetitions_[axis]) {
                return false;
            }
        }
        return true;
    }

    template <class Function>
    void enumerate_secondary(Index local_node, Function&& function) const {
        CountPoint node;
        unflatten_node(local_node, node);

        constexpr std::size_t pattern_count = std::size_t{1} << Dim;
        for (std::size_t minus_mask = 1; minus_mask < pattern_count; ++minus_mask) {
            CountPoint lower;
            bool valid = true;
            for (int axis = 0; axis < Dim; ++axis) {
                const bool minus =
                    (minus_mask & (std::size_t{1} << axis)) != 0;
                if (minus) {
                    if (node[axis] == Index{0}) {
                        valid = false;
                        break;
                    }
                    lower[axis] = static_cast<Index>(node[axis] - Index{1});
                } else {
                    if (node[axis] + Index{1} >= repetitions_[axis]) {
                        valid = false;
                        break;
                    }
                    lower[axis] = node[axis];
                }
            }
            if (!valid) {
                continue;
            }
            const Address address = flatten_vertex(lower);
            if (active_vertices_[address]) {
                function(address);
            }
        }
    }

    NodePoint start_;
    NodePoint spacing_;
    CountPoint repetitions_;
    CountPoint node_stride_;
    StaticPoint<Address, Dim> vertex_stride_;
    Index node_count_ = Index{0};
    Address vertex_count_ = Address{0};
    std::vector<std::uint8_t> active_vertices_;
};

} // namespace highvoronoi





