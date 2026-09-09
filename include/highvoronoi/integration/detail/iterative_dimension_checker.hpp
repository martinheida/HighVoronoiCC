#pragma once

/**
 * @file iterative_dimension_checker.hpp
 * @brief Julia-faithful recursion-path dimension checker for polygon integrators.
 */

#include <highvoronoi/algorithm/edge_iterator.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace highvoronoi::detail {

/**
 * Direct port of HighVoronoi.jl's IterativeDimensionChecker.
 *
 * The checker owns and reuses the orthonormal-basis/cone scratch required by
 * recursive polytope descent. PolygonIntegrator additionally uses its
 * EdgeIterator; FastPolygonIntegrator uses the exposed basis and current path
 * for projected distances and canonical facet identifiers.
 */
template <class Nodes>
class IterativeDimensionChecker final {
public:
    using Iterator = EdgeIterator<Nodes>;
    using Index = typename Nodes::Index;
    using Point = typename Iterator::Point;

    IterativeDimensionChecker(
        const Nodes& nodes,
        std::size_t dimension)
        : nodes_(&nodes),
          dimension_(dimension),
          local_basis_(make_points(dimension)),
          valid_neighbors_(dimension),
          current_path_(dimension, invalid_index()),
          edge_iterator_(nodes),
          cell_buffer_(make_point(dimension)),
          node_buffer_(make_point(dimension)) {}

    [[nodiscard]] Iterator& edge_iterator() noexcept {
        return edge_iterator_;
    }

    [[nodiscard]] const std::vector<Point>& local_basis() const noexcept {
        return local_basis_;
    }

    [[nodiscard]] const std::vector<Index>& current_path() const noexcept {
        return current_path_;
    }

    [[nodiscard]] const std::vector<Index>& neighbours() const noexcept {
        return neighbours_;
    }

    [[nodiscard]] bool trivial() const noexcept { return trivial_; }

    template <class Vertices>
    [[nodiscard]] std::size_t reset(
        const std::vector<Index>& neighbours,
        Index cell,
        const Vertices& vertices,
        bool anyway = true) {
        trivial_ = true;

        // Exact Julia shortcut.
        if (dimension_ == std::size_t{2}) {
            return std::size_t{3};
        }

        std::size_t maximum_signature = dimension_ + 1;
        for (const auto& vertex : vertices) {
            const std::size_t size = vertex.sigma.size();
            if (size > dimension_ + 1) {
                maximum_signature = (std::max)(maximum_signature, size);
                trivial_ = false;
            }
        }

        trivial_ = trivial_ && anyway;
        if (trivial_) {
            return dimension_ + 1;
        }

        neighbours_ = neighbours;
        resize_cones(neighbours_.size());
        for (auto& valid : valid_neighbors_) {
            valid.resize(neighbours_.size());
        }

        nodes_->copy_node(cell, cell_buffer_.data());
        for (std::size_t i = 0; i < neighbours_.size(); ++i) {
            nodes_->copy_node(neighbours_[i], node_buffer_.data());
            local_cone_[i] = node_buffer_ - cell_buffer_;
            const auto norm = local_cone_[i].norm();
            if (!(norm > typename Point::Scalar{0})) {
                throw std::logic_error(
                    "IterativeDimensionChecker found coincident generators");
            }
            local_cone_[i] /= norm;
        }

        if (!valid_neighbors_.empty()) {
            std::fill(
                valid_neighbors_[0].begin(),
                valid_neighbors_[0].end(),
                std::uint8_t{1});
        }
        if (valid_neighbors_.size() > 1) {
            std::fill(
                valid_neighbors_[1].begin(),
                valid_neighbors_[1].end(),
                std::uint8_t{1});
        }

        return maximum_signature;
    }

    /** Julia `set_dimension(idc, entry, _Cell, neighbor)`. */
    [[nodiscard]] bool set_dimension(
        std::size_t entry_one_based,
        Index cell,
        Index neighbour) {
        if (neighbour == cell) {
            return false;
        }
        if (trivial_) {
            return true;
        }
        if (entry_one_based == 0 || entry_one_based > dimension_) {
            throw std::out_of_range(
                "IterativeDimensionChecker entry outside dimension");
        }

        const auto found = std::find(
            neighbours_.begin(),
            neighbours_.end(),
            neighbour);
        if (found == neighbours_.end()) {
            return false;
        }
        const std::size_t index = static_cast<std::size_t>(
            std::distance(neighbours_.begin(), found));
        const std::size_t entry = entry_one_based - 1;

        if (entry == 0) {
            local_basis_[0] = local_cone_[index];
        } else {
            if (entry >= valid_neighbors_.size() ||
                index >= valid_neighbors_[entry].size() ||
                valid_neighbors_[entry][index] == std::uint8_t{0}) {
                return false;
            }

            if (entry + 1 < dimension_) {
                valid_neighbors_[entry + 1] = valid_neighbors_[entry];
            }

            local_basis_[entry] = local_cone_[index];

            // Julia performs the complete projection twice.
            for (int repeat = 0; repeat < 2; ++repeat) {
                for (std::size_t previous = 0;
                     previous < entry;
                     ++previous) {
                    local_basis_[entry] -=
                        local_basis_[entry].dot(local_basis_[previous]) *
                        local_basis_[previous];
                }
            }

            const auto norm = local_basis_[entry].norm();
            if (norm < typename Point::Scalar{1.0e-5}) {
                return false;
            }
            local_basis_[entry] /= norm;
        }

        current_path_[entry] = neighbour;
        return true;
    }

private:
    [[nodiscard]] static Index invalid_index() noexcept {
        return (std::numeric_limits<Index>::max)();
    }

    [[nodiscard]] static Point make_point(std::size_t dimension) {
        Point point;
        if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
            point.resize(static_cast<Eigen::Index>(dimension));
        }
        point.setZero();
        return point;
    }

    [[nodiscard]] static std::vector<Point> make_points(
        std::size_t dimension) {
        std::vector<Point> result;
        result.reserve(dimension);
        for (std::size_t i = 0; i < dimension; ++i) {
            result.push_back(make_point(dimension));
        }
        return result;
    }

    void resize_cones(std::size_t size) {
        const std::size_t old = local_cone_.size();
        local_cone_.resize(size);
        if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
            for (std::size_t i = old; i < size; ++i) {
                local_cone_[i].resize(static_cast<Eigen::Index>(dimension_));
            }
        }
    }

    const Nodes* nodes_ = nullptr;
    std::size_t dimension_ = 0;
    std::vector<Point> local_basis_;
    std::vector<Index> neighbours_;
    std::vector<Point> local_cone_;
    std::vector<std::vector<std::uint8_t>> valid_neighbors_;
    std::vector<Index> current_path_;
    bool trivial_ = true;
    Iterator edge_iterator_;
    Point cell_buffer_;
    Point node_buffer_;
};

} // namespace highvoronoi::detail
