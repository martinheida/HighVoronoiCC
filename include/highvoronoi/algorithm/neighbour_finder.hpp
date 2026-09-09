#pragma once

/**
 * @file neighbour_finder.hpp
 * @brief Reusable cell-local Voronoi facet-neighbour classification.
 *
 * The finder deliberately works in stable internal numbering. Ordinary nodes
 * use their stable internal indices; boundary mirrors use AbstractMesh's
 * high-end encoding max(Index)-1-plane.
 *
 * Candidate classification is local and monotone for one cell:
 *
 * - a generator first seen at a general-position vertex is a facet neighbour;
 * - at a degenerate vertex, EdgeIterator::OnCellEdges computes the complete
 *   top-level FEI valid-node set for the active cell;
 * - generators in that vertex signature which are not FEI-valid are permanent
 *   non-neighbours of the active cell and may be removed from later vertex
 *   signatures before repeating the FEI analysis.
 *
 * This uses two packed BitVectors over the complete ordinary+boundary candidate
 * universe. The vectors keep their allocation across reset() calls; only slots
 * touched by the previous cell are cleared. No candidate-by-vertex matrix is
 * stored.
 */

#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/algorithm/edge_iterator.hpp>
#include <highvoronoi/core/point.hpp>
#include <highvoronoi/core/voronoi_nodes.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::detail {

/**
 * Stable-internal ordinary nodes plus on-demand high-end boundary mirrors.
 *
 * EdgeIterator needs only dimension() and copy_node().  This adapter therefore
 * avoids constructing a second ExtendedVoronoiNodes object and, importantly,
 * understands the persistent high-end boundary encoding used by database
 * signatures.
 */
template <class InternalNodesT, class BoundaryT>
class InternalNeighbourNodes final
    : public highvoronoi::AbstractVoronoiNodes<
          typename InternalNodesT::Scalar,
          InternalNodesT::DimensionAtCompileTime,
          typename InternalNodesT::Index> {
public:
    using InternalNodes = InternalNodesT;
    using Boundary = BoundaryT;
    using Scalar = typename InternalNodes::Scalar;
    using Index = typename InternalNodes::Index;
    static constexpr int DimensionAtCompileTime =
        InternalNodes::DimensionAtCompileTime;
    using Base = AbstractVoronoiNodes<Scalar, DimensionAtCompileTime, Index>;
    using Point = PointType<Scalar, DimensionAtCompileTime>;

    InternalNeighbourNodes(
        const InternalNodes& nodes,
        const Boundary& boundary)
        : Base(nodes.dimension()),
          nodes_(&nodes),
          boundary_(&boundary),
          active_cell_point_(make_point(nodes.dimension())) {}

    void rebind_boundary(const Boundary& boundary) noexcept {
        boundary_ = &boundary;
    }

    [[nodiscard]] Index size() const noexcept override {
        return nodes_->size();
    }

    [[nodiscard]] Scalar get_data(
        Index index,
        Index coordinate) const override {
        if (index < nodes_->size()) {
            return nodes_->get_data(index, coordinate);
        }

        const Index plane = boundary_plane(index);
        const auto& boundary_plane_ref = (*boundary_)[plane];
        const Eigen::Index c = static_cast<Eigen::Index>(coordinate);
        const Scalar signed_distance = signed_distance_to_plane(boundary_plane_ref);
        return active_cell_point_[c] -
               Scalar{2} * signed_distance * boundary_plane_ref.normal()[c];
    }

    void activate_internal_cell(Index cell) {
        if (cell >= nodes_->size()) {
            throw std::out_of_range(
                "Neighbour analysis active cell is not an internal ordinary node.");
        }
        nodes_->copy_node(cell, active_cell_point_.data());
    }

protected:
    void copy_node_impl(Index index, Scalar* target) const override {
        if (index < nodes_->size()) {
            nodes_->copy_node(index, target);
            return;
        }

        const Index plane = boundary_plane(index);
        const auto& boundary_plane_ref = (*boundary_)[plane];
        const Scalar signed_distance = signed_distance_to_plane(boundary_plane_ref);

        for (Index coordinate = Index{0};
             coordinate < this->dimension();
             ++coordinate) {
            const Eigen::Index c = static_cast<Eigen::Index>(coordinate);
            target[static_cast<std::size_t>(coordinate)] =
                active_cell_point_[c] -
                Scalar{2} * signed_distance * boundary_plane_ref.normal()[c];
        }
    }

private:
    [[nodiscard]] static Point make_point(Index dimension) {
        if constexpr (DimensionAtCompileTime == Dynamic) {
            return Point(static_cast<Eigen::Index>(dimension));
        } else {
            (void)dimension;
            return Point{};
        }
    }

    [[nodiscard]] Index boundary_plane(Index encoded_index) const {
        const Index maximum = (std::numeric_limits<Index>::max)();
        if (encoded_index == maximum) {
            throw std::out_of_range(
                "Maximum Index is the invalid-node marker, not a boundary mirror.");
        }

        const Index plane = static_cast<Index>(
            maximum - static_cast<Index>(encoded_index + Index{1}));
        if (plane >= boundary_->size()) {
            throw std::out_of_range(
                "Neighbour-analysis index is neither an ordinary node nor a valid boundary mirror.");
        }
        return plane;
    }

    template <class Plane>
    [[nodiscard]] Scalar signed_distance_to_plane(const Plane& plane) const {
        Scalar value = Scalar{0};
        for (Index coordinate = Index{0};
             coordinate < this->dimension();
             ++coordinate) {
            const Eigen::Index c = static_cast<Eigen::Index>(coordinate);
            value +=
                (active_cell_point_[c] - plane.base()[c]) * plane.normal()[c];
        }
        return value;
    }

    const InternalNodes* nodes_;
    const Boundary* boundary_;
    Point active_cell_point_;
};

/**
 * Reusable classifier for the neighbours of one stable internal cell.
 *
 * `seen_` means that the candidate has already been classified.  Candidates
 * with seen=true and invalid=false are confirmed neighbours.  This works with
 * only two bit vectors because every candidate is classified at its first
 * vertex occurrence; there is no persistent Unknown state between vertices.
 */
template <class InternalNodesT,
          class BoundaryT,
          typename VertexScalarT = typename InternalNodesT::Scalar>
class NeighbourFinder final {
public:
    using InternalNodes = InternalNodesT;
    using Boundary = BoundaryT;
    using Scalar = typename InternalNodes::Scalar;
    using VertexScalar = VertexScalarT;
    using Index = typename InternalNodes::Index;
    static constexpr int DimensionAtCompileTime =
        InternalNodes::DimensionAtCompileTime;
    using VertexPoint = PointType<VertexScalar, DimensionAtCompileTime>;
    using Sigma = std::vector<Index>;
    using AnalysisNodes = InternalNeighbourNodes<InternalNodes, Boundary>;
    using Iterator = EdgeIterator<AnalysisNodes>;

    NeighbourFinder(
        const InternalNodes& nodes,
        const Boundary& boundary)
        : nodes_(nodes),
          boundary_(&boundary),
          analysis_nodes_(nodes, boundary),
          edge_iterator_(analysis_nodes_),
          vertex_buffer_(make_vertex_point(nodes.dimension())) {}

    void rebind_boundary(const Boundary& boundary) noexcept {
        boundary_ = &boundary;
        analysis_nodes_.rebind_boundary(boundary);
    }

    /** Start classification of one stable internal ordinary cell. */
    void reset(Index cell) {
        if (cell >= nodes_.size()) {
            throw std::out_of_range(
                "NeighbourFinder cell is outside stable internal node storage.");
        }

        const std::size_t current_node_count =
            static_cast<std::size_t>(nodes_.size());
        const std::size_t current_boundary_count =
            static_cast<std::size_t>(boundary_->size());

        // Ordinary candidates occupy [0,node_count), while boundary mirrors
        // occupy [node_count,node_count+boundary_count).  The boundary part of
        // this compact layout therefore moves whenever stable node storage
        // grows.  This is exactly what happens across RefineVoronoi rounds.
        //
        // Never try to clear candidates from the previous cell through the new
        // layout: an old boundary bit could otherwise remain at its old slot,
        // which may now be the slot of a newly appended ordinary node.  On a
        // structural layout change, clear the packed words once and keep their
        // allocation.  Normal cell-to-cell resets retain the O(adjacency)
        // touched-slot path below.
        const bool candidate_layout_changed =
            current_node_count != candidate_node_count_ ||
            current_boundary_count != candidate_boundary_count_;

        if (candidate_layout_changed) {
            seen_.clear();
            invalid_.clear();
        } else {
            // Clear only candidate slots touched by the previous cell.  This
            // keeps the normal reset cost proportional to the previous
            // adjacency count rather than the total global node count.
            for (const Index candidate : candidates_) {
                const std::size_t slot = candidate_slot(candidate);
                seen_.reset(slot);
                invalid_.reset(slot);
            }
        }

        candidates_.clear();
        filtered_sigma_.clear();
        valid_buffer_.clear();

        candidate_node_count_ = current_node_count;
        candidate_boundary_count_ = current_boundary_count;

        cell_ = cell;
        const std::size_t slot_count = candidate_slot_count();
        if (seen_.size() < slot_count) {
            seen_.resize(slot_count, false);
            invalid_.resize(slot_count, false);
        }
        analysis_nodes_.activate_internal_cell(cell_);
    }

    /**
     * Classify every generator in one active finite vertex of the current cell.
     * The input signature must use stable-internal/high-end-boundary encoding.
     */
    template <class PointLike>
    void process_vertex(const Sigma& sigma, const PointLike& position) {
        if (cell_ == invalid_index()) {
            throw std::logic_error(
                "NeighbourFinder::reset must be called before process_vertex.");
        }
        if (!std::binary_search(sigma.begin(), sigma.end(), cell_)) {
            throw std::logic_error(
                "Cell-local vertex signature does not contain the active cell.");
        }

        filtered_sigma_.clear();
        filtered_sigma_.reserve(sigma.size());

        for (const Index generator : sigma) {
            if (generator == cell_) {
                filtered_sigma_.push_back(generator);
                continue;
            }

            const std::size_t slot = candidate_slot(generator);
            if (!seen_.test(slot)) {
                candidates_.push_back(generator);
            } else if (invalid_.test(slot)) {
                // A generator rejected at one shared vertex cannot define a
                // positive (d-1)-facet with this cell. Remove the redundant
                // constraint from all later local FEI problems.
                continue;
            }
            filtered_sigma_.push_back(generator);
        }

        const std::size_t minimum_vertex_size =
            static_cast<std::size_t>(analysis_nodes_.dimension()) + 1U;
        if (filtered_sigma_.size() < minimum_vertex_size) {
            throw std::logic_error(
                "Removing previously invalid neighbour candidates destroyed the local vertex rank.");
        }

        if (filtered_sigma_.size() == minimum_vertex_size) {
            // General position after removing globally redundant candidates:
            // every remaining non-cell generator defines one cell facet.
            for (const Index generator : filtered_sigma_) {
                if (generator != cell_) {
                    mark_neighbour(generator);
                }
            }
            return;
        }

        edge_iterator_.reset(
            filtered_sigma_,
            position,
            cell_,
            typename Iterator::OnCellEdges{});
        edge_iterator_.valid_generators(valid_buffer_);
        std::sort(valid_buffer_.begin(), valid_buffer_.end());

        // Any as-yet-unclassified candidate at this vertex which does not
        // survive Lemma 2.18's local cone reduction is permanently invalid.
        for (const Index generator : filtered_sigma_) {
            if (generator == cell_) {
                continue;
            }

            const std::size_t slot = candidate_slot(generator);
            const bool locally_valid = std::binary_search(
                valid_buffer_.begin(), valid_buffer_.end(), generator);

            if (seen_.test(slot)) {
                // Already-confirmed true neighbours must remain locally valid
                // at every later shared vertex. Treat a violation as a geometry
                // invariant failure rather than silently changing classification.
                if (!invalid_.test(slot) && !locally_valid) {
                    throw std::logic_error(
                        "A previously confirmed neighbour is not FEI-valid at another shared vertex.");
                }
                continue;
            }

            if (locally_valid) {
                mark_neighbour(generator);
            } else {
                mark_invalid(generator);
            }
        }
    }

    /** Write the sorted stable-internal true-neighbour list. */
    void finish(Sigma& neighbours) const {
        neighbours.clear();
        neighbours.reserve(candidates_.size());

        for (const Index candidate : candidates_) {
            const std::size_t slot = candidate_slot(candidate);
            if (seen_.test(slot) && !invalid_.test(slot)) {
                neighbours.push_back(candidate);
            }
        }

        std::sort(neighbours.begin(), neighbours.end());
    }

    /** Caller-recycled DB read signature buffer owned by this workspace. */
    [[nodiscard]] Sigma& read_sigma_buffer() noexcept {
        return read_sigma_buffer_;
    }

    /** Caller-recycled finite-vertex position buffer owned by this workspace. */
    [[nodiscard]] VertexPoint& vertex_buffer() noexcept {
        return vertex_buffer_;
    }

    /** Caller-recycled final sorted internal neighbour list. */
    [[nodiscard]] Sigma& result_buffer() noexcept {
        return result_buffer_;
    }

    [[nodiscard]] std::size_t candidate_count() const noexcept {
        return candidates_.size();
    }

    [[nodiscard]] std::size_t rejected_count() const {
        std::size_t count = 0;
        for (const Index candidate : candidates_) {
            if (invalid_.test(candidate_slot(candidate))) {
                ++count;
            }
        }
        return count;
    }

private:
    [[nodiscard]] static VertexPoint make_vertex_point(Index dimension) {
        if constexpr (DimensionAtCompileTime == Dynamic) {
            return VertexPoint(static_cast<Eigen::Index>(dimension));
        } else {
            (void)dimension;
            return VertexPoint{};
        }
    }

    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return (std::numeric_limits<Index>::max)();
    }

    [[nodiscard]] std::size_t candidate_slot_count() const {
        return static_cast<std::size_t>(nodes_.size()) +
               static_cast<std::size_t>(boundary_->size());
    }

    [[nodiscard]] std::size_t candidate_slot(Index candidate) const {
        if (candidate < nodes_.size()) {
            return static_cast<std::size_t>(candidate);
        }

        const Index maximum = (std::numeric_limits<Index>::max)();
        if (candidate == maximum) {
            throw std::out_of_range(
                "Maximum Index is not a valid neighbour candidate.");
        }
        const Index plane = static_cast<Index>(
            maximum - static_cast<Index>(candidate + Index{1}));
        if (plane >= boundary_->size()) {
            throw std::out_of_range(
                "Neighbour candidate is neither an internal node nor a boundary mirror.");
        }
        return static_cast<std::size_t>(nodes_.size()) +
               static_cast<std::size_t>(plane);
    }

    void mark_neighbour(Index candidate) {
        const std::size_t slot = candidate_slot(candidate);
        seen_.set(slot);
        invalid_.reset(slot);
    }

    void mark_invalid(Index candidate) {
        const std::size_t slot = candidate_slot(candidate);
        seen_.set(slot);
        invalid_.set(slot);
    }

    const InternalNodes& nodes_;
    const Boundary* boundary_;
    AnalysisNodes analysis_nodes_;
    Iterator edge_iterator_;

    Index cell_{invalid_index()};

    // Layout under which seen_/invalid_ currently encode candidates.  Boundary
    // slots are relative to the ordinary-node count, so structural node growth
    // requires one packed-vector clear before the workspace is reused.
    std::size_t candidate_node_count_{0};
    std::size_t candidate_boundary_count_{0};
    BitVector seen_;
    BitVector invalid_;
    Sigma candidates_;
    Sigma filtered_sigma_;
    Sigma valid_buffer_;

    // Persistent caller-side I/O scratch. Keeping these here is important:
    // a reused NeighbourFinder should not allocate a fresh DB signature or
    // result vector for every cell.
    Sigma read_sigma_buffer_;
    Sigma result_buffer_;
    VertexPoint vertex_buffer_;
};

} // namespace highvoronoi::detail
