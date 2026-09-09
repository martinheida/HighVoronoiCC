#pragma once

/**
 * @file mapped_voronoi_nodes.hpp
 * @brief Non-owning public-node facade applying an external public/internal map.
 *
 * The adapter stores references only to the underlying ordinary-node provider
 * and to the mapping. It deliberately knows nothing about the owning mesh.
 * This keeps compute/view facades independent of the concrete mesh layout and
 * avoids repeated friend access to mesh internals in node hot paths.
 */

#include <highvoronoi/core/voronoi_nodes.hpp>

#include <type_traits>
#include <utility>

namespace highvoronoi::detail {

template <class NodesT, class MappingT>
class MappedVoronoiNodes final
    : public AbstractVoronoiNodes<
          typename NodesT::Scalar,
          NodesT::DimensionAtCompileTime,
          typename NodesT::Index> {
public:
    using Nodes = NodesT;
    using Mapping = MappingT;
    using Scalar = typename Nodes::Scalar;
    using Index = typename Nodes::Index;
    static constexpr int Dim = Nodes::DimensionAtCompileTime;
    static constexpr int DimensionAtCompileTime = Dim;
    using Base = AbstractVoronoiNodes<Scalar, Dim, Index>;

    MappedVoronoiNodes(
        const Nodes& nodes,
        const Mapping& mapping,
        Index runtime_dimension)
        : Base(runtime_dimension), nodes_(nodes), mapping_(mapping) {}

    [[nodiscard]] Index size() const noexcept override {
        return mapping_.size();
    }

    [[nodiscard]] Scalar get_data(
        Index public_node,
        Index coordinate) const override {
        return nodes_.get_data(
            mapping_.public_to_internal(public_node),
            coordinate);
    }

    /** Concrete-provider optimization retained when the source exposes it. */
    template <class SourceNodes = Nodes>
    [[nodiscard]] auto is_stored(Index public_node) const
        -> decltype(std::declval<const SourceNodes&>().is_stored(Index{}), bool{}) {
        return nodes_.is_stored(mapping_.public_to_internal(public_node));
    }

private:
    void copy_node_impl(Index public_node, Scalar* target) const override {
        nodes_.copy_node(
            mapping_.public_to_internal(public_node),
            target);
    }

    const Nodes& nodes_;
    const Mapping& mapping_;
};

} // namespace highvoronoi::detail
