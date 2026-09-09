
#pragma once

/**
 * @file brute_force_search_tree_crtp.hpp
 * @brief Exact linear-search backend using reusable backend-native points.
 */

#include <highvoronoi/search/abstract_search_tree_crtp.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::geometry {

/**
 * @brief Exact brute-force tree over ordinary public mesh nodes.
 *
 * The backend point deliberately does not reuse `Mesh::NodePoint`: fixed
 * dimensions use `std::array`, dynamic dimensions use `std::vector`.
 */
template <class MeshT>
class BruteForceSearchTree final
    : public AbstractSearchTree<BruteForceSearchTree<MeshT>, MeshT> {
public:
    using Base = AbstractSearchTree<BruteForceSearchTree<MeshT>, MeshT>;
    using Mesh = typename Base::Mesh;
    using Scalar = typename Base::Scalar;
    using MeshIndex = typename Base::MeshIndex;
    using Entry = typename Base::Entry;
    using Point = detail::SearchPoint<
        Scalar,
        Base::DimensionAtCompileTime>;

    struct SearchData
        : BasicSearchData<Point, MeshIndex, Scalar> {
        using Common = BasicSearchData<Point, MeshIndex, Scalar>;
        using Common::Common;
    };

    using BackendData = SearchData;

    explicit BruteForceSearchTree(Mesh& mesh)
        : Base(mesh) {}

    BruteForceSearchTree(
        Mesh& mesh,
        typename Base::ExtendedNodes extended_nodes)
        : Base(mesh, std::move(extended_nodes)) {}

    [[nodiscard]]
    BruteForceSearchTree safe_copy() const {
        return BruteForceSearchTree(
            this->shared_mesh(),
            this->extended_nodes().safe_copy());
    }

    /** @brief Recreate the same backend for another compatible mesh facade. */
    template <class NewMesh>
    [[nodiscard]] auto rebind(NewMesh& mesh) const {
        return BruteForceSearchTree<NewMesh>(mesh);
    }

    [[nodiscard]] SearchData make_backend_data_impl() const {
        const std::size_t dimension =
            static_cast<std::size_t>(this->dimension());
        return SearchData(
            detail::make_search_point<
                Scalar,
                Base::DimensionAtCompileTime>(dimension),
            detail::make_search_point<
                Scalar,
                Base::DimensionAtCompileTime>(dimension));
    }

    template <class PointLike>
    void write_point_impl(
        SearchData& data,
        const PointLike& source) const {
        const std::size_t dimension =
            static_cast<std::size_t>(this->dimension());
        if (static_cast<std::size_t>(source.size()) != dimension) {
            throw std::invalid_argument(
                "Source point dimension does not match tree dimension.");
        }

        for (std::size_t coordinate = 0;
             coordinate < dimension;
             ++coordinate) {
            data.point[coordinate] =
                static_cast<Scalar>(source[coordinate]);
        }
    }

    void rebuild_impl() noexcept {}

    template <class Skip>
    void public_knn_impl(
        SearchData& data,
        std::size_t count,
        Scalar maximum_squared_distance,
        const Skip& skip) const {
        data.public_candidates.clear();
        if (count == 0) {
            return;
        }
        data.public_candidates.reserve(count);

        const auto& nodes = this->search_nodes();
        const MeshIndex node_count = this->mesh().size();

        for (MeshIndex index = MeshIndex{0};
             index < node_count;
             ++index) {
            if (skip(index)) {
                continue;
            }

            nodes.copy_node(index, data.node_point.data());
            const Scalar distance_squared =
                calculate_squared_distance(data.point, data.node_point);
            if (distance_squared > maximum_squared_distance) {
                continue;
            }

            const Entry candidate{index, distance_squared};
            const auto insertion = std::lower_bound(
                data.public_candidates.begin(),
                data.public_candidates.end(),
                candidate,
                Base::candidate_less);

            if (data.public_candidates.size() < count) {
                data.public_candidates.insert(insertion, candidate);
            } else if (insertion != data.public_candidates.end()) {
                data.public_candidates.insert(insertion, candidate);
                data.public_candidates.pop_back();
            }
        }
    }

    template <class Skip>
    void public_inrange_impl(
        SearchData& data,
        Scalar radius_squared,
        const Skip& skip) const {
        data.public_candidates.clear();

        const auto& nodes = this->search_nodes();
        const MeshIndex node_count = this->mesh().size();

        for (MeshIndex index = MeshIndex{0};
             index < node_count;
             ++index) {
            if (skip(index)) {
                continue;
            }

            nodes.copy_node(index, data.node_point.data());
            const Scalar distance_squared =
                calculate_squared_distance(data.point, data.node_point);
            if (distance_squared < radius_squared) {
                data.public_candidates.push_back(
                    Entry{index, distance_squared});
            }
        }
    }

private:
    [[nodiscard]] Scalar calculate_squared_distance(
        const Point& left,
        const Point& right) const noexcept {
        Scalar result = Scalar{0};
        const std::size_t dimension =
            static_cast<std::size_t>(this->dimension());

        for (std::size_t coordinate = 0;
             coordinate < dimension;
             ++coordinate) {
            const Scalar difference =
                left[coordinate] - right[coordinate];
            result += difference * difference;
        }
        return result;
    }
};

} // namespace highvoronoi::geometry

