#pragma once

/**
 * @file nanoflann_search_tree_crtp.hpp
 * @brief nanoflann KD-tree backend with explicit backend-native search data.
 */

#include <highvoronoi/geometry/abstract_search_tree_crtp.hpp>
#include <highvoronoi/detail/nanoflann.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::geometry {

/**
 * @brief Exact nanoflann KD-tree over ordinary public mesh nodes.
 *
 * `BackendIndexT` is independent of `Mesh::Index`. The default deliberately
 * uses `std::size_t`, making the conversion boundary visible and testable even
 * when a mesh uses a compact integer such as `std::uint16_t`.
 */
template <class MeshT,
          typename BackendIndexT = std::size_t>
class NanoflannSearchTree final
    : public AbstractSearchTree<
          NanoflannSearchTree<MeshT, BackendIndexT>,
          MeshT> {
public:
    using Base = AbstractSearchTree<
        NanoflannSearchTree<MeshT, BackendIndexT>,
        MeshT>;
    using Mesh = typename Base::Mesh;
    using Scalar = typename Base::Scalar;
    using MeshIndex = typename Base::MeshIndex;
    using BackendIndex = BackendIndexT;
    using Entry = typename Base::Entry;
    using Point = detail::SearchPoint<
        Scalar,
        Base::DimensionAtCompileTime>;

    static_assert(std::is_integral_v<BackendIndex>,
                  "nanoflann backend index must be integral.");
    static_assert(std::is_unsigned_v<BackendIndex>,
                  "nanoflann backend index must be unsigned.");
    static_assert(std::is_floating_point_v<Scalar>,
                  "NanoflannSearchTree requires a floating-point scalar.");

    /**
     * @brief Public reusable data for one caller or worker thread.
     *
     * `point` and `list` are inherited public fields. The additional vectors
     * are the exact native arrays used by the nanoflann result callbacks.
     */
    struct SearchData
        : BasicSearchData<Point, MeshIndex, Scalar> {
        using Common = BasicSearchData<Point, MeshIndex, Scalar>;

        std::vector<BackendIndex> backend_indices;
        std::vector<Scalar> backend_squared_distances;

        using Common::Common;

        void reserve(std::size_t count) {
            Common::reserve(count);
            backend_indices.reserve(count);
            backend_squared_distances.reserve(count);
        }

        void clear_backend_results() noexcept {
            backend_indices.clear();
            backend_squared_distances.clear();
        }
    };

    using BackendData = SearchData;

    explicit NanoflannSearchTree(
        Mesh& mesh,
        std::size_t leaf_max_size = 10,
        unsigned int build_thread_count = 1)
        : Base(mesh),
          core_(std::make_shared<BackendCore>(
              mesh,
              leaf_max_size,
              build_thread_count)) {
        ensure_mesh_size_fits_backend();
    }

    [[nodiscard]]
    NanoflannSearchTree safe_copy() const {
        return NanoflannSearchTree(
            this->shared_mesh(),
            core_,
            this->extended_nodes().safe_copy());
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

    void rebuild_impl() {
        ensure_mesh_size_fits_backend();
        core_->index.buildIndex();
    }

    template <class Skip>
    void public_knn_impl(
        SearchData& data,
        std::size_t count,
        Scalar maximum_squared_distance,
        const Skip& skip) const {
        data.public_candidates.clear();
        data.clear_backend_results();
        if (count == 0 || this->mesh().size() == MeshIndex{0}) {
            return;
        }

        data.backend_indices.reserve(count);
        data.backend_squared_distances.reserve(count);
        data.public_candidates.reserve(count);

        KNearestResultSet<Skip> result_set(
            *this,
            data,
            count,
            maximum_squared_distance,
            skip);
        core_->index.findNeighbors(
            result_set,
            data.point.data(),
            nanoflann::SearchParameters{});

        export_backend_results(data);
    }

    template <class Skip>
    void public_inrange_impl(
        SearchData& data,
        Scalar radius_squared,
        const Skip& skip) const {
        data.public_candidates.clear();
        data.clear_backend_results();
        if (this->mesh().size() == MeshIndex{0}) {
            return;
        }

        RadiusResultSet<Skip> result_set(
            *this,
            data,
            radius_squared,
            skip);
        core_->index.findNeighbors(
            result_set,
            data.point.data(),
            nanoflann::SearchParameters{});

        export_backend_results(data);
    }

private:
    class DatasetAdaptor {
    public:
        using NodesAccess = std::remove_cv_t<
            std::remove_reference_t<
                decltype(std::declval<const Mesh&>().nodes())>>;

        explicit DatasetAdaptor(const Mesh& mesh)
            : nodes_(mesh.nodes()) {}

        [[nodiscard]] std::size_t
        kdtree_get_point_count() const noexcept {
            return static_cast<std::size_t>(nodes_.size());
        }

        [[nodiscard]] Scalar kdtree_get_pt(
            BackendIndex node,
            std::size_t coordinate) const {
            return nodes_.get_data(
                static_cast<MeshIndex>(node),
                static_cast<MeshIndex>(coordinate));
        }

        template <class BoundingBox>
        [[nodiscard]] bool kdtree_get_bbox(
            BoundingBox& /* bbox */) const noexcept {
            return false;
        }

    private:
        const NodesAccess& nodes_;
    };
    using Distance = nanoflann::L2_Simple_Adaptor<
        Scalar,
        DatasetAdaptor,
        Scalar,
        BackendIndex>;

    using KDIndex = nanoflann::KDTreeSingleIndexAdaptor<
        Distance,
        DatasetAdaptor,
        Base::DimensionAtCompileTime,
        BackendIndex>;

    struct BackendCore {
        DatasetAdaptor dataset;
        KDIndex index;

        BackendCore(
            Mesh& mesh,
            std::size_t leaf_max_size,
            unsigned int build_thread_count)
            : dataset(mesh),
              index(
                  static_cast<std::size_t>(mesh.dimension()),
                  dataset,
                  nanoflann::KDTreeSingleIndexAdaptorParams(
                      leaf_max_size,
                      nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                      build_thread_count)) {}
    };

    NanoflannSearchTree(
        Mesh& mesh,
        std::shared_ptr<BackendCore> core,
        typename Base::ExtendedNodes extended_nodes)
        : Base(mesh, std::move(extended_nodes)),
          core_(std::move(core)) {}

    template <class Skip>
    class KNearestResultSet {
    public:
        using DistanceType = Scalar;
        using IndexType = BackendIndex;

        KNearestResultSet(
            const NanoflannSearchTree& tree,
            SearchData& data,
            std::size_t count,
            Scalar maximum_squared_distance,
            const Skip& skip)
            : tree_(tree),
              data_(data),
              count_(count),
              maximum_squared_distance_(maximum_squared_distance),
              skip_(skip) {}

        [[nodiscard]] std::size_t size() const noexcept {
            return data_.backend_indices.size();
        }

        [[nodiscard]] bool empty() const noexcept {
            return data_.backend_indices.empty();
        }

        [[nodiscard]] bool full() const noexcept {
            return size() >= count_;
        }

        bool addPoint(
            Scalar squared_distance,
            BackendIndex backend_index) {
            const MeshIndex mesh_index =
                tree_.mesh_index_from_backend(backend_index);
            if (skip_(mesh_index) ||
                squared_distance > maximum_squared_distance_) {
                return true;
            }

            const std::size_t position = insertion_position(
                squared_distance,
                backend_index);

            if (size() < count_) {
                insert(position, backend_index, squared_distance);
            } else if (position < size()) {
                insert(position, backend_index, squared_distance);
                data_.backend_indices.pop_back();
                data_.backend_squared_distances.pop_back();
            }
            return true;
        }

        [[nodiscard]] Scalar worstDist() const noexcept {
            if (!full()) {
                return maximum_squared_distance_;
            }
            return std::min(
                maximum_squared_distance_,
                data_.backend_squared_distances.back());
        }

    private:
        [[nodiscard]] std::size_t insertion_position(
            Scalar squared_distance,
            BackendIndex backend_index) const {
            std::size_t first = 0;
            std::size_t last = size();

            while (first < last) {
                const std::size_t middle = first + (last - first) / 2;
                const Scalar middle_distance =
                    data_.backend_squared_distances[middle];
                const BackendIndex middle_index =
                    data_.backend_indices[middle];

                if (middle_distance < squared_distance ||
                    (middle_distance == squared_distance &&
                     middle_index < backend_index)) {
                    first = middle + 1;
                } else {
                    last = middle;
                }
            }
            return first;
        }

        void insert(
            std::size_t position,
            BackendIndex backend_index,
            Scalar squared_distance) {
            data_.backend_indices.insert(
                data_.backend_indices.begin() +
                    static_cast<std::ptrdiff_t>(position),
                backend_index);
            data_.backend_squared_distances.insert(
                data_.backend_squared_distances.begin() +
                    static_cast<std::ptrdiff_t>(position),
                squared_distance);
        }

        const NanoflannSearchTree& tree_;
        SearchData& data_;
        std::size_t count_;
        Scalar maximum_squared_distance_;
        const Skip& skip_;
    };

    template <class Skip>
    class RadiusResultSet {
    public:
        using DistanceType = Scalar;
        using IndexType = BackendIndex;

        RadiusResultSet(
            const NanoflannSearchTree& tree,
            SearchData& data,
            Scalar radius_squared,
            const Skip& skip)
            : tree_(tree),
              data_(data),
              radius_squared_(radius_squared),
              skip_(skip) {}

        [[nodiscard]] std::size_t size() const noexcept {
            return data_.backend_indices.size();
        }

        [[nodiscard]] bool empty() const noexcept {
            return data_.backend_indices.empty();
        }

        [[nodiscard]] bool full() const noexcept {
            return true;
        }

        bool addPoint(
            Scalar squared_distance,
            BackendIndex backend_index) {
            const MeshIndex mesh_index =
                tree_.mesh_index_from_backend(backend_index);
            if (!skip_(mesh_index) &&
                squared_distance < radius_squared_) {
                data_.backend_indices.push_back(backend_index);
                data_.backend_squared_distances.push_back(
                    squared_distance);
            }
            return true;
        }

        [[nodiscard]] Scalar worstDist() const noexcept {
            return radius_squared_;
        }

    private:
        const NanoflannSearchTree& tree_;
        SearchData& data_;
        Scalar radius_squared_;
        const Skip& skip_;
    };

    [[nodiscard]] MeshIndex mesh_index_from_backend(
        BackendIndex backend_index) const {
        if (static_cast<std::size_t>(backend_index) >=
            static_cast<std::size_t>(this->mesh().size())) {
            throw std::out_of_range(
                "nanoflann returned an invalid backend index.");
        }
        return static_cast<MeshIndex>(backend_index);
    }

    void export_backend_results(SearchData& data) const {
        data.public_candidates.clear();
        data.public_candidates.reserve(data.backend_indices.size());

        for (std::size_t position = 0;
             position < data.backend_indices.size();
             ++position) {
            data.public_candidates.push_back(Entry{
                mesh_index_from_backend(data.backend_indices[position]),
                data.backend_squared_distances[position]});
        }
    }

    void ensure_mesh_size_fits_backend() const {
        const std::size_t mesh_size =
            static_cast<std::size_t>(this->mesh().size());
        const std::size_t maximum_backend_index =
            static_cast<std::size_t>(
                std::numeric_limits<BackendIndex>::max());

        if (mesh_size != 0 &&
            mesh_size - std::size_t{1} > maximum_backend_index) {
            throw std::overflow_error(
                "Mesh node count does not fit nanoflann backend index type.");
        }
    }

    std::shared_ptr<BackendCore> core_;
};

} // namespace highvoronoi::geometry
