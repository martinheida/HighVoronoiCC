#pragma once

/**
 * @file composite_search_tree_crtp.hpp
 * @brief Exact structured-base plus sparse-overlay nearest-neighbour backend.
 *
 * CompositeSearchTree partitions ordinary public mesh nodes through the concrete
 * node facade's `is_stored(index)` predicate. Computed/structured nodes live in
 * one persistent exact KD index. Explicitly stored sparse nodes live in a
 * second exact KD index. `rebuild()` refreshes and rebuilds only the sparse
 * index; the structured index is intentionally immutable for the lifetime of a
 * tree instance.
 *
 * Both partitions return ordinary mesh indices and are merged before the common
 * AbstractSearchTree layer adds active boundary-extension nodes. Consequently
 * RayCaster, VoronoiWorker, SystematicVoronoi and ComputeVoronoi remain unaware
 * of the split.
 *
 * The structured KD index is the generic exact baseline. Future RepeatedCell or
 * Product providers may replace that partition locally with an analytic search
 * implementation without changing the surrounding algorithm contract.
 */

#include <highvoronoi/search/abstract_search_tree_crtp.hpp>
#include <highvoronoi/search/detail/nanoflann.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::geometry {

template <class MeshT,
          typename BackendIndexT = std::size_t>
class CompositeSearchTree final
    : public AbstractSearchTree<
          CompositeSearchTree<MeshT, BackendIndexT>,
          MeshT> {
public:
    using Base = AbstractSearchTree<
        CompositeSearchTree<MeshT, BackendIndexT>,
        MeshT>;
    using Mesh = typename Base::Mesh;
    using Scalar = typename Base::Scalar;
    using MeshIndex = typename Base::MeshIndex;
    using BackendIndex = BackendIndexT;
    using Entry = typename Base::Entry;
    using Point = detail::SearchPoint<
        Scalar,
        Base::DimensionAtCompileTime>;
    using SearchNodes = typename Base::SearchNodes;

    static_assert(std::is_integral_v<BackendIndex>);
    static_assert(std::is_unsigned_v<BackendIndex>);
    static_assert(std::is_floating_point_v<Scalar>);

    struct SearchData
        : BasicSearchData<Point, MeshIndex, Scalar> {
        using Common = BasicSearchData<Point, MeshIndex, Scalar>;

        std::vector<Entry> structured_candidates;
        std::vector<Entry> sparse_candidates;
        std::vector<BackendIndex> backend_indices;
        std::vector<Scalar> backend_squared_distances;

        using Common::Common;

        void reserve(std::size_t count) {
            Common::reserve(count);
            structured_candidates.reserve(count);
            sparse_candidates.reserve(count);
            backend_indices.reserve(count);
            backend_squared_distances.reserve(count);
        }

        void clear_composite_results() noexcept {
            structured_candidates.clear();
            sparse_candidates.clear();
            backend_indices.clear();
            backend_squared_distances.clear();
        }

        void clear_backend_results() noexcept {
            backend_indices.clear();
            backend_squared_distances.clear();
        }
    };

    using BackendData = SearchData;

    explicit CompositeSearchTree(
        Mesh& mesh,
        std::size_t leaf_max_size = 10,
        unsigned int build_thread_count = 1)
        : Base(mesh),
          core_(std::make_shared<BackendCore>(
              mesh,
              leaf_max_size,
              build_thread_count)) {
        ensure_partition_sizes_fit_backend();
    }

    [[nodiscard]] CompositeSearchTree safe_copy() const {
        return CompositeSearchTree(
            this->shared_mesh(),
            core_,
            this->extended_nodes().safe_copy());
    }

    template <class NewMesh>
    [[nodiscard]] auto rebind(NewMesh& mesh) const {
        return CompositeSearchTree<NewMesh, BackendIndex>(
            mesh,
            core_->leaf_max_size,
            core_->build_thread_count);
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

    /** @brief Refresh and rebuild only the mutable stored-node partition. */
    void rebuild_impl() {
        core_->sparse_dataset.refresh();
        ensure_sparse_size_fits_backend();
        core_->sparse_index.buildIndex();
    }

    template <class Skip>
    void public_knn_impl(
        SearchData& data,
        std::size_t count,
        Scalar maximum_squared_distance,
        const Skip& skip) const {
        data.public_candidates.clear();
        data.clear_composite_results();
        if (count == 0) {
            return;
        }

        collect_partition_knn(
            core_->structured_index,
            core_->structured_dataset,
            data,
            data.structured_candidates,
            count,
            maximum_squared_distance,
            skip);

        Scalar sparse_limit = maximum_squared_distance;
        if (data.structured_candidates.size() >= count) {
            sparse_limit = std::min(
                sparse_limit,
                data.structured_candidates[count - 1].squared_distance);
        }

        collect_partition_knn(
            core_->sparse_index,
            core_->sparse_dataset,
            data,
            data.sparse_candidates,
            count,
            sparse_limit,
            skip);

        merge_candidates(
            data.structured_candidates,
            data.sparse_candidates,
            count,
            data.public_candidates);
    }

    template <class Skip>
    void public_inrange_impl(
        SearchData& data,
        Scalar radius_squared,
        const Skip& skip) const {
        data.public_candidates.clear();
        data.clear_composite_results();

        collect_partition_inrange(
            core_->structured_index,
            core_->structured_dataset,
            data,
            data.structured_candidates,
            radius_squared,
            skip);
        collect_partition_inrange(
            core_->sparse_index,
            core_->sparse_dataset,
            data,
            data.sparse_candidates,
            radius_squared,
            skip);

        merge_candidates(
            data.structured_candidates,
            data.sparse_candidates,
            (std::numeric_limits<std::size_t>::max)(),
            data.public_candidates);
    }

private:
    template <class Nodes, class = void>
    struct HasStoredPredicate : std::false_type {};

    template <class Nodes>
    struct HasStoredPredicate<
        Nodes,
        std::void_t<decltype(
            std::declval<const Nodes&>().is_stored(
                std::declval<MeshIndex>()))>> : std::true_type {};

    static_assert(
        HasStoredPredicate<SearchNodes>::value,
        "CompositeSearchTree requires concrete nodes with is_stored(index).");

    class DatasetAdaptor {
    public:
        DatasetAdaptor(
            const Mesh& mesh,
            bool stored_partition)
            : mesh_(mesh),
              nodes_(detail::SearchTreeNodesTraits<Mesh>::get(mesh)),
              stored_partition_(stored_partition) {
            refresh();
        }

        void refresh() {
            public_indices_.clear();
            const MeshIndex count = mesh_.size();
            for (MeshIndex index = MeshIndex{0}; index < count; ++index) {
                if (nodes_.is_stored(index) == stored_partition_) {
                    public_indices_.push_back(index);
                }
            }
        }

        [[nodiscard]] std::size_t kdtree_get_point_count() const noexcept {
            return public_indices_.size();
        }

        [[nodiscard]] Scalar kdtree_get_pt(
            BackendIndex backend_index,
            std::size_t coordinate) const {
            return nodes_.get_data(
                public_index_from_backend(backend_index),
                static_cast<MeshIndex>(coordinate));
        }

        template <class BoundingBox>
        [[nodiscard]] bool kdtree_get_bbox(BoundingBox&) const noexcept {
            return false;
        }

        [[nodiscard]] MeshIndex public_index_from_backend(
            BackendIndex backend_index) const {
            const std::size_t position =
                static_cast<std::size_t>(backend_index);
            if (position >= public_indices_.size()) {
                throw std::out_of_range(
                    "Composite search backend returned an invalid index.");
            }
            return public_indices_[position];
        }

    private:
        const Mesh& mesh_;
        const SearchNodes& nodes_;
        bool stored_partition_;
        std::vector<MeshIndex> public_indices_;
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
        DatasetAdaptor structured_dataset;
        DatasetAdaptor sparse_dataset;
        KDIndex structured_index;
        KDIndex sparse_index;
        std::size_t leaf_max_size;
        unsigned int build_thread_count;

        BackendCore(
            Mesh& mesh,
            std::size_t requested_leaf_max_size,
            unsigned int requested_build_thread_count)
            : structured_dataset(mesh, false),
              sparse_dataset(mesh, true),
              structured_index(
                  static_cast<std::size_t>(mesh.dimension()),
                  structured_dataset,
                  nanoflann::KDTreeSingleIndexAdaptorParams(
                      requested_leaf_max_size,
                      nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                      requested_build_thread_count)),
              sparse_index(
                  static_cast<std::size_t>(mesh.dimension()),
                  sparse_dataset,
                  nanoflann::KDTreeSingleIndexAdaptorParams(
                      requested_leaf_max_size,
                      nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                      requested_build_thread_count)),
              leaf_max_size(requested_leaf_max_size),
              build_thread_count(requested_build_thread_count) {}
    };

    CompositeSearchTree(
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
            const DatasetAdaptor& dataset,
            SearchData& data,
            std::size_t count,
            Scalar maximum_squared_distance,
            const Skip& skip)
            : dataset_(dataset),
              data_(data),
              count_(count),
              maximum_squared_distance_(maximum_squared_distance),
              skip_(skip) {}

        [[nodiscard]] std::size_t size() const noexcept {
            return data_.backend_indices.size();
        }
        [[nodiscard]] bool empty() const noexcept { return size() == 0; }
        [[nodiscard]] bool full() const noexcept { return size() >= count_; }

        bool addPoint(Scalar squared_distance, BackendIndex backend_index) {
            const MeshIndex mesh_index =
                dataset_.public_index_from_backend(backend_index);
            if (skip_(mesh_index) ||
                squared_distance > maximum_squared_distance_) {
                return true;
            }

            std::size_t first = 0;
            std::size_t last = size();
            while (first < last) {
                const std::size_t middle = first + (last - first) / 2;
                const Scalar middle_distance =
                    data_.backend_squared_distances[middle];
                const MeshIndex middle_mesh_index =
                    dataset_.public_index_from_backend(
                        data_.backend_indices[middle]);
                if (middle_distance < squared_distance ||
                    (middle_distance == squared_distance &&
                     middle_mesh_index < mesh_index)) {
                    first = middle + 1;
                } else {
                    last = middle;
                }
            }

            if (size() < count_ || first < size()) {
                data_.backend_indices.insert(
                    data_.backend_indices.begin() +
                        static_cast<std::ptrdiff_t>(first),
                    backend_index);
                data_.backend_squared_distances.insert(
                    data_.backend_squared_distances.begin() +
                        static_cast<std::ptrdiff_t>(first),
                    squared_distance);
                if (size() > count_) {
                    data_.backend_indices.pop_back();
                    data_.backend_squared_distances.pop_back();
                }
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
        const DatasetAdaptor& dataset_;
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
            const DatasetAdaptor& dataset,
            SearchData& data,
            Scalar radius_squared,
            const Skip& skip)
            : dataset_(dataset),
              data_(data),
              radius_squared_(radius_squared),
              skip_(skip) {}

        [[nodiscard]] std::size_t size() const noexcept {
            return data_.backend_indices.size();
        }
        [[nodiscard]] bool empty() const noexcept { return size() == 0; }
        [[nodiscard]] bool full() const noexcept { return true; }

        bool addPoint(Scalar squared_distance, BackendIndex backend_index) {
            const MeshIndex mesh_index =
                dataset_.public_index_from_backend(backend_index);
            if (!skip_(mesh_index) && squared_distance < radius_squared_) {
                data_.backend_indices.push_back(backend_index);
                data_.backend_squared_distances.push_back(squared_distance);
            }
            return true;
        }

        [[nodiscard]] Scalar worstDist() const noexcept {
            return radius_squared_;
        }

    private:
        const DatasetAdaptor& dataset_;
        SearchData& data_;
        Scalar radius_squared_;
        const Skip& skip_;
    };

    template <class Skip>
    void collect_partition_knn(
        const KDIndex& index,
        const DatasetAdaptor& dataset,
        SearchData& data,
        std::vector<Entry>& output,
        std::size_t count,
        Scalar maximum_squared_distance,
        const Skip& skip) const {
        output.clear();
        data.clear_backend_results();
        if (dataset.kdtree_get_point_count() == 0 || count == 0) {
            return;
        }

        data.backend_indices.reserve(count);
        data.backend_squared_distances.reserve(count);
        KNearestResultSet<Skip> result_set(
            dataset,
            data,
            count,
            maximum_squared_distance,
            skip);
        index.findNeighbors(
            result_set,
            data.point.data(),
            nanoflann::SearchParameters{});
        export_partition_results(dataset, data, output);
    }

    template <class Skip>
    void collect_partition_inrange(
        const KDIndex& index,
        const DatasetAdaptor& dataset,
        SearchData& data,
        std::vector<Entry>& output,
        Scalar radius_squared,
        const Skip& skip) const {
        output.clear();
        data.clear_backend_results();
        if (dataset.kdtree_get_point_count() == 0) {
            return;
        }

        RadiusResultSet<Skip> result_set(
            dataset,
            data,
            radius_squared,
            skip);
        index.findNeighbors(
            result_set,
            data.point.data(),
            nanoflann::SearchParameters{});
        export_partition_results(dataset, data, output);
        Base::sort_candidates(output);
    }

    static void export_partition_results(
        const DatasetAdaptor& dataset,
        const SearchData& data,
        std::vector<Entry>& output) {
        output.clear();
        output.reserve(data.backend_indices.size());
        for (std::size_t position = 0;
             position < data.backend_indices.size();
             ++position) {
            output.push_back(Entry{
                dataset.public_index_from_backend(
                    data.backend_indices[position]),
                data.backend_squared_distances[position]});
        }
    }

    static void merge_candidates(
        const std::vector<Entry>& first,
        const std::vector<Entry>& second,
        std::size_t maximum_count,
        std::vector<Entry>& output) {
        output.clear();
        const std::size_t available = first.size() + second.size();
        const std::size_t requested =
            maximum_count == (std::numeric_limits<std::size_t>::max)()
                ? available
                : std::min(maximum_count, available);
        output.reserve(requested);

        auto left = first.begin();
        auto right = second.begin();
        while (output.size() < requested &&
               (left != first.end() || right != second.end())) {
            if (right == second.end() ||
                (left != first.end() && Base::candidate_less(*left, *right))) {
                output.push_back(*left++);
            } else {
                output.push_back(*right++);
            }
        }
    }

    static void ensure_size_fits_backend(
        std::size_t count,
        const char* message) {
        const std::size_t maximum_backend_index =
            static_cast<std::size_t>(
                (std::numeric_limits<BackendIndex>::max)());
        if (count != 0 && count - std::size_t{1} > maximum_backend_index) {
            throw std::overflow_error(message);
        }
    }

    void ensure_sparse_size_fits_backend() const {
        ensure_size_fits_backend(
            core_->sparse_dataset.kdtree_get_point_count(),
            "Sparse node count does not fit composite backend index type.");
    }

    void ensure_partition_sizes_fit_backend() const {
        ensure_size_fits_backend(
            core_->structured_dataset.kdtree_get_point_count(),
            "Structured node count does not fit composite backend index type.");
        ensure_sparse_size_fits_backend();
    }

    std::shared_ptr<BackendCore> core_;
};

} // namespace highvoronoi::geometry
