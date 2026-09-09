
#pragma once

/**
 * @file abstract_search_tree_crtp.hpp
 * @brief CRTP base for allocation-reusing HighVoronoi search backends.
 *
 * A concrete search tree owns the persistent search index, while every caller
 * owns a reusable `SearchData` object. The concrete backend determines the
 * query-point representation and may add arbitrary public scratch buffers.
 * The common base combines searches over ordinary public mesh nodes with the
 * currently active boundary-extension nodes exposed by `extended_nodes()`.
 *
 * The required concrete-tree protocol is:
 *
 * @code{.cpp}
 * struct DerivedTree : AbstractSearchTree<DerivedTree, Mesh> {
 *     struct SearchData; // public, with the common fields described below
 *
 *     SearchData make_backend_data_impl() const;
 *
 *     template<class PointLike>
 *     void write_point_impl(SearchData&, const PointLike&) const;
 *
 *     template<class Skip>
 *     void public_knn_impl(SearchData&, std::size_t, Scalar, const Skip&) const;
 *
 *     template<class Skip>
 *     void public_inrange_impl(SearchData&, Scalar, const Skip&) const;
 *
 *     void rebuild_impl();
 * };
 * @endcode
 *
 * `SearchData` should derive from `BasicSearchData<BackendPoint, MeshIndex,
 * Scalar>` and may add backend-native buffers. All fields remain public so the
 * calling algorithm can fill `point`, inspect `list`, reserve storage, and
 * reuse one object across arbitrarily many searches.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi::geometry {

namespace detail {

/** Backend-owned contiguous query point for fixed or runtime dimension. */
template <typename Scalar, int Dim, bool Dynamic = (Dim < 0)>
struct SearchPointSelector;

template <typename Scalar, int Dim>
struct SearchPointSelector<Scalar, Dim, false> {
    static_assert(Dim > 0, "Fixed search dimension must be positive.");
    using type = std::array<Scalar, static_cast<std::size_t>(Dim)>;

    [[nodiscard]] static type make(std::size_t dimension) {
        if (dimension != static_cast<std::size_t>(Dim)) {
            throw std::invalid_argument(
                "Runtime dimension does not match fixed search dimension.");
        }
        return type{};
    }
};

template <typename Scalar, int Dim>
struct SearchPointSelector<Scalar, Dim, true> {
    using type = std::vector<Scalar>;

    [[nodiscard]] static type make(std::size_t dimension) {
        return type(dimension);
    }
};

template <typename Scalar, int Dim>
using SearchPoint = typename SearchPointSelector<Scalar, Dim>::type;

template <typename Scalar, int Dim>
[[nodiscard]] SearchPoint<Scalar, Dim>
make_search_point(std::size_t dimension) {
    return SearchPointSelector<Scalar, Dim>::make(dimension);
}

/**
 * @brief Select the cell-local extended-node object owned by a SearchTree.
 *
 * Real HighVoronoi mesh implementations expose `concrete_extended_nodes()`.
 * For those meshes the tree must obtain its independent state exclusively
 * through the concrete node container's `safe_copy()` semantics.
 *
 * Small mesh-like adapters used by search-tree tests may intentionally expose
 * only `extended_nodes()`. Those legacy adapters are copied by value so the
 * generic search backend remains independently testable without requiring the
 * complete mesh hierarchy.
 */
template <class Mesh, class = void>
struct SearchTreeExtendedNodesTraits {
    using type = std::remove_cv_t<
        std::remove_reference_t<
            decltype(std::declval<Mesh&>().extended_nodes())>>;

    [[nodiscard]] static type make(Mesh& mesh) {
        return type(mesh.extended_nodes());
    }
};

template <class Mesh>
struct SearchTreeExtendedNodesTraits<
    Mesh,
    std::void_t<
        decltype(std::declval<Mesh&>().concrete_extended_nodes())>> {
    using type = std::remove_cv_t<
        std::remove_reference_t<
            decltype(std::declval<Mesh&>().concrete_extended_nodes())>>;

    [[nodiscard]] static type make(Mesh& mesh) {
        return mesh.concrete_extended_nodes().safe_copy();
    }
};

/**
 * @brief Select the statically known ordinary-node facade used by a search tree.
 *
 * Mesh implementations may expose `concrete_nodes()` in addition to the
 * polymorphic `nodes()` compatibility view. Search backends prefer that
 * concrete facade so hotpath calls to get_data()/copy_node() can be inlined and
 * devirtualized. Legacy mesh adapters continue to work through nodes().
 */
template <class Mesh, class = void>
struct SearchTreeNodesTraits {
    using type = std::remove_cv_t<
        std::remove_reference_t<
            decltype(std::declval<const Mesh&>().nodes())>>;

    [[nodiscard]] static const type& get(const Mesh& mesh) noexcept {
        return mesh.nodes();
    }
};

template <class Mesh>
struct SearchTreeNodesTraits<
    Mesh,
    std::void_t<
        decltype(std::declval<const Mesh&>().concrete_nodes())>> {
    using type = std::remove_cv_t<
        std::remove_reference_t<
            decltype(std::declval<const Mesh&>().concrete_nodes())>>;

    [[nodiscard]] static const type& get(const Mesh& mesh) noexcept {
        return mesh.concrete_nodes();
    }
};

} // namespace detail

/** @brief One search result in public mesh numbering. */
template <typename MeshIndexT, typename ScalarT>
struct SearchResultEntry {
    using MeshIndex = MeshIndexT;
    using Scalar = ScalarT;

    MeshIndex index{};
    Scalar squared_distance{};
};

/**
 * @brief Public vector-like result list returned through `SearchData::list`.
 *
 * The list always uses the mesh's public index type. Backend-native indices
 * belong in additional backend-specific fields of the concrete `SearchData`.
 */
template <typename MeshIndexT, typename ScalarT>
struct SearchResultList {
    using MeshIndex = MeshIndexT;
    using Scalar = ScalarT;
    using Entry = SearchResultEntry<MeshIndex, Scalar>;

    std::vector<Entry> entries;

    [[nodiscard]] bool empty() const noexcept {
        return entries.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return entries.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return entries.capacity();
    }

    void clear() noexcept {
        entries.clear();
    }

    void reserve(std::size_t count) {
        entries.reserve(count);
    }

    [[nodiscard]] MeshIndex operator[](std::size_t position) const {
        return entries.at(position).index;
    }

    [[nodiscard]] MeshIndex index(std::size_t position) const {
        return entries.at(position).index;
    }

    [[nodiscard]] Scalar squared_distance(std::size_t position) const {
        return entries.at(position).squared_distance;
    }

    [[nodiscard]] Scalar distance(std::size_t position) const {
        using std::sqrt;
        return sqrt(entries.at(position).squared_distance);
    }
};

/**
 * @brief Common public portion of every concrete backend's reusable data.
 *
 * @tparam BackendPointT Query-point type required by the concrete backend.
 * @tparam MeshIndexT Public mesh index type returned to HighVoronoi.
 * @tparam ScalarT Distance and coordinate scalar.
 */
template <class BackendPointT,
          typename MeshIndexT,
          typename ScalarT>
struct BasicSearchData {
    using Point = BackendPointT;
    using MeshIndex = MeshIndexT;
    using Scalar = ScalarT;
    using Entry = SearchResultEntry<MeshIndex, Scalar>;
    using List = SearchResultList<MeshIndex, Scalar>;
    using CandidateList = std::vector<Entry>;

    /** Backend-native query point filled by the caller. */
    Point point;

    /** Final public results in `MeshIndex` numbering. */
    List list;

    /** Reusable point used while reading ordinary and boundary mesh nodes. */
    Point node_point;

    /** Reusable ordinary-node candidates produced by the concrete backend. */
    CandidateList public_candidates;

    /** Reusable active-boundary candidates produced by the common base. */
    CandidateList boundary_candidates;

    BasicSearchData(Point query_point, Point node_buffer)
        : point(std::move(query_point)),
          node_point(std::move(node_buffer)) {}

    /** Reserve common result storage. Concrete data may hide/extend this. */
    void reserve(std::size_t count) {
        list.reserve(count);
        public_candidates.reserve(count);
        boundary_candidates.reserve(count);
    }

    /** Clear result content without releasing any capacity. */
    void clear_results() noexcept {
        list.clear();
        public_candidates.clear();
        boundary_candidates.clear();
    }
};

/**
 * @brief Static-polymorphic base combining public and boundary-node searches.
 *
 * The base deliberately does not name `Derived::SearchData` at class scope,
 * because `Derived` is incomplete while this CRTP base is instantiated.
 */
template <class DerivedT, class MeshT>
class AbstractSearchTree {
public:
    using Derived = DerivedT;
    using Mesh = MeshT;
    using Scalar = typename Mesh::NodeScalar;
    using MeshIndex = typename Mesh::Index;
    using ExtendedNodes =
        typename detail::SearchTreeExtendedNodesTraits<Mesh>::type;
    using SearchNodes =
        typename detail::SearchTreeNodesTraits<Mesh>::type;
    using Entry = SearchResultEntry<MeshIndex, Scalar>;
    using List = SearchResultList<MeshIndex, Scalar>;
    using CandidateList = std::vector<Entry>;

    static constexpr int DimensionAtCompileTime =
        Mesh::DimensionAtCompileTime;

    struct NearestNeighbor {
        MeshIndex index{};
        Scalar distance{};
    };

    AbstractSearchTree(const AbstractSearchTree&) = delete;
    AbstractSearchTree& operator=(const AbstractSearchTree&) = delete;
    AbstractSearchTree(AbstractSearchTree&&) = delete;
    AbstractSearchTree& operator=(AbstractSearchTree&&) = delete;

    [[nodiscard]] Mesh& mesh() noexcept {
        return mesh_;
    }

    [[nodiscard]] const Mesh& mesh() const noexcept {
        return mesh_;
    }

    [[nodiscard]] MeshIndex dimension() const noexcept {
        return mesh_.dimension();
    }

    /** @brief Independent cell-local extended-node state of this tree. */
    [[nodiscard]] ExtendedNodes& extended_nodes() noexcept {
        return extended_nodes_;
    }

    [[nodiscard]] const ExtendedNodes& extended_nodes() const noexcept {
        return extended_nodes_;
    }

    /**
     * @brief Construct one empty, reusable backend-specific search workspace.
     */
    [[nodiscard]] auto make_backend_data() const {
        return derived().make_backend_data_impl();
    }

    /**
     * @brief Construct reusable data and initialize its backend-native point.
     */
    template <class PointLike>
    [[nodiscard]] auto make_backend_data(const PointLike& point) const {
        auto data = make_backend_data();
        write_point(data, point);
        return data;
    }

    /**
     * @brief Copy an arbitrary readable point into `data.point`.
     */
    template <class Data, class PointLike>
    void write_point(Data& data, const PointLike& point) const {
        require_own_data<Data>();
        derived().write_point_impl(data, point);
        require_query_dimension(data);
    }

    /** @brief Rebuild persistent backend state after public-node changes. */
    void rebuild() {
        derived().rebuild_impl();
    }

    /**
     * @brief Find the nearest ordinary or active boundary node.
     *
     * `data.point` is the query. `data.list` is cleared and receives zero or
     * one public mesh index. The same data object may be reused immediately.
     */
    template <class Data, class Skip>
    [[nodiscard]] std::optional<NearestNeighbor> nn(
        Data& data,
        const Skip& skip) const {
        require_own_data<Data>();
        require_query_dimension(data);
        clear_common_results(data);

        collect_all_boundary_candidates(data, skip);
        sort_candidates(data.boundary_candidates);

        const Scalar maximum_squared_distance =
            data.boundary_candidates.empty()
                ? infinity()
                : data.boundary_candidates.front().squared_distance;

        derived().public_knn_impl(
            data,
            std::size_t{1},
            maximum_squared_distance,
            skip);
        sort_candidates(data.public_candidates);

        merge_candidates(
            data.boundary_candidates,
            data.public_candidates,
            std::size_t{1},
            data.list.entries);

        if (data.list.empty()) {
            return std::nullopt;
        }

        using std::sqrt;
        return NearestNeighbor{
            data.list.entries.front().index,
            sqrt(data.list.entries.front().squared_distance)};
    }

    /**
     * @brief Find up to `count` nearest ordinary or active boundary nodes.
     */
    template <class Data, class Skip>
    void knn(
        Data& data,
        std::size_t count,
        const Skip& skip) const {
        require_own_data<Data>();
        require_query_dimension(data);
        clear_common_results(data);

        if (count == 0) {
            return;
        }

        collect_all_boundary_candidates(data, skip);
        sort_candidates(data.boundary_candidates);

        const Scalar maximum_squared_distance =
            data.boundary_candidates.size() >= count
                ? data.boundary_candidates[count - 1].squared_distance
                : infinity();

        derived().public_knn_impl(
            data,
            count,
            maximum_squared_distance,
            skip);
        sort_candidates(data.public_candidates);

        merge_candidates(
            data.boundary_candidates,
            data.public_candidates,
            count,
            data.list.entries);
    }

    /**
     * @brief Find all ordinary and active boundary nodes strictly inside radius.
     *
     * As requested by HighVoronoi's search semantics, the concrete public-node
     * backend runs first; active boundary entries are checked afterwards.
     */
    template <class Data, class Skip>
    void inrange(
        Data& data,
        Scalar radius,
        const Skip& skip) const {
        require_own_data<Data>();
        require_query_dimension(data);
        clear_common_results(data);

        if (radius < Scalar{0}) {
            throw std::invalid_argument("Search radius must be non-negative.");
        }

        const Scalar radius_squared = radius * radius;
        derived().public_inrange_impl(data, radius_squared, skip);
        sort_candidates(data.public_candidates);

        collect_boundary_candidates_inrange(
            data,
            radius_squared,
            skip);
        sort_candidates(data.boundary_candidates);

        merge_candidates(
            data.boundary_candidates,
            data.public_candidates,
            unlimited_count(),
            data.list.entries);
    }

    /**
     * @brief Placeholder for the later direct-cast geometry operation.
     *
     * `start` is copied into `data.point`; `data.list` is the future output.
     * The geometric acceptance and ordering rules still need to be specified.
     */
    template <class PointLike,
              class Direction,
              class Data,
              class Skip>
    void direct_cast(
        const PointLike& start,
        const Direction& /* direction */,
        Data& data,
        const Skip& /* skip */) const {
        require_own_data<Data>();
        write_point(data, start);
        throw std::logic_error(
            "direct_cast semantics have not yet been specified.");
    }

protected:
    explicit AbstractSearchTree(Mesh& mesh)
        : mesh_(mesh),
          extended_nodes_(
              detail::SearchTreeExtendedNodesTraits<Mesh>::make(mesh)) {}

    AbstractSearchTree(
        Mesh& mesh,
        ExtendedNodes extended_nodes)
        : mesh_(mesh),
          extended_nodes_(std::move(extended_nodes)) {}

    /**
    * @brief Return the shared mutable mesh reference for safe copies.
    *
    * Constness of a search-tree object does not propagate to the externally
    * owned mesh referenced by it. This accessor is intentionally protected:
    * public const tree access remains read-only through mesh() const.
    */
    [[nodiscard]] Mesh& shared_mesh() const noexcept {
        return mesh_;
    }

    /** @brief Statically known ordinary-node facade selected for hotpath access. */
    [[nodiscard]] const SearchNodes& search_nodes() const noexcept {
        return detail::SearchTreeNodesTraits<Mesh>::get(mesh_);
    }

    [[nodiscard]] static bool candidate_less(
        const Entry& left,
        const Entry& right) noexcept {
        if (left.squared_distance < right.squared_distance) {
            return true;
        }
        if (right.squared_distance < left.squared_distance) {
            return false;
        }
        return left.index < right.index;
    }

    static void sort_candidates(CandidateList& candidates) {
        std::sort(
            candidates.begin(),
            candidates.end(),
            candidate_less);
    }

private:
    template <class Data>
    static void require_own_data() {
        using Actual = std::remove_cv_t<std::remove_reference_t<Data>>;
        using Expected = typename Derived::SearchData;
        static_assert(
            std::is_same_v<Actual, Expected>,
            "The supplied search data belongs to another tree type.");
    }

    template <class Data>
    void require_query_dimension(const Data& data) const {
        if (data.point.size() != static_cast<std::size_t>(dimension())) {
            throw std::invalid_argument(
                "Search point dimension does not match tree dimension.");
        }
        if (data.node_point.size() != static_cast<std::size_t>(dimension())) {
            throw std::logic_error(
                "Backend node-point buffer has the wrong dimension.");
        }
    }

    template <class Data>
    static void clear_common_results(Data& data) noexcept {
        data.list.clear();
        data.public_candidates.clear();
        data.boundary_candidates.clear();
    }

    template <class Data>
    [[nodiscard]] Scalar squared_distance_to_loaded_node(
        const Data& data) const noexcept {
        Scalar result = Scalar{0};
        const std::size_t dim = static_cast<std::size_t>(dimension());

        for (std::size_t coordinate = 0;
             coordinate < dim;
             ++coordinate) {
            const Scalar difference =
                static_cast<Scalar>(data.point[coordinate]) -
                static_cast<Scalar>(data.node_point[coordinate]);
            result += difference * difference;
        }
        return result;
    }

    template <class Data, class Skip>
    void collect_all_boundary_candidates(
        Data& data,
        const Skip& skip) const {
        const auto& extended_nodes = extended_nodes_;
        const MeshIndex count = extended_nodes.active_boundary_size();
        data.boundary_candidates.reserve(
            static_cast<std::size_t>(count));

        for (MeshIndex position = MeshIndex{0};
             position < count;
             ++position) {
            const MeshIndex index =
                extended_nodes.active_boundary_index(position);
            if (skip(index)) {
                continue;
            }

            extended_nodes.copy_node(index, data.node_point.data());
            data.boundary_candidates.push_back(
                Entry{index, squared_distance_to_loaded_node(data)});
        }
    }

    template <class Data, class Skip>
    void collect_boundary_candidates_inrange(
        Data& data,
        Scalar radius_squared,
        const Skip& skip) const {
        const auto& extended_nodes = extended_nodes_;
        const MeshIndex count = extended_nodes.active_boundary_size();
        data.boundary_candidates.reserve(
            static_cast<std::size_t>(count));

        for (MeshIndex position = MeshIndex{0};
             position < count;
             ++position) {
            const MeshIndex index =
                extended_nodes.active_boundary_index(position);
            if (skip(index)) {
                continue;
            }

            extended_nodes.copy_node(index, data.node_point.data());
            const Scalar squared_distance =
                squared_distance_to_loaded_node(data);
            if (squared_distance < radius_squared) {
                data.boundary_candidates.push_back(
                    Entry{index, squared_distance});
            }
        }
    }

    static void merge_candidates(
        const CandidateList& first,
        const CandidateList& second,
        std::size_t maximum_count,
        CandidateList& output) {
        output.clear();
        const std::size_t available = first.size() + second.size();
        const std::size_t requested =
            maximum_count == unlimited_count()
                ? available
                : std::min(maximum_count, available);
        output.reserve(requested);

        auto left = first.begin();
        auto right = second.begin();

        while (output.size() < requested &&
               (left != first.end() || right != second.end())) {
            if (right == second.end() ||
                (left != first.end() && candidate_less(*left, *right))) {
                output.push_back(*left);
                ++left;
            } else {
                output.push_back(*right);
                ++right;
            }
        }
    }

    [[nodiscard]] static constexpr std::size_t
    unlimited_count() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }

    [[nodiscard]] static constexpr Scalar infinity() noexcept {
        if constexpr (std::numeric_limits<Scalar>::has_infinity) {
            return std::numeric_limits<Scalar>::infinity();
        } else {
            return std::numeric_limits<Scalar>::max();
        }
    }

    [[nodiscard]] Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    [[nodiscard]] const Derived& derived() const noexcept {
        return static_cast<const Derived&>(*this);
    }

    Mesh& mesh_;
    ExtendedNodes extended_nodes_;
};

} // namespace highvoronoi::geometry

