#pragma once

/**
 * @warning PERFORMANCE / ALLOCATION STATUS
 *
 * The Polygon mathematical algorithm remains the validated Julia-faithful
 * reference implementation. Its temporary storage, however, no longer uses
 * unordered_map<vector<Index>, PolyEdge>, shared_ptr payloads, or an owning
 * per-cell vertex snapshot.
 *
 * Mesh vertices are read directly through the cell vertex range. Recursive
 * PolyEdge occurrences are stored in one rewindable worker-local block database;
 * Julia-style shared integral vectors live in a second rewindable payload
 * database and are referenced by compact addresses. Every logical recursive
 * dictionary owns only a recyclable exact hash index from edge signature to
 * edge-record address. Database blocks and dictionary capacities are retained
 * from cell to cell within one PolygonWorkspace.
 *
 * Capacity growth can still allocate when a worker encounters a cell larger
 * than any previously processed by that workspace. Further optimization should
 * be measured separately and must not change the mathematical recursion.
 */

/**
 * @file polygon_integrator.hpp
 * @brief Julia-faithful recursive polygon/polytope integration.
 *
 * This ports HighVoronoi.jl's Polygon_Integrator / iterative_volume algorithm
 * onto the current transactional C++ IntegrationView.
 *
 * The mathematical structure is intentionally preserved:
 *
 * - top-level Voronoi edges are collected per true neighbour;
 * - lower-dimensional faces are built recursively from shared edge keys;
 * - IterativeDimensionChecker rejects linearly dependent recursion paths in
 *   degenerate configurations;
 * - IncrementalMinors recycles all lower-order subdeterminants while descending
 *   through the cone hierarchy;
 * - a shared interface is computed on exactly one update-cell side and reused
 *   by the later cell, matching Julia's `buffer > _Cell` ownership rule;
 * - CLEAN neighbours are completed in cleanup_cell from their persistent
 *   interface data, matching Julia's `!(n in calculate)` branch.
 *
 * The old Julia FastEdgeIterator is NOT reimplemented here. The current C++
 * EdgeIterator is already its maintained port. OnCellEdges exposes exactly the
 * cell-local general/degenerate edges needed by Polygon_Integrator and returns
 * the complete supporting edge through EdgeView::full_indices().
 */

#include <highvoronoi/algorithm/edge_iterator.hpp>
#include <highvoronoi/integration/detail/incremental_minors.hpp>
#include <highvoronoi/integration/detail/iterative_dimension_checker.hpp>
#include <highvoronoi/integration/detail/polygon_edge_storage.hpp>
#include <highvoronoi/integration/high_voronoi_integration_view.hpp>
#include <highvoronoi/integration/integration_view.hpp>
#include <highvoronoi/integration/integrator.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/** Marker used when PolygonAlgorithm computes geometry only. */
struct NoPolygonIntegrand final {};

namespace detail {

template <class T, class = void>
struct PolygonIndexedResult : std::false_type {};

template <class T>
struct PolygonIndexedResult<
    T,
    std::void_t<
        decltype(std::declval<const T&>().size()),
        decltype(std::declval<const T&>()[std::declval<std::size_t>()])>>
    : std::true_type {};

template <class Function, class Point, class Scalar>
void evaluate_polygon_integrand(
    const Function& function,
    const Point& point,
    std::vector<Scalar>& output) {
    const auto value = function(point);
    using Value = std::decay_t<decltype(value)>;

    if constexpr (std::is_arithmetic_v<Value>) {
        if (output.size() != std::size_t{1}) {
            throw std::logic_error(
                "Scalar PolygonAlgorithm integrand requires one component");
        }
        output[0] = static_cast<Scalar>(value);
    } else if constexpr (PolygonIndexedResult<Value>::value) {
        if (static_cast<std::size_t>(value.size()) != output.size()) {
            throw std::logic_error(
                "PolygonAlgorithm integrand component count mismatch");
        }
        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] = static_cast<Scalar>(value[i]);
        }
    } else {
        static_assert(
            std::is_arithmetic_v<Value> ||
                PolygonIndexedResult<Value>::value,
            "PolygonAlgorithm integrand must return a scalar or indexable vector");
    }
}

template <class View, class = void>
struct PolygonViewHasReferences : std::false_type {};

template <class View>
struct PolygonViewHasReferences<
    View,
    std::void_t<decltype(std::declval<const View&>().is_reference(
        std::declval<typename View::Index>()))>> : std::true_type {};

template <class View>
[[nodiscard]] bool polygon_is_reference(
    const View& view,
    typename View::Index index) {
    if constexpr (PolygonViewHasReferences<View>::value) {
        if (static_cast<std::size_t>(index) <
            static_cast<std::size_t>(view.size())) {
            return view.is_reference(index);
        }
    }
    return false;
}

template <class View, class = void>
struct PolygonViewHasGlobalUpdateQuery : std::false_type {};

template <class View>
struct PolygonViewHasGlobalUpdateQuery<
    View,
    std::void_t<decltype(std::declval<const View&>().is_update_cell(
        std::declval<typename View::Index>()))>> : std::true_type {};

template <class View>
[[nodiscard]] bool polygon_is_update_cell(
    const View& view,
    typename View::Index index,
    std::size_t local_update_count) {
    if (static_cast<std::size_t>(index) >=
        static_cast<std::size_t>(view.size())) {
        return false;
    }
    if constexpr (PolygonViewHasGlobalUpdateQuery<View>::value) {
        return view.is_update_cell(index);
    } else {
        // HighVoronoi keeps the established serial prefix semantics until its
        // own worker-view partitioning is implemented explicitly.
        return static_cast<std::size_t>(index) < local_update_count;
    }
}

/**
 * Scratch and recursive implementation for one concrete IntegrationPass type.
 *
 * The object is created once per integration pass because its EdgeIterator must
 * remain bound to that pass's temporary reordered extended-node provider. All
 * heavy buffers are then recycled from cell to cell.
 */
template <class Pass, class Function>
class PolygonWorkspace final {
public:
    using View = typename Pass::View;
    using Update = typename Pass::Update;
    using Index = typename View::Index;
    using AreaScalar = typename View::AreaScalar;
    using IntegralScalar = typename View::IntegralScalar;
    using Nodes = std::remove_reference_t<
        decltype(std::declval<Update&>().extended_nodes())>;
    using Iterator = EdgeIterator<Nodes>;
    using Point = typename Iterator::Point;

private:
    using EdgeDatabase = PolygonEdgeDatabase<Index, Point>;
    using PayloadDatabase = PolygonPayloadDatabase<IntegralScalar>;
    using EdgeMap = PolygonEdgeDictionary<EdgeDatabase>;
    using EdgeAddress = typename EdgeDatabase::Address;
    using PayloadAddress = typename PayloadDatabase::Address;

    struct EdgeState {
        Point r1;
        Point r2;
        PayloadAddress payload{0};

        explicit EdgeState(std::size_t dimension)
            : r1(make_point(dimension)),
              r2(make_point(dimension)) {}
    };


public:
    PolygonWorkspace(
        View& view,
        Nodes& nodes,
        std::size_t integral_components,
        const Function& function)
        : view_(&view),
          nodes_(&nodes),
          dimension_(static_cast<std::size_t>(nodes.dimension())),
          components_(integral_components),
          function_(&function),
          edge_database_(dimension_),
          payload_database_(integral_components),
          checker_(nodes, dimension_),
          minors_(dimension_),
          edge_state_(dimension_),
          vertex_position_buffer_(make_point(dimension_)),
          empty_vector_(make_point(dimension_)),
          cell_point_(make_point(dimension_)),
          node_point_(make_point(dimension_)),
          evaluation_point_(make_point(dimension_)),
          function_value_(integral_components),
          second_function_value_(integral_components),
          payload_value_(integral_components),
          clean_cell_scratch_() {
        if (dimension_ < std::size_t{2}) {
            throw std::invalid_argument(
                "PolygonAlgorithm requires dimension >= 2");
        }
        lower_lists_.resize(dimension_ - 1);
        active_neighbours_levels_.resize(dimension_ - 1);
        taboo_.assign(dimension_, invalid_index());
    }

    void integrate_cell(
        Pass& pass,
        std::size_t position) {
        Update& update = pass.writable_cell(position);
        reset_cell(update);

        auto& data = update.data();
        auto& areas = data.area();
        auto& bulk = data.bulk_integral();
        auto& interface_integrals =
            data.interface_integral();

        require_layout(pass, update);

        AreaScalar volume = AreaScalar{0};

        iterative_volume(
            pass,
            position,
            dimension_,
            top_dummy_,
            top_dummy_,
            cell_point_,
            volume,
            bulk,
            areas.data(),
            interface_integrals.empty()
                ? nullptr
                : interface_integrals.data());

        data.set_volume(volume);
    }

    /**
     * Julia Polygon cleanup: complete contributions from neighbours outside
     * the current calculate/update set using persistent CLEAN interface data.
     */
    void cleanup_cell(
        Pass& pass,
        std::size_t position) {
        Update& update = pass.writable_cell(position);
        auto& data = update.data();
        auto& areas = data.area();
        auto& bulk = data.bulk_integral();
        auto& interface_integrals =
            data.interface_integral();
        const auto& neighbours = update.neighbours();

        // prepare_cleanup() has restored this cell's extended-node geometry.
        nodes_->copy_node(
            update.cell(),
            cell_point_.data());

        for (std::size_t ordinal = 0;
             ordinal < neighbours.size();
             ++ordinal) {
            const Index neighbour = neighbours[ordinal];
            if (is_calculate_neighbour(pass, neighbour)) {
                continue;
            }

            if (!is_clean_integral_neighbour(pass, neighbour)) {
                throw std::logic_error(
                    "Polygon cleanup encountered unsupported geometry-only neighbour");
            }

            nodes_->copy_node(neighbour, node_point_.data());
            const AreaScalar distance =
                static_cast<AreaScalar>(
                    typename Point::Scalar{0.5} *
                    (cell_point_ - node_point_).norm());

            const bool has_area_data =
                areas.at(ordinal) > AreaScalar{0};

            AreaScalar area = areas.at(ordinal);
            std::size_t opposite_ordinal = 0;

            if (!has_area_data) {
                if (!update.read_integral_cell(
                        neighbour,
                        clean_cell_scratch_)) {
                    throw std::logic_error(
                        "Polygon cleanup could not read CLEAN opposite cell");
                }
                opposite_ordinal = reciprocal_ordinal(
                    neighbours,
                    ordinal,
                    static_cast<Index>(position),
                    clean_cell_scratch_.neighbours());
                area = clean_cell_scratch_.area().at(opposite_ordinal);
            }

            const AreaScalar cone_volume =
                area * distance /
                static_cast<AreaScalar>(dimension_);
            data.set_volume(data.volume() + cone_volume);
            areas[ordinal] = area;

            if (has_function()) {
                IntegralScalar* target =
                    interface_block(interface_integrals, ordinal);

                if (!has_area_data) {
                    copy_interface_block(
                        clean_cell_scratch_.interface_integral(),
                        opposite_ordinal,
                        target);
                }

                if (!bulk.empty()) {
                    const IntegralScalar factor =
                        static_cast<IntegralScalar>(
                            distance /
                            static_cast<AreaScalar>(dimension_));
                    for (std::size_t component = 0;
                         component < components_;
                         ++component) {
                        bulk[component] +=
                            target[component] * factor;
                    }
                }
            }
        }
    }

private:
    [[nodiscard]] static Index invalid_index() noexcept {
        return (std::numeric_limits<Index>::max)();
    }

    [[nodiscard]] static Point make_point(
        std::size_t dimension) {
        Point point;
        if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
            point.resize(static_cast<Eigen::Index>(dimension));
        }
        point.setZero();
        return point;
    }

    void reset_cell(Update& update) {
        neighbours_.assign(
            update.neighbours().begin(),
            update.neighbours().end());

        // All recursive Polygon state is cell-local. Rewind the shared worker
        // databases but keep every allocated block for the next cell.
        edge_database_.rewind();
        payload_database_.rewind();

        resize_and_clear_maps(
            top_lists_,
            neighbours_.size());
        for (auto& level : lower_lists_) {
            clear_maps(level);
        }

        std::fill(
            taboo_.begin(),
            taboo_.end(),
            invalid_index());
        prepare_map(empty_map_);
        empty_map_.clear();
        prepare_map(top_dummy_);
        top_dummy_.clear();

        nodes_->copy_node(
            update.cell(),
            cell_point_.data());

        // Julia passes vertices_iterator(mesh, cell) directly to the checker
        // and then iterates the mesh vertices again while building top edges.
        // Do the same: there is no owning vertex snapshot in PolygonWorkspace.
        const auto vertices = update.vertices();
        (void)checker_.reset(
            neighbours_,
            update.cell(),
            vertices);
    }

    template <class PassLike>
    void require_layout(
        const PassLike& pass,
        const Update& update) const {
        const auto& data = update.data();
        const auto& integral_data =
            pass.view().integral().data();

        // This is the same data dependency as Julia enable(volume=true):
        // polygon volume integration keeps per-interface area state.
        if (!integral_data.stores_volume() ||
            !integral_data.stores_area()) {
            throw std::logic_error(
                "PolygonAlgorithm requires volume and area storage");
        }

        if (data.area().size() !=
            update.neighbours().size()) {
            throw std::logic_error(
                "Polygon area storage is not neighbour-aligned");
        }

        if (integral_data.stores_bulk_integral() &&
            !integral_data.stores_interface_integral()) {
            throw std::logic_error(
                "Polygon bulk integration requires interface-integral storage, "
                "matching Julia enable_integral_data()");
        }

        if (has_function()) {
            if (components_ == 0) {
                throw std::logic_error(
                    "Polygon integrand supplied with zero integral components");
            }
            if (!integral_data.stores_interface_integral()) {
                throw std::logic_error(
                    "Polygon integrand requires interface-integral storage");
            }
            if (data.interface_integral().size() !=
                update.neighbours().size() * components_) {
                throw std::logic_error(
                    "Polygon interface-integral layout mismatch");
            }
            if (integral_data.stores_bulk_integral() &&
                data.bulk_integral().size() != components_) {
                throw std::logic_error(
                    "Polygon bulk-integral layout mismatch");
            }
        } else {
            if (integral_data.stores_bulk_integral() ||
                integral_data.stores_interface_integral()) {
                throw std::logic_error(
                    "Polygon integral storage enabled without integrand");
            }
        }
    }

    [[nodiscard]] bool has_function() const noexcept {
        return !std::is_same_v<Function, NoPolygonIntegrand>;
    }

    [[nodiscard]] bool is_calculate_neighbour(
        const Pass& pass,
        Index neighbour) const {
        const std::size_t value =
            static_cast<std::size_t>(neighbour);

        // Ordinary worker views keep the complete global NEW/DIRTY snapshot
        // even though only this worker's assigned slice is at the local prefix.
        // Thus cross-worker dirty neighbours are recomputed from geometry rather
        // than being mistaken for persistent CLEAN cells.
        if (polygon_is_update_cell(
                pass.view(),
                neighbour,
                pass.size())) {
            return true;
        }

        // HighVoronoi periodic reference nodes are geometry-only but must be
        // calculated like Julia's extended/boundary generators.
        if (polygon_is_reference(pass.view(), neighbour)) {
            return true;
        }

        // Boundary mirrors are presented after the complete integration view.
        if (value >=
            static_cast<std::size_t>(pass.view().size())) {
            return true;
        }

        // Remaining visible cells are CLEAN and intentionally left for cleanup.
        return false;
    }

    [[nodiscard]] bool is_clean_integral_neighbour(
        const Pass& pass,
        Index neighbour) const {
        const std::size_t value =
            static_cast<std::size_t>(neighbour);
        return value <
                   static_cast<std::size_t>(
                       pass.view().size()) &&
               !polygon_is_update_cell(
                   pass.view(),
                   neighbour,
                   pass.size()) &&
               !polygon_is_reference(
                   pass.view(),
                   neighbour);
    }

    [[nodiscard]] std::size_t neighbour_index(
        const std::vector<Index>& neighbours,
        Index value) const noexcept {
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            if (neighbours[i] == value) {
                return i;
            }
        }
        return npos;
    }

    [[nodiscard]] static bool contains(
        const std::vector<Index>& values,
        Index value) {
        return std::find(
                   values.begin(),
                   values.end(),
                   value) != values.end();
    }

    [[nodiscard]] std::size_t first_relevant_edge_index(
        const std::vector<Index>& edge,
        Index cell,
        const std::vector<Index>& neighbours) const noexcept {
        for (const Index generator : edge) {
            if (generator == cell) {
                continue;
            }
            const std::size_t index =
                neighbour_index(
                    neighbours,
                    generator);
            if (index != npos) {
                return index;
            }
        }
        return npos;
    }

    void read_edge(
        EdgeAddress address,
        EdgeState& state) const {
        edge_database_.read_edge(
            address,
            state.r1,
            state.r2,
            state.payload);
    }

    [[nodiscard]] EdgeAddress extend_poly_edge(
        const std::vector<Index>& signature,
        EdgeAddress old_address,
        const Point& point) {
        read_edge(old_address, edge_state_);

        Point& first = edge_state_.r1;
        Point& second = edge_state_.r2;
        const Point direction = second - first;
        bool changed = false;

        if ((point - first).dot(direction) <=
            typename Point::Scalar{0}) {
            if ((point - first).squaredNorm() !=
                typename Point::Scalar{0}) {
                first = point;
                changed = true;
            }
        } else if ((point - second).dot(direction) >
                   typename Point::Scalar{0}) {
            second = point;
            changed = true;
        }

        if (!changed) {
            return old_address;
        }

        return edge_database_.push(
            signature,
            first,
            second,
            edge_state_.payload);
    }

    /**
     * Direct port of Julia queue_integral_edge().
     *
     * Each logical dictionary owns only its hash/index. PolyEdge occurrences
     * live in one rewindable worker-local database. Identical dictionary values
     * may share one immutable record; extending one value appends a new record
     * and redirects only the dictionaries Julia assigns to.
     */
    void queue_integral_edge(
        std::vector<EdgeMap>& dictionaries,
        const std::vector<Index>& edge,
        const Point& position,
        Index cell) {
        const std::size_t first_index =
            first_relevant_edge_index(
                edge,
                cell,
                neighbours_);
        if (first_index == npos) {
            return;
        }

        const Index first_neighbour =
            neighbours_[first_index];
        EdgeAddress found_address = EdgeAddress{0};

        if (dictionaries[first_index].find_address(
                edge,
                found_address)) {
            read_edge(found_address, edge_state_);
            if ((edge_state_.r1 - position).squaredNorm() ==
                typename Point::Scalar{0}) {
                return;
            }

            const EdgeAddress coordinates =
                extend_poly_edge(
                    edge,
                    found_address,
                    position);

            for (const Index generator : edge) {
                if (generator <= cell ||
                    !contains(neighbours_, generator)) {
                    continue;
                }

                const std::size_t index =
                    neighbour_index(
                        neighbours_,
                        generator);
                if (index == npos) {
                    continue;
                }
                dictionaries[index].insert_or_assign(
                    edge,
                    coordinates);
            }
            return;
        }

        const PayloadAddress payload =
            payload_database_.allocate();
        const EdgeAddress coordinates =
            edge_database_.push_point(
                edge,
                position,
                payload);

        for (const Index generator : edge) {
            if ((generator <= cell &&
                 generator != first_neighbour) ||
                !contains(neighbours_, generator)) {
                continue;
            }

            const std::size_t index =
                neighbour_index(
                    neighbours_,
                    generator);
            if (index == npos) {
                continue;
            }
            (void)dictionaries[index].insert(
                edge,
                coordinates);
        }
    }

    void collect_top_edges(
        Update& update,
        Index cell) {
        Iterator& iterator =
            checker_.edge_iterator();
        edge_buffer_.clear();

        // Read directly from the mesh-owned vertex range, matching Julia's
        // second pass over vertices_iterator(mesh, cell). No owning snapshot.
        for (const auto& vertex : update.vertices()) {
            iterator.reset(
                vertex.sigma,
                vertex.position,
                cell,
                typename Iterator::OnCellEdges{});

            for (std::size_t coordinate = 0;
                 coordinate < dimension_;
                 ++coordinate) {
                vertex_position_buffer_[
                    static_cast<Eigen::Index>(coordinate)] =
                    static_cast<typename Point::Scalar>(
                        vertex.position[
                            static_cast<Eigen::Index>(coordinate)]);
            }

            while (const auto candidate = iterator.next()) {
                edge_buffer_.clear();

                // Julia's degenerate branch converts the FEI full supporting
                // edge to global generators and then filters it to true cell
                // neighbours. OnCellEdges already handles the general branch.
                for (const Index generator :
                     candidate->full_indices()) {
                    if (contains(
                            neighbours_,
                            generator)) {
                        edge_buffer_.push_back(
                            generator);
                    }
                }

                edge_buffer_.push_back(cell);
                std::sort(
                    edge_buffer_.begin(),
                    edge_buffer_.end());
                if (edge_buffer_.empty() ||
                    edge_buffer_.back() <= cell) {
                    continue;
                }

                queue_integral_edge(
                    top_lists_,
                    edge_buffer_,
                    vertex_position_buffer_,
                    cell);
            }
        }
    }

    void evaluate_top_edges() {
        if (!has_function()) {
            return;
        }

        for (const auto& dictionary : top_lists_) {
            for (const auto& entry : dictionary.entries()) {
                read_edge(entry.address, edge_state_);

                evaluate(edge_state_.r1, function_value_);
                evaluate(
                    edge_state_.r2,
                    second_function_value_);

                for (std::size_t component = 0;
                     component < components_;
                     ++component) {
                    payload_value_[component] =
                        IntegralScalar{0.5} *
                        (function_value_[component] +
                         second_function_value_[component]);
                }
                payload_database_.write(
                    edge_state_.payload,
                    payload_value_.data(),
                    components_);
            }
        }
    }

    void evaluate(
        const Point& point,
        std::vector<IntegralScalar>& output) {
        if constexpr (
            std::is_same_v<
                Function,
                NoPolygonIntegrand>) {
            if (!output.empty()) {
                throw std::logic_error(
                    "PolygonAlgorithm attempted function evaluation "
                    "without an integrand");
            }
        } else {
            evaluation_point_ = point;
            wrap_integration_evaluation_point(
                *view_,
                evaluation_point_);
            evaluate_polygon_integrand(
                *function_,
                evaluation_point_,
                output);
        }
    }

    /**
     * Direct Julia `midpoint(vertslist, vertslist2, ..., cell_center)`.
     *
     * Both PolyEdge endpoints contribute, then the absolute cell generator is
     * subtracted to obtain the local vector used by IncrementalMinors. The
     * cell generator remains the recursion origin at every level, exactly as
     * Julia's persistent `vector` argument does.
     */
    [[nodiscard]] Point midpoint(
        const EdgeMap& first,
        const EdgeMap& second,
        const Point& cell_center) {
        empty_vector_.setZero();

        for (const auto& entry : first.entries()) {
            read_edge(entry.address, edge_state_);
            empty_vector_ += edge_state_.r1;
            empty_vector_ += edge_state_.r2;
        }
        for (const auto& entry : second.entries()) {
            read_edge(entry.address, edge_state_);
            empty_vector_ += edge_state_.r1;
            empty_vector_ += edge_state_.r2;
        }

        const std::size_t count =
            first.size() + second.size();
        if (count == 0) {
            throw std::logic_error(
                "Polygon midpoint requested for empty face");
        }

        empty_vector_ *=
            typename Point::Scalar{0.5} /
            static_cast<typename Point::Scalar>(count);
        empty_vector_ -= cell_center;
        return empty_vector_;
    }

    struct SupEdge {
        Point first;
        Point second;
        PayloadAddress payload{0};

        explicit SupEdge(std::size_t dimension)
            : first(make_point(dimension)),
              second(make_point(dimension)) {}
    };

    /** Direct port of Julia get_sup_edge(). The input dictionary is consumed. */
    [[nodiscard]] SupEdge get_sup_edge(
        EdgeMap& edges) {
        if (edges.empty()) {
            throw std::logic_error(
                "Polygon 1D recursion received no edge segment");
        }

        SupEdge result(dimension_);
        const EdgeAddress first_address =
            edges.pop_address();
        edge_database_.read_edge(
            first_address,
            result.first,
            result.second,
            result.payload);
        const Point direction =
            result.second - result.first;

        while (!edges.empty()) {
            const EdgeAddress address =
                edges.pop_address();
            read_edge(address, edge_state_);

            const Point* endpoints[2]{
                &edge_state_.r1,
                &edge_state_.r2};
            for (const Point* point : endpoints) {
                if ((*point - result.first).dot(direction) <
                    typename Point::Scalar{0}) {
                    result.first = *point;
                } else if ((*point - result.second).dot(direction) >
                           typename Point::Scalar{0}) {
                    result.second = *point;
                }
            }
        }

        return result;
    }

    void iterative_volume(
        Pass& pass,
        std::size_t position,
        std::size_t current_dimension,
        EdgeMap& vertices,
        EdgeMap& vertices2,
        const Point& reference_center,
        AreaScalar& volume,
        std::vector<IntegralScalar>& bulk,
        AreaScalar* area,
        IntegralScalar* interface_values) {
        const Index cell =
            static_cast<Index>(position);

        if (current_dimension == std::size_t{1}) {
            SupEdge edge =
                get_sup_edge(vertices);

            Point first_local =
                edge.first - reference_center;
            Point second_local =
                edge.second - reference_center;

            minors_.update(
                dimension_ - 1,
                first_local);
            minors_.update(
                dimension_,
                second_local);

            const AreaScalar measure =
                static_cast<AreaScalar>(
                    std::abs(minors_.determinant()));

            area[0] += measure;

            if (has_function()) {
                payload_database_.read(
                    edge.payload,
                    payload_value_.data(),
                    components_);
                for (std::size_t component = 0;
                     component < components_;
                     ++component) {
                    interface_values[component] +=
                        static_cast<IntegralScalar>(
                            measure) *
                        payload_value_[component];
                }
            }
            return;
        }

        if (current_dimension == dimension_) {
            collect_top_edges(
                pass.writable_cell(position),
                cell);
            evaluate_top_edges();

            taboo_[dimension_ - 1] = cell;

            for (std::size_t ordinal = 0;
                 ordinal < neighbours_.size();
                 ++ordinal) {
                const Index neighbour =
                    neighbours_[ordinal];

                if (!is_calculate_neighbour(
                        pass,
                        neighbour)) {
                    top_lists_[ordinal].clear();
                    continue;
                }

                EdgeMap& buffer =
                    top_lists_[ordinal];

                if (neighbour > cell &&
                    buffer.empty()) {
                    continue;
                }

                taboo_[dimension_ - 2] =
                    neighbour;
                neighbours_[ordinal] =
                    invalid_index();

                AreaScalar raw_area =
                    AreaScalar{0};
                IntegralScalar* interface_block_ptr =
                    has_function()
                        ? interface_values +
                            ordinal * components_
                        : nullptr;

                if (neighbour > cell) {
                    (void)checker_.set_dimension(
                        1,
                        cell,
                        neighbour);

                    if (has_function()) {
                        std::fill_n(
                            interface_block_ptr,
                            components_,
                            IntegralScalar{0});
                    }

                    const Point center_relative =
                        midpoint(
                            buffer,
                            empty_map_,
                            cell_point_);
                    Point center_absolute =
                        center_relative + cell_point_;

                    iterative_volume(
                        pass,
                        position,
                        current_dimension - 1,
                        buffer,
                        empty_map_,
                        cell_point_,
                        volume,
                        bulk,
                        &raw_area,
                        interface_block_ptr);

                    neighbours_[ordinal] =
                        neighbour;
                    buffer.clear();

                    nodes_->copy_node(
                        neighbour,
                        node_point_.data());
                    const AreaScalar distance =
                        static_cast<AreaScalar>(
                            typename Point::Scalar{0.5} *
                            (cell_point_ - node_point_).norm());
                    if (!(distance > AreaScalar{0})) {
                        throw std::logic_error(
                            "Polygon interface has zero generator distance");
                    }

                    AreaScalar factor =
                        AreaScalar{1};
                    for (std::size_t k = 1;
                         k < current_dimension;
                         ++k) {
                        factor /= static_cast<AreaScalar>(k);
                    }

                    const AreaScalar cone_volume =
                        raw_area * factor /
                        static_cast<AreaScalar>(
                            current_dimension);
                    volume += cone_volume;

                    factor /= distance;
                    raw_area *= factor;
                    area[ordinal] = raw_area;

                    if (has_function()) {
                        for (std::size_t component = 0;
                             component < components_;
                             ++component) {
                            interface_block_ptr[component] *=
                                static_cast<IntegralScalar>(
                                    factor);
                        }

                        evaluate(
                            center_absolute,
                            function_value_);

                        const IntegralScalar boundary_weight =
                            static_cast<IntegralScalar>(
                                static_cast<AreaScalar>(
                                    current_dimension - 1) /
                                static_cast<AreaScalar>(
                                    current_dimension));
                        const IntegralScalar center_weight =
                            static_cast<IntegralScalar>(
                                raw_area /
                                static_cast<AreaScalar>(
                                    current_dimension));

                        for (std::size_t component = 0;
                             component < components_;
                             ++component) {
                            interface_block_ptr[component] =
                                interface_block_ptr[component] *
                                    boundary_weight +
                                function_value_[component] *
                                    center_weight;
                        }

                        if (!bulk.empty()) {
                            evaluate(
                                cell_point_,
                                function_value_);

                            const IntegralScalar center_factor =
                                static_cast<IntegralScalar>(
                                    cone_volume /
                                    static_cast<AreaScalar>(
                                        current_dimension + 1));
                            const IntegralScalar interface_factor =
                                static_cast<IntegralScalar>(
                                    distance /
                                    static_cast<AreaScalar>(
                                        current_dimension + 1));

                            for (std::size_t component = 0;
                                 component < components_;
                                 ++component) {
                                bulk[component] +=
                                    function_value_[component] *
                                        center_factor +
                                    interface_block_ptr[component] *
                                        interface_factor;
                            }
                        }
                    }
                } else {
                    // Exact Julia Full_Matrix branch: the lower-index update
                    // cell has already calculated this shared interface.
                    buffer.clear();

                    nodes_->copy_node(
                        neighbour,
                        node_point_.data());
                    const AreaScalar distance =
                        static_cast<AreaScalar>(
                            typename Point::Scalar{0.5} *
                            (cell_point_ - node_point_).norm());

                    const std::size_t opposite =
                        reciprocal_ordinal_from_update(
                            pass,
                            position,
                            ordinal,
                            neighbour);

                    const auto& other =
                        pass.cell(
                            static_cast<std::size_t>(
                                neighbour));
                    const auto& other_data =
                        other.data();

                    const AreaScalar shared_area =
                        other_data.area().at(opposite);
                    const AreaScalar cone_volume =
                        shared_area * distance /
                        static_cast<AreaScalar>(
                            current_dimension);

                    volume += cone_volume;
                    area[ordinal] = shared_area;

                    if (has_function()) {
                        const auto& source =
                            other_data.interface_integral();
                        copy_interface_block(
                            source,
                            opposite,
                            interface_block_ptr);

                        if (!bulk.empty()) {
                            evaluate(
                                cell_point_,
                                function_value_);

                            const IntegralScalar center_factor =
                                static_cast<IntegralScalar>(
                                    cone_volume /
                                    static_cast<AreaScalar>(
                                        current_dimension + 1));
                            const IntegralScalar interface_factor =
                                static_cast<IntegralScalar>(
                                    distance /
                                    static_cast<AreaScalar>(
                                        current_dimension + 1));

                            for (std::size_t component = 0;
                                 component < components_;
                                 ++component) {
                                bulk[component] +=
                                    function_value_[component] *
                                        center_factor +
                                    interface_block_ptr[component] *
                                        interface_factor;
                            }
                        }
                    }
                }

                neighbours_[ordinal] =
                    neighbour;
            }
            return;
        }

        // ---------------------------------------------------------------
        // Julia 1 < dim < space_dim recursion.
        // ---------------------------------------------------------------
        const Point center_relative =
            midpoint(
                vertices,
                vertices2,
                reference_center);
        minors_.update(
            dimension_ - current_dimension,
            center_relative);

        std::vector<EdgeMap>& dictionaries =
            lower_lists_.at(
                current_dimension - 2);

        std::vector<Index>& active_neighbours =
            active_neighbours_levels_.at(
                current_dimension - 2);
        active_neighbours.clear();
        for (const Index neighbour : neighbours_) {
            if (neighbour != invalid_index()) {
                active_neighbours.push_back(
                    neighbour);
            }
        }

        resize_and_clear_maps(
            dictionaries,
            active_neighbours.size());

        // Julia consumes the parent dictionary with pop!(). The parent
        // occurrence record itself is immutable, so children may reference the
        // same database address until a later operation explicitly creates a
        // different PolyEdge state.
        while (!vertices.empty()) {
            const EdgeAddress address =
                vertices.pop_address();
            edge_database_.read(
                address,
                edge_signature_buffer_,
                edge_state_.r1,
                edge_state_.r2,
                edge_state_.payload);

            if ((edge_state_.r1 - edge_state_.r2).squaredNorm() ==
                typename Point::Scalar{0}) {
                continue;
            }

            for (const Index candidate : edge_signature_buffer_) {
                if (contains_taboo(candidate)) {
                    continue;
                }

                const std::size_t index =
                    neighbour_index(
                        active_neighbours,
                        candidate);
                if (index == npos) {
                    continue;
                }

                // The supplied Julia `count == dim` condition never mutates
                // count and is therefore inactive; it is intentionally not
                // given new semantics here.
                dictionaries[index].insert_or_assign(
                    edge_signature_buffer_,
                    address);
            }
        }

        if (current_dimension == std::size_t{2}) {
            merge_linear_dictionaries(
                dictionaries,
                active_neighbours.size());
        }

        std::size_t active_index = 0;
        for (std::size_t ordinal = 0;
             ordinal < neighbours_.size();
             ++ordinal) {
            const Index neighbour =
                neighbours_[ordinal];
            if (neighbour == invalid_index()) {
                continue;
            }

            if (active_index >=
                active_neighbours.size()) {
                throw std::logic_error(
                    "Polygon recursive neighbour/dictionary mismatch");
            }

            EdgeMap& buffer =
                dictionaries[active_index++];

            const bool valid =
                checker_.set_dimension(
                    dimension_ - current_dimension + 1,
                    cell,
                    neighbour);
            if (!valid) {
                buffer.clear();
            }

            if (buffer.empty()) {
                continue;
            }

            if (!std::isfinite(
                    static_cast<double>(area[0]))) {
                buffer.clear();
                for (auto& dictionary : dictionaries) {
                    dictionary.clear();
                }
                return;
            }

            neighbours_[ordinal] =
                invalid_index();
            taboo_[current_dimension - 2] =
                neighbour;

            iterative_volume(
                pass,
                position,
                current_dimension - 1,
                buffer,
                empty_map_,
                reference_center,
                volume,
                bulk,
                area,
                interface_values);

            neighbours_[ordinal] =
                neighbour;
            taboo_[current_dimension - 2] =
                invalid_index();

            // Julia: if !isempty(bufferlist) pop!(bufferlist) end
            if (!buffer.empty()) {
                (void)buffer.pop_address();
            }
        }
    }

    void merge_linear_dictionaries(
        std::vector<EdgeMap>& dictionaries,
        std::size_t count) {
        if (count < 2) {
            return;
        }

        for (std::size_t left = 0;
             left + 1 < count;
             ++left) {
            if (dictionaries[left].empty()) {
                continue;
            }

            for (std::size_t right = left + 1;
                 right < count;
                 ++right) {
                bool common_key = false;
                for (const auto& entry :
                     dictionaries[left].entries()) {
                    edge_database_.read_signature(
                        entry.address,
                        edge_signature_buffer_);
                    if (dictionaries[right].contains(
                            edge_signature_buffer_)) {
                        common_key = true;
                        break;
                    }
                }

                if (!common_key) {
                    continue;
                }

                // Julia merge!(dd[k], dd[i]): right-hand values replace
                // duplicate left-hand values. The occurrence address carries
                // both endpoints and the shared payload address.
                for (const auto& entry :
                     dictionaries[right].entries()) {
                    edge_database_.read_signature(
                        entry.address,
                        edge_signature_buffer_);
                    dictionaries[left].insert_or_assign(
                        edge_signature_buffer_,
                        entry.address);
                }
                dictionaries[right].clear();
            }
        }
    }

    [[nodiscard]] bool contains_taboo(
        Index value) const {
        return std::find(
                   taboo_.begin(),
                   taboo_.end(),
                   value) != taboo_.end();
    }

    [[nodiscard]] std::size_t
    reciprocal_ordinal_from_update(
        const Pass& pass,
        std::size_t current_position,
        std::size_t current_ordinal,
        Index other) const {
        if (static_cast<std::size_t>(other) >=
            current_position) {
            throw std::logic_error(
                "Polygon opposite update cell has not been calculated yet");
        }

        const auto& current_neighbours =
            pass.cell(current_position).neighbours();
        const auto& other_neighbours =
            pass.cell(
                static_cast<std::size_t>(
                    other)).neighbours();

        return reciprocal_ordinal(
            current_neighbours,
            current_ordinal,
            static_cast<Index>(
                current_position),
            other_neighbours);
    }

    /**
     * Match duplicate neighbour occurrences by occurrence rank rather than
     * collapsing them. Persistent neighbour multiplicity remains semantic.
     */
    [[nodiscard]] static std::size_t reciprocal_ordinal(
        const std::vector<Index>& current_neighbours,
        std::size_t current_ordinal,
        Index current_cell,
        const std::vector<Index>& other_neighbours) {
        if (current_ordinal >=
            current_neighbours.size()) {
            throw std::out_of_range(
                "Polygon reciprocal ordinal source out of range");
        }

        const Index other =
            current_neighbours[current_ordinal];
        std::size_t occurrence = 0;
        for (std::size_t i = 0;
             i <= current_ordinal;
             ++i) {
            if (current_neighbours[i] == other) {
                ++occurrence;
            }
        }

        for (std::size_t i = 0;
             i < other_neighbours.size();
             ++i) {
            if (other_neighbours[i] != current_cell) {
                continue;
            }
            if (--occurrence == 0) {
                return i;
            }
        }

        throw std::logic_error(
            "Polygon shared interface has no reciprocal neighbour occurrence");
    }

    [[nodiscard]] IntegralScalar* interface_block(
        std::vector<IntegralScalar>& values,
        std::size_t ordinal) const {
        if (!has_function()) {
            return nullptr;
        }
        const std::size_t begin =
            ordinal * components_;
        if (begin + components_ > values.size()) {
            throw std::out_of_range(
                "Polygon interface block outside storage");
        }
        return values.data() + begin;
    }

    void copy_interface_block(
        const std::vector<IntegralScalar>& source,
        std::size_t ordinal,
        IntegralScalar* target) const {
        const std::size_t begin =
            ordinal * components_;
        if (begin + components_ >
            source.size()) {
            throw std::out_of_range(
                "Polygon source interface block outside storage");
        }
        std::copy_n(
            source.data() + begin,
            components_,
            target);
    }

    void prepare_map(EdgeMap& map) {
        map.bind(edge_database_);
    }

    void clear_maps(
        std::vector<EdgeMap>& maps) {
        for (auto& map : maps) {
            prepare_map(map);
            map.clear();
        }
    }

    void resize_and_clear_maps(
        std::vector<EdgeMap>& maps,
        std::size_t size) {
        while (maps.size() < size) {
            maps.emplace_back(edge_database_);
        }
        for (auto& map : maps) {
            prepare_map(map);
            map.clear();
        }
    }

    static constexpr std::size_t npos =
        (std::numeric_limits<std::size_t>::max)();

    View* view_ = nullptr;
    Nodes* nodes_ = nullptr;
    std::size_t dimension_ = 0;
    std::size_t components_ = 0;
    const Function* function_ = nullptr;

    EdgeDatabase edge_database_;
    PayloadDatabase payload_database_;
    detail::IterativeDimensionChecker<Nodes> checker_;
    IncrementalMinors<AreaScalar> minors_;

    std::vector<Index> neighbours_;
    std::vector<std::vector<Index>> active_neighbours_levels_;
    std::vector<Index> edge_buffer_;
    std::vector<Index> edge_signature_buffer_;
    std::vector<Index> taboo_;

    std::vector<EdgeMap> top_lists_;
    std::vector<std::vector<EdgeMap>> lower_lists_;
    EdgeMap empty_map_;
    EdgeMap top_dummy_;

    EdgeState edge_state_;
    Point vertex_position_buffer_;
    Point empty_vector_;
    Point cell_point_;
    Point node_point_;
    Point evaluation_point_;

    std::vector<IntegralScalar> function_value_;
    std::vector<IntegralScalar> second_function_value_;
    std::vector<IntegralScalar> payload_value_;
    typename View::CellData clean_cell_scratch_;
};

} // namespace detail

/**
 * @brief C++ port of HighVoronoi.jl Polygon_Integrator.
 *
 * IntegralT is part of the type so the algorithm can own a persistent,
 * allocation-reusing workspace for the concrete IntegrationPass. This avoids
 * weakening Julia's one-sided interface ownership merely to fit a type-erased
 * algorithm callback.
 */
template <class IntegralT, class FunctionT = NoPolygonIntegrand>
class PolygonAlgorithm final {
public:
    using Integral = IntegralT;
    using Function = FunctionT;
    using View = decltype(
        make_integration_view(
            std::declval<Integral&>()));
    using Pass = IntegrationPass<View>;
    using Workspace =
        detail::PolygonWorkspace<Pass, Function>;

    template <
        class F = Function,
        std::enable_if_t<
            std::is_same_v<F, NoPolygonIntegrand>,
            int> = 0>
    PolygonAlgorithm()
        : function_(NoPolygonIntegrand{}) {}

    explicit PolygonAlgorithm(Function function)
        : function_(std::move(function)) {}

    void begin_pass(Pass& pass) {
        auto& nodes =
            pass.view().mesh().extended_nodes();
        workspace_ =
            std::make_unique<Workspace>(
                pass.view(),
                nodes,
                pass.view()
                    .integral()
                    .data()
                    .integral_components(),
                function_);
    }

    void integrate_cell(
        Pass& pass,
        std::size_t position) {
        require_workspace();
        workspace_->integrate_cell(
            pass,
            position);
    }

    void cleanup_cell(
        Pass& pass,
        std::size_t position) {
        require_workspace();
        workspace_->cleanup_cell(
            pass,
            position);
    }

    void end_pass(Pass&) noexcept {
        workspace_.reset();
    }

    /**
     * Independent algorithm state for one parallel worker view.
     * The function object stays owned by the parent; only the mutable recursive
     * PolygonWorkspace is duplicated per worker.
     */
    class ParallelWorker final {
    public:
        explicit ParallelWorker(const PolygonAlgorithm& parent)
            : parent_(&parent) {}

        void begin_pass(Pass& pass) {
            auto& nodes = pass.view().mesh().extended_nodes();
            workspace_ = std::make_unique<Workspace>(
                pass.view(),
                nodes,
                pass.view().integral().data().integral_components(),
                parent_->function_);
        }

        void integrate_cell(Pass& pass, std::size_t position) {
            require_workspace();
            workspace_->integrate_cell(pass, position);
        }

        void end_pass(Pass&) noexcept {
            workspace_.reset();
        }

    private:
        void require_workspace() const {
            if (!workspace_) {
                throw std::logic_error(
                    "Polygon parallel worker begin_pass was not called");
            }
        }

        const PolygonAlgorithm* parent_ = nullptr;
        std::unique_ptr<Workspace> workspace_;
    };

    [[nodiscard]] ParallelWorker make_worker_algorithm(std::size_t) const {
        return ParallelWorker(*this);
    }

private:
    void require_workspace() const {
        if (!workspace_) {
            throw std::logic_error(
                "PolygonAlgorithm begin_pass was not called");
        }
    }

    Function function_;
    std::unique_ptr<Workspace> workspace_;
};

template <class Integral>
[[nodiscard]] PolygonAlgorithm<Integral>
make_polygon_algorithm(Integral&) {
    return PolygonAlgorithm<Integral>{};
}

template <class Integral, class Function>
[[nodiscard]] auto make_polygon_algorithm(
    Integral&,
    Function&& function) {
    using StoredFunction =
        std::decay_t<Function>;
    return PolygonAlgorithm<
        Integral,
        StoredFunction>(
            std::forward<Function>(function));
}

} // namespace highvoronoi
