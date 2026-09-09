
#pragma once

/**
 * @file fast_polygon_integrator.hpp
 * @brief Julia-faithful serial port of Fast_Polygon_Integrator.
 *
 * FastPolygonIntegrator uses the same cone-recursion formula as the Julia
 * implementation, but replaces the ordinary PolygonIntegrator's determinant
 * recursion by direct projected heights and memoizes lower-dimensional facet
 * results in a recursive hierarchy.
 *
 * The recursive kernel is parameterized by a facet-store contract. The serial
 * implementation uses SerialFastPolygonFacetStore. A later parallel layer can
 * provide the same load/store contract with Julia's KeyDict/MultiKeyDict-style
 * shared sharding without changing the geometric/integration recursion.
 */

#include <highvoronoi/integration/detail/fast_polygon_cache.hpp>
#include <highvoronoi/integration/detail/iterative_dimension_checker.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>

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

namespace highvoronoi::detail {

/**
 * Reusable serial FastPolygon recursion workspace.
 *
 * `FacetStore` is intentionally abstract at the class boundary. It must expose:
 *
 *   void begin_pass(level_count, integral_components)
 *   bool load(level, key, volume, integral)
 *   void store(level, key, volume, integral)
 *   const FastPolygonCacheStats& stats() const
 *
 * This is the C++ counterpart of Julia's DictHierarchy/ThreadsafeDictHierarchy
 * split. Geometry and recursion are independent of serial/parallel storage.
 */
template <class Pass, class Function, class FacetStore>
class FastPolygonWorkspace final {
public:
    using View = typename Pass::View;
    using Update = typename Pass::Update;
    using Index = typename View::Index;
    using AreaScalar = typename View::AreaScalar;
    using IntegralScalar = typename View::IntegralScalar;
    using Nodes = std::remove_reference_t<
        decltype(std::declval<Update&>().extended_nodes())>;
    using Checker = IterativeDimensionChecker<Nodes>;
    using Point = typename Checker::Point;
    using Key = typename FacetStore::Key;

private:
    struct VertexData {
        std::vector<Index> sigma;
        Point position;

        explicit VertexData(std::size_t dimension)
            : position(make_point(dimension)) {}
    };

    class ActiveVertexRange final {
    public:
        ActiveVertexRange(
            const std::vector<VertexData>& data,
            std::size_t count)
            : data_(&data), count_(count) {}

        [[nodiscard]] auto begin() const { return data_->begin(); }
        [[nodiscard]] auto end() const {
            return data_->begin() + static_cast<std::ptrdiff_t>(count_);
        }

    private:
        const std::vector<VertexData>* data_ = nullptr;
        std::size_t count_ = 0;
    };

    using VertexList = std::vector<std::size_t>;
    using ListLevel = std::vector<VertexList>;

public:
    FastPolygonWorkspace(
        View& view,
        Nodes& nodes,
        std::size_t integral_components,
        const Function& function,
        FacetStore& facet_store)
        : view_(&view),
          nodes_(&nodes),
          dimension_(static_cast<std::size_t>(nodes.dimension())),
          components_(integral_components),
          function_(&function),
          facet_store_(&facet_store),
          checker_(nodes, dimension_),
          cell_point_(make_point(dimension_)),
          node_point_(make_point(dimension_)),
          evaluation_point_(make_point(dimension_)),
          distance_buffer_(make_point(dimension_)),
          edge_function_first_(integral_components),
          edge_function_second_(integral_components),
          clean_cell_scratch_() {
        if (dimension_ < std::size_t{2}) {
            throw std::invalid_argument(
                "FastPolygonAlgorithm requires dimension >= 2");
        }

        lower_lists_.resize(dimension_ + 1);
        active_neighbours_.resize(dimension_ + 1);
        centers_.reserve(dimension_ + 1);
        for (std::size_t i = 0; i <= dimension_; ++i) {
            centers_.push_back(make_point(dimension_));
        }

        recursive_integrals_.resize(dimension_ + 1);
        base_values_.resize(dimension_ + 1);
        for (auto& values : recursive_integrals_) {
            values.resize(components_);
        }
        for (auto& values : base_values_) {
            values.resize(components_);
        }

        key_scratch_.resize(
            dimension_ > 2 ? dimension_ - 2 : std::size_t{0});
        common_generators_.resize(dimension_ + 1);
        taboo_.assign(dimension_, invalid_index());
    }

    void integrate_cell(Pass& pass, std::size_t position) {
        Update& update = pass.writable_cell(position);
        reset_cell(update);
        require_layout(pass, update);

        auto& data = update.data();
        auto& areas = data.area();
        auto& bulk = data.bulk_integral();
        auto& interface_integrals = data.interface_integral();

        std::fill(bulk.begin(), bulk.end(), IntegralScalar{0});

        const Index cell = static_cast<Index>(position);
        taboo_[dimension_ - 1] = cell;

        AreaScalar volume = AreaScalar{0};
        std::vector<IntegralScalar>& total_integral =
            recursive_integrals_[dimension_];
        std::fill(
            total_integral.begin(),
            total_integral.end(),
            IntegralScalar{0});

        if (has_function()) {
            evaluate(cell_point_, base_values_[dimension_]);
        }

        for (std::size_t ordinal = 0;
             ordinal < neighbours_.size();
             ++ordinal) {
            const Index neighbour = neighbours_[ordinal];
            VertexList& buffer = top_lists_[ordinal];

            if (!is_calculate_neighbour(pass, neighbour)) {
                buffer.clear();
                continue;
            }

            if (neighbour > cell && buffer.empty()) {
                continue;
            }

            taboo_[dimension_ - 2] = neighbour;
            neighbours_[ordinal] = invalid_index();

            AreaScalar facet_volume = AreaScalar{0};
            std::vector<IntegralScalar>& facet_integral =
                recursive_integrals_[dimension_ - 1];
            std::fill(
                facet_integral.begin(),
                facet_integral.end(),
                IntegralScalar{0});

            if (neighbour > cell) {
                // Julia ignores the return value at the first dimension.
                (void)checker_.set_dimension(1, cell, neighbour);

                midpoint_vertices(
                    buffer,
                    centers_[dimension_ - 1]);

                recursive_volume(
                    pass,
                    position,
                    dimension_ - 1,
                    buffer,
                    facet_volume,
                    facet_integral);

                buffer.clear();
                areas[ordinal] = facet_volume;

                if (has_function()) {
                    IntegralScalar* target =
                        interface_block(interface_integrals, ordinal);
                    std::copy(
                        facet_integral.begin(),
                        facet_integral.end(),
                        target);
                }
            } else {
                buffer.clear();
                const std::size_t opposite =
                    reciprocal_ordinal_from_update(
                        pass,
                        position,
                        ordinal,
                        neighbour);
                const auto& other_data =
                    pass.cell(static_cast<std::size_t>(neighbour)).data();
                facet_volume = other_data.area().at(opposite);
                areas[ordinal] = facet_volume;

                if (has_function()) {
                    copy_interface_block(
                        other_data.interface_integral(),
                        opposite,
                        facet_integral.data());
                    std::copy(
                        facet_integral.begin(),
                        facet_integral.end(),
                        interface_block(interface_integrals, ordinal));
                }
            }

            nodes_->copy_node(neighbour, node_point_.data());
            const AreaScalar distance = static_cast<AreaScalar>(
                typename Point::Scalar{0.5} *
                (cell_point_ - node_point_).norm());

            volume += facet_volume * distance;
            if (has_function()) {
                for (std::size_t component = 0;
                     component < components_;
                     ++component) {
                    total_integral[component] +=
                        facet_integral[component] *
                        static_cast<IntegralScalar>(distance);
                }
            }

            neighbours_[ordinal] = neighbour;
            taboo_[dimension_ - 2] = invalid_index();
        }

        volume /= static_cast<AreaScalar>(dimension_);
        data.set_volume(volume);

        if (has_function() && !bulk.empty()) {
            const IntegralScalar divisor =
                static_cast<IntegralScalar>(dimension_ + 1);
            for (std::size_t component = 0;
                 component < components_;
                 ++component) {
                bulk[component] =
                    (total_integral[component] +
                     static_cast<IntegralScalar>(volume) *
                         base_values_[dimension_][component]) /
                    divisor;
            }
        }
    }

    /** Same Julia cleanup_cell branch shared by Polygon/FastPolygon. */
    void cleanup_cell(Pass& pass, std::size_t position) {
        Update& update = pass.writable_cell(position);
        auto& data = update.data();
        auto& areas = data.area();
        auto& bulk = data.bulk_integral();
        auto& interface_integrals = data.interface_integral();
        const auto& neighbours = update.neighbours();

        nodes_->copy_node(update.cell(), cell_point_.data());

        for (std::size_t ordinal = 0;
             ordinal < neighbours.size();
             ++ordinal) {
            const Index neighbour = neighbours[ordinal];
            if (is_calculate_neighbour(pass, neighbour)) {
                continue;
            }
            if (!is_clean_integral_neighbour(pass, neighbour)) {
                throw std::logic_error(
                    "FastPolygon cleanup encountered unsupported geometry-only neighbour");
            }

            nodes_->copy_node(neighbour, node_point_.data());
            const AreaScalar distance = static_cast<AreaScalar>(
                typename Point::Scalar{0.5} *
                (cell_point_ - node_point_).norm());

            const bool has_area_data =
                areas.at(ordinal) > AreaScalar{0};
            AreaScalar area = areas.at(ordinal);
            std::size_t opposite = 0;

            if (!has_area_data) {
                if (!update.read_integral_cell(
                        neighbour,
                        clean_cell_scratch_)) {
                    throw std::logic_error(
                        "FastPolygon cleanup could not read CLEAN opposite cell");
                }
                opposite = reciprocal_ordinal(
                    neighbours,
                    ordinal,
                    static_cast<Index>(position),
                    clean_cell_scratch_.neighbours());
                area = clean_cell_scratch_.area().at(opposite);
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
                        opposite,
                        target);
                }

                if (!bulk.empty()) {
                    // Deliberately Julia-faithful cleanup factor: distance/dim.
                    const IntegralScalar factor =
                        static_cast<IntegralScalar>(
                            distance /
                            static_cast<AreaScalar>(dimension_));
                    for (std::size_t component = 0;
                         component < components_;
                         ++component) {
                        bulk[component] += target[component] * factor;
                    }
                }
            }
        }
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

    [[nodiscard]] bool has_function() const noexcept {
        return !std::is_same_v<Function, NoPolygonIntegrand>;
    }

    void evaluate(
        const Point& point,
        std::vector<IntegralScalar>& output) {
        if constexpr (std::is_same_v<Function, NoPolygonIntegrand>) {
            if (!output.empty()) {
                throw std::logic_error(
                    "FastPolygon attempted function evaluation without integrand");
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

    void ensure_vertex(std::size_t index) {
        if (index >= vertices_.size()) {
            vertices_.emplace_back(dimension_);
        }
    }

    static void ensure_lists(ListLevel& lists, std::size_t count) {
        if (lists.size() < count) {
            lists.resize(count);
        }
        for (auto& list : lists) {
            list.clear();
        }
    }

    void reset_cell(Update& update) {
        neighbours_.assign(
            update.neighbours().begin(),
            update.neighbours().end());
        ensure_lists(top_lists_, neighbours_.size());

        active_vertex_count_ = 0;
        for (const auto& record : update.vertices()) {
            ensure_vertex(active_vertex_count_);
            VertexData& vertex = vertices_[active_vertex_count_];
            vertex.sigma.assign(record.sigma.begin(), record.sigma.end());
            for (std::size_t coordinate = 0;
                 coordinate < dimension_;
                 ++coordinate) {
                vertex.position[static_cast<Eigen::Index>(coordinate)] =
                    static_cast<typename Point::Scalar>(
                        record.position[static_cast<Eigen::Index>(coordinate)]);
            }

            const std::size_t vertex_id = active_vertex_count_++;
            for (const Index generator : vertex.sigma) {
                if (generator == update.cell()) {
                    continue;
                }
                const std::size_t ordinal =
                    neighbour_index(neighbours_, generator);
                if (ordinal != npos) {
                    top_lists_[ordinal].push_back(vertex_id);
                }
            }
        }

        for (auto& level : lower_lists_) {
            for (auto& list : level) {
                list.clear();
            }
        }
        for (auto& active : active_neighbours_) {
            active.clear();
        }
        std::fill(taboo_.begin(), taboo_.end(), invalid_index());

        nodes_->copy_node(update.cell(), cell_point_.data());

        // FastPolygon needs a real local basis even in general position because
        // projected_distance() uses it. Julia therefore calls reset(..., false).
        (void)checker_.reset(
            neighbours_,
            update.cell(),
            ActiveVertexRange(vertices_, active_vertex_count_),
            false);
    }

    template <class PassLike>
    void require_layout(
        const PassLike& pass,
        const Update& update) const {
        const auto& data = update.data();
        const auto& integral_data = pass.view().integral().data();

        if (!integral_data.stores_volume() ||
            !integral_data.stores_area()) {
            throw std::logic_error(
                "FastPolygonAlgorithm requires volume and area storage");
        }
        if (data.area().size() != update.neighbours().size()) {
            throw std::logic_error(
                "FastPolygon area storage is not neighbour-aligned");
        }
        if (integral_data.stores_bulk_integral() &&
            !integral_data.stores_interface_integral()) {
            throw std::logic_error(
                "FastPolygon bulk integration requires interface-integral storage");
        }

        if (has_function()) {
            if (components_ == 0) {
                throw std::logic_error(
                    "FastPolygon integrand supplied with zero integral components");
            }
            if (!integral_data.stores_interface_integral()) {
                throw std::logic_error(
                    "FastPolygon integrand requires interface-integral storage");
            }
            if (data.interface_integral().size() !=
                update.neighbours().size() * components_) {
                throw std::logic_error(
                    "FastPolygon interface-integral layout mismatch");
            }
            if (integral_data.stores_bulk_integral() &&
                data.bulk_integral().size() != components_) {
                throw std::logic_error(
                    "FastPolygon bulk-integral layout mismatch");
            }
        } else if (integral_data.stores_bulk_integral() ||
                   integral_data.stores_interface_integral()) {
            throw std::logic_error(
                "FastPolygon integral storage enabled without integrand");
        }
    }

    [[nodiscard]] std::size_t neighbour_index(
        const std::vector<Index>& values,
        Index value) const noexcept {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i] == value) {
                return i;
            }
        }
        return npos;
    }

    [[nodiscard]] bool contains_taboo(Index value) const noexcept {
        return std::find(
                   taboo_.begin(),
                   taboo_.end(),
                   value) != taboo_.end();
    }

    [[nodiscard]] bool is_calculate_neighbour(
        const Pass& pass,
        Index neighbour) const {
        const std::size_t value =
            static_cast<std::size_t>(neighbour);

        // A parallel worker owns only its local prefix, but its reordered view
        // still knows the complete global NEW/DIRTY set.  Cross-worker update
        // cells must therefore be treated as CALCULATE rather than CLEAN.
        // Because the worker's own prefix is placed first, such neighbours lie
        // after the active cell and are computed geometrically on this worker
        // instead of trying to read another worker's in-flight transaction.
        if (polygon_is_update_cell(
                pass.view(),
                neighbour,
                pass.size())) {
            return true;
        }

        if (polygon_is_reference(pass.view(), neighbour)) {
            return true;
        }
        if (value >= static_cast<std::size_t>(pass.view().size())) {
            return true;
        }
        return false;
    }

    [[nodiscard]] bool is_clean_integral_neighbour(
        const Pass& pass,
        Index neighbour) const {
        const std::size_t value =
            static_cast<std::size_t>(neighbour);
        return value < static_cast<std::size_t>(pass.view().size()) &&
               !polygon_is_update_cell(
                   pass.view(),
                   neighbour,
                   pass.size()) &&
               !polygon_is_reference(pass.view(), neighbour);
    }

    void midpoint_vertices(
        const VertexList& vertices,
        Point& output) const {
        if (vertices.empty()) {
            throw std::logic_error(
                "FastPolygon midpoint requested for empty vertex list");
        }
        output.setZero();
        for (const std::size_t vertex_id : vertices) {
            if (vertex_id >= active_vertex_count_) {
                throw std::out_of_range(
                    "FastPolygon vertex id outside active cell data");
            }
            output += vertices_[vertex_id].position;
        }
        output /= static_cast<typename Point::Scalar>(vertices.size());
    }

    void recursive_volume(
        Pass& pass,
        std::size_t position,
        std::size_t current_dimension,
        VertexList& vertex_ids,
        AreaScalar& output_volume,
        std::vector<IntegralScalar>& output_integral) {
        const Index cell = static_cast<Index>(position);
        output_volume = AreaScalar{0};
        std::fill(
            output_integral.begin(),
            output_integral.end(),
            IntegralScalar{0});

        if (current_dimension == std::size_t{1}) {
            if (vertex_ids.size() < std::size_t{2}) {
                vertex_ids.clear();
                return;
            }

            const Point& first = vertices_[vertex_ids[0]].position;
            const Point& second = vertices_[vertex_ids[1]].position;
            output_volume = static_cast<AreaScalar>((first - second).norm());

            if (has_function()) {
                evaluate(first, edge_function_first_);
                evaluate(second, edge_function_second_);
                const IntegralScalar factor =
                    static_cast<IntegralScalar>(
                        AreaScalar{0.5} * output_volume);
                for (std::size_t component = 0;
                     component < components_;
                     ++component) {
                    output_integral[component] =
                        (edge_function_first_[component] +
                         edge_function_second_[component]) *
                        factor;
                }
            }
            return;
        }

        Point& center = centers_[current_dimension];
        // The parent already wrote the absolute midpoint into this level's
        // center, exactly matching Julia `buffer_data.center .= empty_vector`.

        std::vector<Index>& active = active_neighbours_[current_dimension];
        active.clear();
        for (const Index neighbour : neighbours_) {
            if (neighbour != invalid_index()) {
                active.push_back(neighbour);
            }
        }

        ListLevel& lists = lower_lists_[current_dimension];
        ensure_lists(lists, active.size());

        for (const std::size_t vertex_id : vertex_ids) {
            const auto& sigma = vertices_[vertex_id].sigma;
            for (const Index generator : sigma) {
                if (contains_taboo(generator)) {
                    continue;
                }
                const std::size_t local = neighbour_index(active, generator);
                if (local != npos) {
                    lists[local].push_back(vertex_id);
                }
            }
        }

        if (active.size() > 1) {
            for (std::size_t k = 0; k + 1 < active.size(); ++k) {
                if (current_dimension == std::size_t{2}) {
                    clear_double_lists_bottom(lists, k, active.size());
                } else {
                    clear_double_lists_general(lists, k, active, current_dimension);
                }
            }
        }

        if (has_function()) {
            evaluate(center, base_values_[current_dimension]);
        }

        AreaScalar accumulated_volume = AreaScalar{0};
        std::vector<IntegralScalar>& accumulated_integral =
            output_integral;
        const std::size_t next_ortho_dimension =
            dimension_ - current_dimension + 1;

        std::size_t active_index = 0;
        for (std::size_t ordinal = 0;
             ordinal < neighbours_.size();
             ++ordinal) {
            const Index neighbour = neighbours_[ordinal];
            if (neighbour == invalid_index()) {
                continue;
            }
            if (active_index >= active.size()) {
                throw std::logic_error(
                    "FastPolygon recursive neighbour/list mismatch");
            }

            VertexList& buffer = lists[active_index++];
            if (buffer.empty()) {
                continue;
            }

            const bool valid = checker_.set_dimension(
                next_ortho_dimension,
                cell,
                neighbour);
            if (!valid) {
                buffer.clear();
                continue;
            }

            const std::size_t cache_level =
                dimension_ - current_dimension - 1;
            Key& key = key_scratch_.at(cache_level);
            build_facet_identifier(
                pass,
                cell,
                neighbour,
                current_dimension,
                key);

            const AreaScalar distance = projected_distance(
                center,
                vertices_[buffer.front()].position,
                next_ortho_dimension);

            AreaScalar child_volume = AreaScalar{0};
            std::vector<IntegralScalar>& child_integral =
                recursive_integrals_[current_dimension - 1];
            std::fill(
                child_integral.begin(),
                child_integral.end(),
                IntegralScalar{0});

            if (!facet_store_->load(
                    cache_level,
                    key,
                    child_volume,
                    child_integral)) {
                midpoint_vertices(
                    buffer,
                    centers_[current_dimension - 1]);

                neighbours_[ordinal] = invalid_index();
                taboo_[current_dimension - 2] = neighbour;

                recursive_volume(
                    pass,
                    position,
                    current_dimension - 1,
                    buffer,
                    child_volume,
                    child_integral);

                neighbours_[ordinal] = neighbour;
                taboo_[current_dimension - 2] = invalid_index();

                facet_store_->store(
                    cache_level,
                    key,
                    child_volume,
                    child_integral);
            }

            buffer.clear();
            accumulated_volume += child_volume * distance;

            if (has_function() && child_volume != AreaScalar{0}) {
                for (std::size_t component = 0;
                     component < components_;
                     ++component) {
                    accumulated_integral[component] +=
                        child_integral[component] *
                        static_cast<IntegralScalar>(distance);
                }
            }
        }

        output_volume =
            accumulated_volume /
            static_cast<AreaScalar>(current_dimension);

        if (has_function()) {
            const IntegralScalar divisor =
                static_cast<IntegralScalar>(current_dimension + 1);
            for (std::size_t component = 0;
                 component < components_;
                 ++component) {
                output_integral[component] =
                    (output_integral[component] +
                     static_cast<IntegralScalar>(output_volume) *
                         base_values_[current_dimension][component]) /
                    divisor;
            }
        }
    }

    [[nodiscard]] AreaScalar projected_distance(
        const Point& center,
        const Point& plane_point,
        std::size_t basis_count) {
        distance_buffer_ = center - plane_point;
        AreaScalar squared = AreaScalar{0};
        const auto& basis = checker_.local_basis();
        if (basis_count > basis.size()) {
            throw std::out_of_range(
                "FastPolygon projected distance basis outside dimension");
        }
        for (std::size_t k = 0; k < basis_count; ++k) {
            const auto projection = basis[k].dot(distance_buffer_);
            squared += static_cast<AreaScalar>(projection * projection);
        }
        using std::sqrt;
        return sqrt(squared);
    }

    void clear_double_lists_bottom(
        ListLevel& lists,
        std::size_t current,
        std::size_t active_count) const {
        if (lists[current].size() != std::size_t{2}) {
            lists[current].clear();
            return;
        }

        for (std::size_t other = current + 1;
             other < active_count;
             ++other) {
            if (lists[other].size() != std::size_t{2}) {
                continue;
            }
            std::size_t common = 0;
            for (const std::size_t value : lists[current]) {
                if (std::find(
                        lists[other].begin(),
                        lists[other].end(),
                        value) != lists[other].end()) {
                    ++common;
                }
            }
            if (common == std::size_t{2}) {
                lists[other].clear();
            }
        }
    }

    void clear_double_lists_general(
        ListLevel& lists,
        std::size_t current,
        const std::vector<Index>& active,
        std::size_t current_dimension) {
        if (lists[current].empty()) {
            return;
        }

        std::vector<Index>& common =
            common_generators_.at(current_dimension);
        common = vertices_[lists[current].front()].sigma;

        for (const std::size_t vertex_id : lists[current]) {
            const auto& sigma = vertices_[vertex_id].sigma;
            common.erase(
                std::remove_if(
                    common.begin(),
                    common.end(),
                    [&](Index generator) {
                        return !std::binary_search(
                            sigma.begin(),
                            sigma.end(),
                            generator);
                    }),
                common.end());
        }

        for (std::size_t other = current + 1;
             other < active.size();
             ++other) {
            if (std::find(
                    common.begin(),
                    common.end(),
                    active[other]) == common.end()) {
                continue;
            }

            bool subset = true;
            for (const std::size_t vertex_id : lists[other]) {
                if (std::find(
                        lists[current].begin(),
                        lists[current].end(),
                        vertex_id) == lists[current].end()) {
                    subset = false;
                    break;
                }
            }
            if (subset) {
                lists[other].clear();
            }
        }
    }

    void build_facet_identifier(
        const Pass& pass,
        Index cell,
        Index facet,
        std::size_t current_dimension,
        Key& key) const {
        const std::size_t path_count =
            dimension_ - current_dimension;
        key.resize(path_count + 2);

        const auto& current_path = checker_.current_path();
        for (std::size_t k = 0; k < path_count; ++k) {
            key[k] = canonical_index(pass, current_path[k]);
        }
        key[path_count] = canonical_index(pass, facet);
        key[path_count + 1] = canonical_index(pass, cell);
        std::sort(key.begin(), key.end());
    }

    [[nodiscard]] Index canonical_index(
        const Pass& pass,
        Index view_index) const {
        const std::size_t value = static_cast<std::size_t>(view_index);
        const std::size_t view_size =
            static_cast<std::size_t>(pass.view().size());

        if (value < view_size) {
            return pass.view().stable_internal_index(view_index);
        }

        // Boundary nodes are presented as view.size()+plane. Persistent mesh
        // signatures encode them at max(Index)-1-plane.
        const std::size_t plane = value - view_size;
        const Index maximum = (std::numeric_limits<Index>::max)();
        if (plane >= static_cast<std::size_t>(maximum)) {
            throw std::overflow_error(
                "FastPolygon boundary index cannot be canonicalized");
        }
        return static_cast<Index>(
            maximum - Index{1} - static_cast<Index>(plane));
    }

    [[nodiscard]] std::size_t reciprocal_ordinal_from_update(
        const Pass& pass,
        std::size_t current_position,
        std::size_t current_ordinal,
        Index other) const {
        if (static_cast<std::size_t>(other) >= current_position) {
            throw std::logic_error(
                "FastPolygon opposite update cell has not been calculated yet");
        }
        return reciprocal_ordinal(
            pass.cell(current_position).neighbours(),
            current_ordinal,
            static_cast<Index>(current_position),
            pass.cell(static_cast<std::size_t>(other)).neighbours());
    }

    [[nodiscard]] static std::size_t reciprocal_ordinal(
        const std::vector<Index>& current_neighbours,
        std::size_t current_ordinal,
        Index current_cell,
        const std::vector<Index>& other_neighbours) {
        if (current_ordinal >= current_neighbours.size()) {
            throw std::out_of_range(
                "FastPolygon reciprocal ordinal source out of range");
        }

        const Index other = current_neighbours[current_ordinal];
        std::size_t occurrence = 0;
        for (std::size_t i = 0; i <= current_ordinal; ++i) {
            if (current_neighbours[i] == other) {
                ++occurrence;
            }
        }
        for (std::size_t i = 0; i < other_neighbours.size(); ++i) {
            if (other_neighbours[i] != current_cell) {
                continue;
            }
            if (--occurrence == 0) {
                return i;
            }
        }
        throw std::logic_error(
            "FastPolygon shared interface has no reciprocal neighbour occurrence");
    }

    [[nodiscard]] IntegralScalar* interface_block(
        std::vector<IntegralScalar>& values,
        std::size_t ordinal) const {
        if (!has_function()) {
            return nullptr;
        }
        const std::size_t begin = ordinal * components_;
        if (begin + components_ > values.size()) {
            throw std::out_of_range(
                "FastPolygon interface block outside storage");
        }
        return values.data() + begin;
    }

    void copy_interface_block(
        const std::vector<IntegralScalar>& source,
        std::size_t ordinal,
        IntegralScalar* target) const {
        const std::size_t begin = ordinal * components_;
        if (begin + components_ > source.size()) {
            throw std::out_of_range(
                "FastPolygon source interface block outside storage");
        }
        std::copy_n(source.data() + begin, components_, target);
    }

    static constexpr std::size_t npos =
        (std::numeric_limits<std::size_t>::max)();

    View* view_ = nullptr;
    Nodes* nodes_ = nullptr;
    std::size_t dimension_ = 0;
    std::size_t components_ = 0;
    const Function* function_ = nullptr;
    FacetStore* facet_store_ = nullptr;

    Checker checker_;

    std::vector<Index> neighbours_;
    std::vector<VertexData> vertices_;
    std::size_t active_vertex_count_ = 0;
    ListLevel top_lists_;
    std::vector<ListLevel> lower_lists_;
    std::vector<std::vector<Index>> active_neighbours_;
    std::vector<Index> taboo_;

    std::vector<Point> centers_;
    Point cell_point_;
    Point node_point_;
    Point evaluation_point_;
    Point distance_buffer_;

    std::vector<std::vector<IntegralScalar>> recursive_integrals_;
    std::vector<std::vector<IntegralScalar>> base_values_;
    std::vector<IntegralScalar> edge_function_first_;
    std::vector<IntegralScalar> edge_function_second_;

    std::vector<Key> key_scratch_;
    std::vector<std::vector<Index>> common_generators_;
    typename View::CellData clean_cell_scratch_;
};

} // namespace highvoronoi::detail

namespace highvoronoi::detail {

template <class Integral>
struct DefaultFastPolygonStore {
    using View = decltype(make_integration_view(std::declval<Integral&>()));
    using type = SerialFastPolygonFacetStore<
        typename View::Index,
        typename View::AreaScalar,
        typename View::IntegralScalar>;
};

template <class Integral>
struct DefaultParallelFastPolygonStore {
    using View = decltype(make_integration_view(std::declval<Integral&>()));
    using type = SharedFastPolygonFacetStore<
        typename View::Index,
        typename View::AreaScalar,
        typename View::IntegralScalar>;
};

} // namespace highvoronoi::detail

namespace highvoronoi {

/**
 * @brief C++ port of HighVoronoi.jl Fast_Polygon_Integrator.
 *
 * The default Store is serial. Store is nevertheless a public policy parameter:
 * a later Julia-style MeshDict/MultiKeyDict adapter can be injected without
 * changing FastPolygonWorkspace or its recursion. A parallel store type may be
 * a lightweight handle whose copies refer to one shared backing database.
 */
template <
    class IntegralT,
    class FunctionT = NoPolygonIntegrand,
    class StoreT = typename detail::DefaultFastPolygonStore<IntegralT>::type,
    class ParallelStoreT =
        typename detail::DefaultParallelFastPolygonStore<IntegralT>::type>
class FastPolygonAlgorithm final {
public:
    using Integral = IntegralT;
    using Function = FunctionT;
    using Store = StoreT;
    using ParallelStore = ParallelStoreT;
    using View = decltype(make_integration_view(std::declval<Integral&>()));
    using Pass = IntegrationPass<View>;
    using Workspace = detail::FastPolygonWorkspace<Pass, Function, Store>;
    using ParallelWorkspace =
        detail::FastPolygonWorkspace<Pass, Function, ParallelStore>;

    template <
        class F = Function,
        std::enable_if_t<std::is_same_v<F, NoPolygonIntegrand>, int> = 0>
    FastPolygonAlgorithm()
        : function_(NoPolygonIntegrand{}), store_() {}

    explicit FastPolygonAlgorithm(Function function)
        : function_(std::move(function)), store_() {}

    FastPolygonAlgorithm(Function function, Store store)
        : function_(std::move(function)), store_(std::move(store)) {}

    FastPolygonAlgorithm(
        Function function,
        Store store,
        ParallelStore parallel_store)
        : function_(std::move(function)),
          store_(std::move(store)),
          parallel_store_(std::move(parallel_store)) {}

    /**
     * Reset the shared Julia-style cache once per parallel integration pass.
     *
     * The integration driver invokes this hook only when more than one worker
     * will actually run, and before worker views/threads are created. Parallel
     * workers must never clear the shared store individually.
     */
    void prepare_parallel_integration(
        Integral& integral,
        std::size_t) {
        const std::size_t dimension = static_cast<std::size_t>(
            integral.mesh().extended_nodes().dimension());
        const std::size_t components =
            integral.data().integral_components();
        parallel_store_.begin_pass(
            dimension > 2 ? dimension - 2 : std::size_t{0},
            components,
            integral.stable_size());
    }

    void begin_pass(Pass& pass) {
        const std::size_t dimension = static_cast<std::size_t>(
            pass.view().mesh().extended_nodes().dimension());
        const std::size_t components =
            pass.view().integral().data().integral_components();

        store_.begin_pass(
            dimension > 2 ? dimension - 2 : std::size_t{0},
            components);

        workspace_ = std::make_unique<Workspace>(
            pass.view(),
            pass.view().mesh().extended_nodes(),
            components,
            function_,
            store_);
    }

    void integrate_cell(Pass& pass, std::size_t position) {
        require_workspace();
        workspace_->integrate_cell(pass, position);
    }

    void cleanup_cell(Pass& pass, std::size_t position) {
        require_workspace();
        workspace_->cleanup_cell(pass, position);
    }

    void end_pass(Pass&) noexcept {
        workspace_.reset();
    }

    /**
     * Worker-local FastPolygon recursion with one shared facet hierarchy.
     *
     * Julia's parallelize_integrators() duplicates PolyBuffer/checker scratch but
     * replaces every worker's DictHierarchy by views of one shared
     * ThreadsafeMeshDictHierarchy.  `parallel_store_` is that shared hierarchy;
     * stable-key conversion already happens in FastPolygonWorkspace.
     */
    class ParallelWorker final {
    public:
        ParallelWorker(
            const FastPolygonAlgorithm& parent,
            ParallelStore store)
            : parent_(&parent),
              store_(std::move(store)) {}

        void begin_pass(Pass& pass) {
            const std::size_t components =
                pass.view().integral().data().integral_components();
            workspace_ = std::make_unique<ParallelWorkspace>(
                pass.view(),
                pass.view().mesh().extended_nodes(),
                components,
                parent_->function_,
                store_);
        }

        void integrate_cell(Pass& pass, std::size_t position) {
            require_workspace();
            workspace_->integrate_cell(pass, position);
        }

        void end_pass(Pass&) noexcept {
            workspace_.reset();
        }

        [[nodiscard]] detail::FastPolygonCacheStats cache_stats() const noexcept {
            return store_.stats();
        }

    private:
        void require_workspace() const {
            if (!workspace_) {
                throw std::logic_error(
                    "FastPolygon parallel worker begin_pass was not called");
            }
        }

        const FastPolygonAlgorithm* parent_ = nullptr;
        ParallelStore store_;
        std::unique_ptr<ParallelWorkspace> workspace_;
    };

    [[nodiscard]] ParallelWorker make_worker_algorithm(std::size_t) const {
        return ParallelWorker(*this, parallel_store_);
    }

    [[nodiscard]] const detail::FastPolygonCacheStats& cache_stats() const noexcept {
        return store_.stats();
    }

    [[nodiscard]] detail::FastPolygonCacheStats parallel_cache_stats() const noexcept {
        return parallel_store_.stats();
    }

    [[nodiscard]] const Store& facet_store() const noexcept { return store_; }
    [[nodiscard]] Store& facet_store() noexcept { return store_; }

private:
    void require_workspace() const {
        if (!workspace_) {
            throw std::logic_error(
                "FastPolygonAlgorithm begin_pass was not called");
        }
    }

    Function function_;
    Store store_;
    ParallelStore parallel_store_;
    std::unique_ptr<Workspace> workspace_;
};

template <class Integral>
[[nodiscard]] FastPolygonAlgorithm<Integral>
make_fast_polygon_algorithm(Integral&) {
    return FastPolygonAlgorithm<Integral>{};
}

template <class Integral, class Function>
[[nodiscard]] auto make_fast_polygon_algorithm(
    Integral&,
    Function&& function) {
    using StoredFunction = std::decay_t<Function>;
    return FastPolygonAlgorithm<Integral, StoredFunction>(
        std::forward<Function>(function));
}

template <class Integral, class Function, class Store>
[[nodiscard]] auto make_fast_polygon_algorithm(
    Integral&,
    Function&& function,
    Store&& store) {
    using StoredFunction = std::decay_t<Function>;
    using StoredStore = std::decay_t<Store>;
    return FastPolygonAlgorithm<Integral, StoredFunction, StoredStore>(
        std::forward<Function>(function),
        std::forward<Store>(store));
}

template <class Integral, class Function, class Store, class ParallelStore>
[[nodiscard]] auto make_fast_polygon_algorithm(
    Integral&,
    Function&& function,
    Store&& store,
    ParallelStore&& parallel_store) {
    using StoredFunction = std::decay_t<Function>;
    using StoredStore = std::decay_t<Store>;
    using StoredParallelStore = std::decay_t<ParallelStore>;
    return FastPolygonAlgorithm<
        Integral,
        StoredFunction,
        StoredStore,
        StoredParallelStore>(
            std::forward<Function>(function),
            std::forward<Store>(store),
            std::forward<ParallelStore>(parallel_store));
}

} // namespace highvoronoi
