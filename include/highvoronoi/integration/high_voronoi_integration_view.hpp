
#pragma once

/**
 * @file high_voronoi_integration_view.hpp
 * @brief HighVoronoi integration view over visible cells and active references.
 *
 * Persistent IntegralData remains indexed by stable internal node identity,
 * exactly as for an ordinary VoronoiIntegral.  The difference is only the
 * temporary geometry presented to a concrete integrator.
 *
 * HighVoronoi has two public notions that must not be mixed here:
 *
 *   - visible public cells are the cells for which integral results are stored;
 *   - invisible periodic reference nodes are real Euclidean generators needed
 *     by the geometry, but are not independent integral cells.
 *
 * The integration view therefore exposes one dense bijective geometry order
 *
 *   [ NEW visible ][ DIRTY OLD visible ][ CLEAN OLD visible ][ REFERENCES ]
 *
 * over HighVoronoiMesh::data_mesh().  The first prefix is the work set; clean
 * visible cells and references stay readable as geometric context.  Crucially,
 * reference nodes keep their own shifted coordinates and their own view index.
 * They are not projected onto the visible original while an integrator is
 * working.
 *
 * Integral records themselves are still owned only by visible cells.  Reading
 * integral data through a reference index resolves to its visible owner.  The
 * neighbour ordinals inside that returned record remain those of the owner's
 * canonical cell; a future periodic integrator may use reference_shift() when
 * matching the opposite interface.
 *
 * As in the ordinary view, every presentation permutation carries area and
 * interface-integral payloads with the exact same neighbour occurrence:
 *
 *   neighbours[k] <-> area[k] <-> interface_integral[k,*]
 */

#include <highvoronoi/mesh/detail/hvview.hpp>
#include <highvoronoi/mesh/mesh_view.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

template <class IntegralT>
class HighVoronoiIntegrationView final {
public:
    using Integral = IntegralT;
    using Mesh = typename Integral::Mesh;
    using Index = typename Integral::Index;
    using AreaScalar = typename Integral::AreaScalar;
    using IntegralScalar = typename Integral::IntegralScalar;
    using Address = typename Integral::Address;
    using Data = typename Integral::Data;

    using IndexView = ShuffleView<
        Index,
        std::vector<Index>,
        std::vector<Index>>;
    using DataMesh = decltype(std::declval<Mesh&>().data_mesh());
    using MeshView = ReorderedMeshView<DataMesh, IndexView>;
    using NodePoint = typename MeshView::NodePoint;

    struct ReciprocalInterface {
        std::size_t cell = 0;
        std::size_t ordinal = 0;
    };

    static_assert(
        std::is_same_v<typename Mesh::IndexMapping, ProjectedIndexMapping<Index>>,
        "HighVoronoiIntegrationView requires HighVoronoi's projected public mapping.");

    /**
     * @brief Reusable cell buffer in the temporary integration numbering.
     *
     * This is deliberately a presentation/scratch object, not persistent
     * storage.  neighbours() is read-only.  Area and interface records have the
     * exact same ordinal semantics as their neighbour entries after reordering.
     */
    class CellData final {
        friend class HighVoronoiIntegrationView;

    public:
        [[nodiscard]] AreaScalar volume() const noexcept { return volume_; }
        void set_volume(AreaScalar value) noexcept { volume_ = value; }

        [[nodiscard]] const std::vector<Index>& neighbours() const noexcept {
            return neighbours_;
        }

        [[nodiscard]] std::vector<AreaScalar>& area() noexcept { return area_; }
        [[nodiscard]] const std::vector<AreaScalar>& area() const noexcept {
            return area_;
        }

        [[nodiscard]] std::vector<IntegralScalar>& bulk_integral() noexcept {
            return bulk_integral_;
        }
        [[nodiscard]] const std::vector<IntegralScalar>& bulk_integral() const noexcept {
            return bulk_integral_;
        }

        [[nodiscard]] std::vector<IntegralScalar>& interface_integral() noexcept {
            return interface_integral_;
        }
        [[nodiscard]] const std::vector<IntegralScalar>& interface_integral() const noexcept {
            return interface_integral_;
        }

        [[nodiscard]] Address neighbour_address() const noexcept {
            return neighbour_address_;
        }
        [[nodiscard]] Address source_neighbour_address() const noexcept {
            return source_neighbour_address_;
        }

        void reserve(
            std::size_t neighbour_capacity,
            std::size_t integral_components) {
            raw_.reserve(neighbour_capacity, integral_components);
            neighbours_.reserve(neighbour_capacity);
            area_.reserve(neighbour_capacity);
            order_.reserve(neighbour_capacity);
            bulk_integral_.reserve(integral_components);
            if (integral_components != 0 &&
                neighbour_capacity >
                    (std::numeric_limits<std::size_t>::max)() /
                        integral_components) {
                throw std::overflow_error(
                    "IntegrationView CellData reserve overflow");
            }
            interface_integral_.reserve(
                neighbour_capacity * integral_components);
        }

    private:
        struct OrderEntry {
            Index neighbour{};
            std::size_t raw_ordinal = 0;
        };

        void clear_presentation() noexcept {
            volume_ = AreaScalar{};
            neighbours_.clear();
            area_.clear();
            bulk_integral_.clear();
            interface_integral_.clear();
            order_.clear();
            source_neighbour_address_ = Address{0};
            neighbour_address_ = Address{0};
        }

        typename Data::CellData raw_;
        AreaScalar volume_{};
        Address source_neighbour_address_ = Address{0};
        Address neighbour_address_ = Address{0};
        std::vector<Index> neighbours_;
        std::vector<AreaScalar> area_;
        std::vector<IntegralScalar> bulk_integral_;
        std::vector<IntegralScalar> interface_integral_;
        std::vector<OrderEntry> order_;
    };

    /** Reusable update transaction in integration-view numbering. */
    class CellUpdate final {
        friend class HighVoronoiIntegrationView;

    public:
        CellUpdate() = default;

        [[nodiscard]] Index cell() const noexcept { return cell_; }
        [[nodiscard]] Index stable_cell() const noexcept { return stable_cell_; }

        [[nodiscard]] CellData& data() noexcept { return data_; }
        [[nodiscard]] const CellData& data() const noexcept { return data_; }
        [[nodiscard]] const std::vector<Index>& neighbours() const noexcept {
            return data_.neighbours();
        }

        [[nodiscard]] bool interface_requires_recompute(
            std::size_t neighbour_ordinal) const {
            return interface_recompute_.at(neighbour_ordinal) != 0;
        }

        [[nodiscard]] bool interface_had_old_value(
            std::size_t neighbour_ordinal) const {
            return matched_old_interface_.at(neighbour_ordinal) != 0;
        }

        /** Raw/internal neighbour ordinal backing one presented neighbour. */
        [[nodiscard]] std::size_t raw_neighbour_ordinal(
            std::size_t neighbour_ordinal) const {
            require_prepared();
            return data_.order_.at(neighbour_ordinal).raw_ordinal;
        }

        [[nodiscard]] const std::vector<Index>& old_internal_neighbours() const noexcept {
            return old_internal_neighbours_;
        }

        [[nodiscard]] const std::vector<Index>& new_internal_neighbours() const noexcept {
            return new_internal_neighbours_;
        }

        [[nodiscard]] bool neighbour_snapshot_changed() const noexcept {
            return neighbour_snapshot_changed_;
        }

        [[nodiscard]] bool neighbour_topology_changed() const noexcept {
            return neighbour_topology_changed_;
        }

        [[nodiscard]] bool had_complete_previous_data() const noexcept {
            return had_complete_previous_data_;
        }

        [[nodiscard]] HighVoronoiIntegrationView& integration_view() {
            require_prepared();
            return *owner_;
        }
        [[nodiscard]] const HighVoronoiIntegrationView& integration_view() const {
            require_prepared();
            return *owner_;
        }

        [[nodiscard]] MeshView& mesh() {
            return integration_view().mesh();
        }
        [[nodiscard]] const MeshView& mesh() const {
            return integration_view().mesh();
        }

        decltype(auto) nodes() { return mesh().nodes(); }
        decltype(auto) nodes() const { return mesh().nodes(); }
        decltype(auto) vertices() { return mesh().vertices(cell_); }
        decltype(auto) vertices() const { return mesh().vertices(cell_); }
        decltype(auto) extended_nodes() { return mesh().extended_nodes(); }
        decltype(auto) extended_nodes() const { return mesh().extended_nodes(); }
        // Geometry algorithms operate on the complete internal HighVoronoi
        // construction domain.  Periodic integrands may additionally need the
        // fixed external domain for wrapping/evaluation.
        decltype(auto) boundary() { return mesh().boundary(); }
        decltype(auto) boundary() const { return mesh().boundary(); }
        decltype(auto) internal_boundary() {
            return integration_view().internal_boundary();
        }
        decltype(auto) internal_boundary() const {
            return integration_view().internal_boundary();
        }
        decltype(auto) external_boundary() {
            return integration_view().external_boundary();
        }
        decltype(auto) external_boundary() const {
            return integration_view().external_boundary();
        }

        [[nodiscard]] bool is_reference(Index view_node) const {
            return integration_view().is_reference(view_node);
        }
        [[nodiscard]] Index visible_owner_cell(Index view_node) const {
            return integration_view().visible_owner_cell(view_node);
        }
        [[nodiscard]] const typename Mesh::ShiftMask& reference_shift(
            Index view_node) const {
            return integration_view().reference_shift(view_node);
        }

        [[nodiscard]] bool read_integral_cell(
            Index other_view_cell,
            CellData& scratch) const {
            return integration_view().read_cell(other_view_cell, scratch);
        }

        void reserve(
            std::size_t neighbour_capacity,
            std::size_t integral_components) {
            data_.reserve(neighbour_capacity, integral_components);
            new_internal_neighbours_.reserve(neighbour_capacity);
            old_internal_neighbours_.reserve(neighbour_capacity);
            raw_matched_.reserve(neighbour_capacity);
            raw_area_scratch_.reserve(neighbour_capacity);
            matched_old_interface_.reserve(neighbour_capacity);
            interface_recompute_.reserve(neighbour_capacity);
            if (integral_components != 0 &&
                neighbour_capacity >
                    (std::numeric_limits<std::size_t>::max)() /
                        integral_components) {
                throw std::overflow_error(
                    "IntegrationView CellUpdate reserve overflow");
            }
            raw_interface_scratch_.reserve(
                neighbour_capacity * integral_components);
        }

    private:
        void require_prepared() const {
            if (!prepared_ || owner_ == nullptr) {
                throw std::logic_error(
                    "IntegrationView CellUpdate is not prepared");
            }
        }

        HighVoronoiIntegrationView* owner_ = nullptr;
        Index cell_{};
        Index stable_cell_{};
        CellData data_;
        std::vector<Index> old_internal_neighbours_;
        std::vector<Index> new_internal_neighbours_;
        std::vector<std::uint8_t> raw_matched_;
        std::vector<AreaScalar> raw_area_scratch_;
        std::vector<IntegralScalar> raw_interface_scratch_;
        std::vector<std::uint8_t> matched_old_interface_;
        std::vector<std::uint8_t> interface_recompute_;
        bool neighbour_snapshot_changed_ = false;
        bool neighbour_topology_changed_ = false;
        bool had_complete_previous_data_ = false;
        bool prepared_ = false;
    };

    // HighVoronoi analogue of Julia's IntegralView setup.  Persistent integral
    // state stays in stable numbering; only this temporary geometry facade moves
    // visible cells and active periodic references.
    explicit HighVoronoiIntegrationView(Integral& integral)
        : integral_(&integral),
          high_mesh_(&integral.mesh()),
          data_mesh_(integral.mesh().data_mesh()) {
        integral_->synchronize_size();
        build_order();
        index_view_.emplace(
            data_to_view_,
            view_to_data_,
            static_cast<Index>(view_to_data_.size()));
        mesh_view_.emplace(data_mesh_, *index_view_);
    }

    HighVoronoiIntegrationView(const HighVoronoiIntegrationView&) = delete;
    HighVoronoiIntegrationView& operator=(const HighVoronoiIntegrationView&) = delete;
    HighVoronoiIntegrationView(HighVoronoiIntegrationView&&) = delete;
    HighVoronoiIntegrationView& operator=(HighVoronoiIntegrationView&&) = delete;

    /**
     * Build one worker-local HighVoronoi presentation of a contiguous master
     * update slice. Assigned visible update cells move to the front, all other
     * visible cells remain context, and periodic references remain behind the
     * complete visible block.
     */
    [[nodiscard]] std::unique_ptr<HighVoronoiIntegrationView> make_worker_view(
        std::size_t update_begin,
        std::size_t update_end) const {
        verify_structure();
        if (update_begin > update_end || update_end > update_count()) {
            throw std::out_of_range(
                "HighVoronoi integration worker range is outside the update prefix");
        }
        return std::unique_ptr<HighVoronoiIntegrationView>(
            new HighVoronoiIntegrationView(
                *integral_,
                *this,
                update_begin,
                update_end));
    }

    /**
     * Resolve every mutable neighbour record required by the fixed master
     * update set before worker threads start. HighVoronoiMesh owns neighbour
     * reconstruction scratch, so this phase deliberately remains serial.
     */
    void ensure_update_neighbours_current() {
        verify_structure();
        for (std::size_t position = 0; position < update_count(); ++position) {
            const Index view_cell = static_cast<Index>(position);
            const Index data_cell = data_public_index(view_cell);
            const Index stable = stable_internal_index(view_cell);
            Address address = high_mesh_->internal_neighbour_address(stable);
            if (data_mesh_.dirty(data_cell) || address == Address{0}) {
                data_mesh_.compute_neighbors(data_cell);
                address = high_mesh_->internal_neighbour_address(stable);
            }
            if (address == Address{0}) {
                throw std::logic_error(
                    "HighVoronoi parallel integration preflight did not obtain a neighbour record");
            }
        }
    }

    [[nodiscard]] Integral& integral() noexcept { return *integral_; }
    [[nodiscard]] const Integral& integral() const noexcept { return *integral_; }

    /** Full active Euclidean geometry used by a concrete HighVoronoi integrator. */
    [[nodiscard]] MeshView& mesh() {
        verify_structure();
        return *mesh_view_;
    }
    [[nodiscard]] const MeshView& mesh() const {
        verify_structure();
        return *mesh_view_;
    }

    /** All active geometry nodes: visible cells first, then reference nodes. */
    [[nodiscard]] std::size_t size() const noexcept {
        return view_to_data_.size();
    }

    /** Number of physical/public cells for which integral results are stored. */
    [[nodiscard]] std::size_t visible_count() const noexcept {
        return visible_count_snapshot_;
    }

    [[nodiscard]] std::size_t reference_count() const noexcept {
        return size() - visible_count();
    }

    [[nodiscard]] std::size_t new_count() const noexcept { return new_count_; }
    [[nodiscard]] std::size_t dirty_old_count() const noexcept {
        return dirty_old_count_;
    }
    [[nodiscard]] std::size_t clean_old_count() const noexcept {
        return visible_count() - new_count_ - dirty_old_count_;
    }
    [[nodiscard]] std::size_t update_count() const noexcept {
        return new_count_ + dirty_old_count_;
    }

    /** Public index in HighVoronoiMesh::data_mesh() before integration reordering. */
    [[nodiscard]] Index data_public_index(Index view_cell) const {
        require_view_cell(view_cell);
        return view_to_data_[static_cast<std::size_t>(view_cell)];
    }

    /** Stable internal generator represented by one geometry-view entry. */
    [[nodiscard]] Index stable_internal_index(Index view_cell) const {
        return data_mesh_.stable_internal_node(data_public_index(view_cell));
    }

    [[nodiscard]] bool is_visible_cell(Index view_cell) const {
        require_view_cell(view_cell);
        return static_cast<std::size_t>(view_cell) < visible_count();
    }

    [[nodiscard]] bool is_reference(Index view_cell) const {
        return !is_visible_cell(view_cell);
    }

    [[nodiscard]] bool is_new(Index view_cell) const {
        require_visible_cell(view_cell);
        return static_cast<std::size_t>(view_cell) < new_count_;
    }

    [[nodiscard]] bool is_dirty_old(Index view_cell) const {
        require_visible_cell(view_cell);
        const std::size_t position = static_cast<std::size_t>(view_cell);
        return position >= new_count_ && position < update_count();
    }

    /** True only for visible cells in the global master NEW/DIRTY snapshot. */
    [[nodiscard]] bool is_update_cell(Index view_cell) const {
        require_view_cell(view_cell);
        if (!is_visible_cell(view_cell)) {
            return false;
        }
        const Index stable = stable_internal_index(view_cell);
        const std::size_t position = static_cast<std::size_t>(stable);
        return position < update_snapshot_.size() &&
               update_snapshot_[position] != std::uint8_t{0};
    }

    /**
     * @brief Resolve a geometry entry to the visible cell owning its integral.
     *
     * For visible nodes this is the node itself.  For an invisible periodic
     * reference it is the canonical visible original.  This does not alter the
     * geometry index used in neighbour lists; references remain distinct there.
     */
    [[nodiscard]] Index visible_owner_cell(Index view_cell) const {
        const Index stable = stable_internal_index(view_cell);
        const Index owner_stable = visible_owner_stable(stable);
        return view_index_of_stable(owner_stable);
    }

    /** Stable internal slot used for persistent IntegralData for this entry. */
    [[nodiscard]] Index integral_stable_internal_index(Index view_cell) const {
        return visible_owner_stable(stable_internal_index(view_cell));
    }

    /** Shift mask of an invisible reference. Visible cells return the empty/zero mask stored by the mesh. */
    [[nodiscard]] const typename Mesh::ShiftMask& reference_shift(Index view_cell) const {
        return high_mesh_->reference_shift(stable_internal_index(view_cell));
    }

    [[nodiscard]] const typename Mesh::BoundaryType& external_boundary() const noexcept {
        return high_mesh_->external_boundary();
    }

    [[nodiscard]] const typename Mesh::BoundaryType& internal_boundary() const noexcept {
        return high_mesh_->internal_boundary();
    }

    /** Raw stable dirty snapshot as received from the mesh tracker. */
    [[nodiscard]] bool was_dirty_stable(Index stable_cell) const {
        const std::size_t position = static_cast<std::size_t>(stable_cell);
        return position < dirty_snapshot_.size() && dirty_snapshot_[position] != 0;
    }

    /**
     * @brief Dirty state of the physical visible integral cell represented here.
     *
     * Periodic references are geometry-only generators and do not own integral
     * cells.  When a reference occurs as a neighbour, resolve it to its visible
     * owner and inspect only the owner's dirty bit.  A dirty reference by itself
     * never schedules or invalidates a visible integral cell.
     */
    [[nodiscard]] bool was_dirty_integral_cell(Index stable_cell) const {
        const Index owner = visible_owner_stable(stable_cell);
        const std::size_t position = static_cast<std::size_t>(owner);
        return position < dirty_snapshot_.size() &&
               dirty_snapshot_[position] != 0;
    }

    /** NEW or DIRTY state of the visible persistent owner. */
    [[nodiscard]] bool was_update_integral_cell(Index stable_cell) const {
        const Index owner = visible_owner_stable(stable_cell);
        const std::size_t position = static_cast<std::size_t>(owner);
        return position < update_snapshot_.size() &&
               update_snapshot_[position] != std::uint8_t{0};
    }

    /**
     * Translate a function-evaluation point from the unfolded HighVoronoi
     * construction domain into the external periodic fundamental domain.
     * Geometry itself is never modified by this operation.
     */
    template <class Point>
    void wrap_evaluation_point(Point& point) const {
        using Scalar = typename std::decay_t<Point>::Scalar;
        const Scalar tolerance =
            Scalar{64} * std::numeric_limits<Scalar>::epsilon();
        const std::size_t maximum_sweeps =
            (std::max)(std::size_t{1},
                       static_cast<std::size_t>(external_boundary().size()) +
                           std::size_t{1});

        for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
            bool changed = false;
            for (Index plane = Index{0};
                 plane < external_boundary().size();
                 ++plane) {
                const auto& boundary_plane = external_boundary()[plane];
                if (static_cast<Scalar>(boundary_plane.halfspace_value(point)) <=
                    tolerance) {
                    continue;
                }
                if (!boundary_plane.is_periodic()) {
                    throw std::logic_error(
                        "HighVoronoi integration evaluation point lies outside a non-periodic external boundary");
                }
                point += external_boundary().periodic_shift(plane)
                             .template cast<Scalar>();
                changed = true;
            }
            if (!changed) {
                return;
            }
        }

        for (Index plane = Index{0};
             plane < external_boundary().size();
             ++plane) {
            if (static_cast<Scalar>(
                    external_boundary()[plane].halfspace_value(point)) >
                tolerance) {
                throw std::logic_error(
                    "Could not wrap HighVoronoi integration evaluation point into the external periodic domain");
            }
        }
    }

    /**
     * Resolve the opposite update-side occurrence of an interface.
     *
     * Ordinary visible interfaces map A<->B directly. A periodic interface
     * A<->reference(B,+s) maps to B<->reference(A,-s), where the reverse shift
     * is obtained by replacing every selected periodic plane by its reciprocal
     * partner. Duplicate occurrences are matched occurrence-by-occurrence.
     */
    template <class Pass>
    [[nodiscard]] std::optional<ReciprocalInterface>
    reciprocal_update_interface(
        const Pass& pass,
        std::size_t position,
        std::size_t ordinal) const {
        if (position >= pass.size()) {
            throw std::out_of_range(
                "HighVoronoi reciprocal source cell outside update pass");
        }

        const auto& current = pass.cell(position);
        const auto& neighbours = current.neighbours();
        if (ordinal >= neighbours.size()) {
            throw std::out_of_range(
                "HighVoronoi reciprocal source ordinal out of range");
        }

        const Index neighbour = neighbours[ordinal];
        if (static_cast<std::size_t>(neighbour) >= size()) {
            // Non-periodic boundary mirrors have no opposite integral cell.
            return std::nullopt;
        }

        const Index other_cell = visible_owner_cell(neighbour);
        const std::size_t other_position =
            static_cast<std::size_t>(other_cell);
        if (other_position >= pass.size()) {
            return std::nullopt;
        }

        // Preserve duplicate multiplicity. The source occurrence count is
        // taken over the exact presented geometry generator.
        std::size_t occurrence = 0;
        for (std::size_t k = 0; k <= ordinal; ++k) {
            if (neighbours[k] == neighbour) {
                ++occurrence;
            }
        }

        const bool periodic = is_reference(neighbour);
        typename Mesh::ShiftMask reverse_shift;
        if (periodic) {
            const auto& shift = reference_shift(neighbour);
            reverse_shift.assign(shift.size(), std::uint8_t{0});
            for (Index plane = Index{0};
                 plane < external_boundary().size();
                 ++plane) {
                const std::uint8_t multiplicity =
                    shift[static_cast<std::size_t>(plane)];
                if (multiplicity == std::uint8_t{0}) {
                    continue;
                }
                const auto partner =
                    external_boundary()[plane].periodic_partner();
                if (!partner) {
                    throw std::logic_error(
                        "HighVoronoi reference shift selects a non-periodic plane");
                }
                reverse_shift[static_cast<std::size_t>(*partner)] =
                    multiplicity;
            }
        }

        const auto& other_neighbours =
            pass.cell(other_position).neighbours();
        for (std::size_t other_ordinal = 0;
             other_ordinal < other_neighbours.size();
             ++other_ordinal) {
            const Index candidate = other_neighbours[other_ordinal];
            bool reciprocal = false;

            if (!periodic) {
                reciprocal = candidate == current.cell();
            } else if (static_cast<std::size_t>(candidate) < size() &&
                       is_reference(candidate)) {
                reciprocal =
                    visible_owner_cell(candidate) == current.cell() &&
                    reference_shift(candidate) == reverse_shift;
            }

            if (!reciprocal) {
                continue;
            }
            if (--occurrence == 0) {
                return ReciprocalInterface{
                    other_position,
                    other_ordinal};
            }
        }

        return std::nullopt;
    }

    /**
     * @brief Read the last committed integral represented by one geometry entry.
     *
     * Reference entries resolve to their visible owner because references do not
     * own independent volume/area/integral records.  The returned neighbour list
     * is nevertheless presented in the full HighVoronoi geometry view, so
     * distinct periodic references keep distinct indices and aligned payloads.
     */
    [[nodiscard]] bool read_cell(Index view_cell, CellData& output) const {
        verify_structure();
        const Index stable_cell = integral_stable_internal_index(view_cell);
        const bool complete = integral_->data().read_cell(
            stable_cell,
            output.raw_);
        present_raw_cell(output);
        return complete;
    }

    /** Prepare one NEW/DIRTY visible cell for a future concrete integrator. */
    void prepare_update(Index view_cell, CellUpdate& update) {
        verify_structure();
        require_update_cell(view_cell);

        update.owner_ = this;
        update.cell_ = view_cell;
        update.stable_cell_ = stable_internal_index(view_cell);
        update.prepared_ = false;
        update.old_internal_neighbours_.clear();
        update.new_internal_neighbours_.clear();
        update.raw_matched_.clear();
        update.matched_old_interface_.clear();
        update.interface_recompute_.clear();

        // Phase 1: read the old committed visible-cell snapshot.  This is the
        // persistent source ordering to which old area/interface payloads belong.
        update.had_complete_previous_data_ = integral_->data().read_cell(
            update.stable_cell_,
            update.data_.raw_);
        const Address source_address =
            update.data_.raw_.source_neighbour_address();
        update.old_internal_neighbours_.assign(
            update.data_.raw_.neighbours().begin(),
            update.data_.raw_.neighbours().end());

        // Phase 2: obtain the current raw HighVoronoi neighbour record for the
        // canonical visible generator.  data_mesh() is bijective over all active
        // internal nodes, so no periodic reference is projected away here.
        const Index data_cell = data_public_index(view_cell);
        Address target_address =
            high_mesh_->internal_neighbour_address(update.stable_cell_);
        if (data_mesh_.dirty(data_cell) || target_address == Address{0}) {
            data_mesh_.compute_neighbors(data_cell);
            target_address =
                high_mesh_->internal_neighbour_address(update.stable_cell_);
        }
        if (target_address == Address{0}) {
            throw std::logic_error(
                "HighVoronoi integration view did not obtain a neighbour record");
        }
        if (!integral_->data().neighbour_database().read(
                target_address,
                update.new_internal_neighbours_)) {
            throw std::logic_error(
                "HighVoronoi integration view could not read current neighbour record");
        }

        update.neighbour_snapshot_changed_ = source_address != target_address;
        update.neighbour_topology_changed_ =
            update.old_internal_neighbours_ != update.new_internal_neighbours_;

        // Phase 3: Julia set_neighbors()-style migration.  Matching is performed
        // on raw stable generator identity, hence two different periodic copies
        // of the same visible node remain two independent interface occurrences.
        integral_->data().retarget_cell(
            update.data_.raw_,
            target_address,
            update.new_internal_neighbours_,
            update.raw_matched_,
            update.raw_area_scratch_,
            update.raw_interface_scratch_);

        // Cell-wide quantities belong to the whole dirty visible cell.  Equal
        // neighbour topology is not sufficient evidence that volume/bulk stayed
        // unchanged, so both are reset for the concrete integrator.
        if (integral_->data().stores_volume()) {
            update.data_.raw_.set_volume(AreaScalar{});
        }
        if (integral_->data().stores_bulk_integral()) {
            std::fill(
                update.data_.raw_.bulk_integral().begin(),
                update.data_.raw_.bulk_integral().end(),
                IntegralScalar{});
        }

        present_raw_cell(update.data_);

        // Julia cell-local order: activate this cell's extended-node scratch
        // immediately before the concrete algorithm is called.  The generic
        // Integrator guarantees that no other cell is prepared in between.
        mesh().extended_nodes().activate_cell(
            view_cell,
            update.data_.neighbours());

        update.matched_old_interface_.assign(
            update.data_.order_.size(),
            std::uint8_t{0});
        update.interface_recompute_.assign(
            update.data_.order_.size(),
            std::uint8_t{0});

        const std::size_t components = integral_->data().integral_components();
        // Phase 4: preserve a surviving interface only when the corresponding
        // physical neighbour cell was clean at view construction.  A periodic
        // reference is resolved to its visible owner for this decision; the
        // reference's own dirty bit is intentionally irrelevant.  Boundary
        // interfaces and newly appearing raw reference occurrences are always
        // recalculated.
        for (std::size_t view_ordinal = 0;
             view_ordinal < update.data_.order_.size();
             ++view_ordinal) {
            const std::size_t raw_ordinal =
                update.data_.order_[view_ordinal].raw_ordinal;
            const Index raw_neighbour =
                update.data_.raw_.neighbours()[raw_ordinal];
            const bool matched =
                raw_ordinal < update.raw_matched_.size() &&
                update.raw_matched_[raw_ordinal] != std::uint8_t{0};
            update.matched_old_interface_[view_ordinal] =
                matched ? std::uint8_t{1} : std::uint8_t{0};

            const bool ordinary =
                static_cast<std::size_t>(raw_neighbour) < stable_size_snapshot_;
            const bool recompute =
                !update.had_complete_previous_data_ ||
                !matched ||
                !ordinary ||
                was_update_integral_cell(raw_neighbour);
            update.interface_recompute_[view_ordinal] =
                recompute ? std::uint8_t{1} : std::uint8_t{0};

            if (!recompute) {
                continue;
            }
            if (integral_->data().stores_area() &&
                view_ordinal < update.data_.area_.size()) {
                update.data_.area_[view_ordinal] = AreaScalar{};
            }
            if (integral_->data().stores_interface_integral() &&
                components != 0) {
                const std::size_t begin = view_ordinal * components;
                std::fill_n(
                    update.data_.interface_integral_.begin() +
                        static_cast<std::ptrdiff_t>(begin),
                    components,
                    IntegralScalar{});
            }
        }

        update.prepared_ = true;
    }

    /** Restore this cell's mutable extended-node scratch before cleanup. */
    void prepare_cleanup(CellUpdate& update) {
        update.require_prepared();
        if (update.owner_ != this) {
            throw std::logic_error(
                "IntegrationView CellUpdate belongs to another view");
        }
        mesh().extended_nodes().activate_cell(
            update.cell_,
            update.data_.neighbours());
    }

    /** Publish final first-pass + cleanup values and finalize the transaction. */
    void commit_update(CellUpdate& update) {
        update.require_prepared();
        if (update.owner_ != this) {
            throw std::logic_error(
                "IntegrationView CellUpdate belongs to another view");
        }
        copy_presentation_to_raw(update.data_);
        integral_->data().write_cell(update.stable_cell_, update.data_.raw_);
        update.prepared_ = false;
    }

    /**
     * Rebase one completed worker-local HighVoronoi transaction into this
     * master view. Raw persistent neighbour ordering is the invariant bridge;
     * worker/master presentation permutations are rebuilt independently.
     */
    void rebase_worker_update(
        CellUpdate& source,
        Index master_view_cell,
        CellUpdate& target) {
        verify_structure();
        require_update_cell(master_view_cell);
        source.require_prepared();
        if (source.owner_ == nullptr || source.owner_->integral_ != integral_) {
            throw std::logic_error(
                "Cannot rebase a HighVoronoi update from a different integral");
        }

        const Index target_stable = stable_internal_index(master_view_cell);
        if (source.stable_cell_ != target_stable) {
            throw std::logic_error(
                "HighVoronoi worker update does not match requested master cell");
        }

        source.owner_->copy_presentation_to_raw(source.data_);

        target.owner_ = this;
        target.cell_ = master_view_cell;
        target.stable_cell_ = source.stable_cell_;
        target.data_ = std::move(source.data_);
        target.old_internal_neighbours_ =
            std::move(source.old_internal_neighbours_);
        target.new_internal_neighbours_ =
            std::move(source.new_internal_neighbours_);
        target.raw_matched_ = std::move(source.raw_matched_);
        target.raw_area_scratch_ = std::move(source.raw_area_scratch_);
        target.raw_interface_scratch_ =
            std::move(source.raw_interface_scratch_);
        target.neighbour_snapshot_changed_ = source.neighbour_snapshot_changed_;
        target.neighbour_topology_changed_ = source.neighbour_topology_changed_;
        target.had_complete_previous_data_ = source.had_complete_previous_data_;

        present_raw_cell(target.data_);

        target.matched_old_interface_.assign(
            target.data_.order_.size(),
            std::uint8_t{0});
        target.interface_recompute_.assign(
            target.data_.order_.size(),
            std::uint8_t{0});

        for (std::size_t ordinal = 0;
             ordinal < target.data_.order_.size();
             ++ordinal) {
            const std::size_t raw_ordinal =
                target.data_.order_[ordinal].raw_ordinal;
            const Index raw_neighbour =
                target.data_.raw_.neighbours().at(raw_ordinal);
            const bool matched =
                raw_ordinal < target.raw_matched_.size() &&
                target.raw_matched_[raw_ordinal] != std::uint8_t{0};
            target.matched_old_interface_[ordinal] =
                matched ? std::uint8_t{1} : std::uint8_t{0};

            const bool ordinary =
                static_cast<std::size_t>(raw_neighbour) < stable_size_snapshot_;
            const bool recompute =
                !target.had_complete_previous_data_ ||
                !matched ||
                !ordinary ||
                was_update_integral_cell(raw_neighbour);
            target.interface_recompute_[ordinal] =
                recompute ? std::uint8_t{1} : std::uint8_t{0};
        }

        target.prepared_ = true;
        source.prepared_ = false;
    }

    /**
     * @brief Finish one global HighVoronoi integral update.
     *
     * Only visible integral cells in the successfully processed update prefix
     * are cleared.  Periodic reference dirty bits are geometry-side information
     * and are deliberately ignored by the integral planner.
     */
    void finish_update() {
        verify_structure();
        if (update_count() == 0) {
            return;
        }

        for (std::size_t k = 0; k < update_count(); ++k) {
            const Index stable = stable_internal_index(static_cast<Index>(k));
            integral_->dirty_tracker()->set_dirty(
                static_cast<std::size_t>(stable),
                false);
        }

        (void)high_mesh_->advance_neighbour_version();
    }

private:
    HighVoronoiIntegrationView(
        Integral& integral,
        const HighVoronoiIntegrationView& master,
        std::size_t update_begin,
        std::size_t update_end)
        : integral_(&integral),
          high_mesh_(&integral.mesh()),
          data_mesh_(integral.mesh().data_mesh()),
          visible_count_snapshot_(master.visible_count_snapshot_),
          stable_size_snapshot_(master.stable_size_snapshot_),
          active_size_snapshot_(master.active_size_snapshot_),
          dirty_snapshot_(master.dirty_snapshot_),
          update_snapshot_(master.update_snapshot_) {
        if (master.integral_ != integral_ || master.high_mesh_ != high_mesh_) {
            throw std::invalid_argument(
                "HighVoronoi worker view must derive from the same master integral");
        }
        if (update_begin > update_end || update_end > master.update_count()) {
            throw std::out_of_range(
                "HighVoronoi worker range is outside the master update prefix");
        }

        const std::size_t local_count = update_end - update_begin;
        new_count_ = std::size_t{0};
        for (std::size_t position = update_begin;
             position < update_end;
             ++position) {
            if (position < master.new_count_) {
                ++new_count_;
            }
        }
        dirty_old_count_ = local_count - new_count_;

        view_to_data_.reserve(active_size_snapshot_);

        // Assigned global update range first.
        view_to_data_.insert(
            view_to_data_.end(),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(update_begin),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(update_end));

        // Remaining global update cells stay before CLEAN visible context.
        view_to_data_.insert(
            view_to_data_.end(),
            master.view_to_data_.begin(),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(update_begin));
        view_to_data_.insert(
            view_to_data_.end(),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(update_end),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(master.update_count()));

        // Then every CLEAN visible cell, and only after the complete visible
        // block the geometry-only periodic references.
        view_to_data_.insert(
            view_to_data_.end(),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(master.update_count()),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(master.visible_count()));
        view_to_data_.insert(
            view_to_data_.end(),
            master.view_to_data_.begin() +
                static_cast<std::ptrdiff_t>(master.visible_count()),
            master.view_to_data_.end());

        if (view_to_data_.size() != active_size_snapshot_) {
            throw std::logic_error(
                "HighVoronoi worker planner did not cover every active node");
        }

        data_to_view_.assign(active_size_snapshot_, Index{});
        for (std::size_t view_position = 0;
             view_position < view_to_data_.size();
             ++view_position) {
            const Index data_public = view_to_data_[view_position];
            data_to_view_.at(static_cast<std::size_t>(data_public)) =
                static_cast<Index>(view_position);
        }

        index_view_.emplace(
            data_to_view_,
            view_to_data_,
            static_cast<Index>(view_to_data_.size()));
        mesh_view_.emplace(data_mesh_, *index_view_);
    }

    /** Build [NEW visible][DIRTY OLD visible][CLEAN OLD visible][REFERENCES]. */
    void build_order() {
        visible_count_snapshot_ =
            static_cast<std::size_t>(high_mesh_->visible_public_count());
        stable_size_snapshot_ =
            static_cast<std::size_t>(high_mesh_->internal_node_count());
        active_size_snapshot_ = static_cast<std::size_t>(data_mesh_.size());

        if (integral_->data().size() != stable_size_snapshot_) {
            throw std::logic_error(
                "IntegralData size does not match HighVoronoi stable mesh size");
        }

        dirty_snapshot_.assign(stable_size_snapshot_, std::uint8_t{0});
        update_snapshot_.assign(stable_size_snapshot_, std::uint8_t{0});

        // First snapshot raw dirty state for every persistent stable slot.  The
        // integral tracker survives mesh-neighbour recomputation, so this set is
        // fixed for the lifetime of the integration view.
        for (std::size_t stable_position = 0;
             stable_position < stable_size_snapshot_;
             ++stable_position) {
            if (integral_->dirty_tracker()->dirty(stable_position)) {
                dirty_snapshot_[stable_position] = std::uint8_t{1};
            }
        }

        std::vector<Index> new_cells;
        std::vector<Index> dirty_old_cells;
        std::vector<Index> clean_old_cells;
        std::vector<Index> reference_cells;
        new_cells.reserve(visible_count_snapshot_);
        dirty_old_cells.reserve(visible_count_snapshot_);
        clean_old_cells.reserve(visible_count_snapshot_);
        reference_cells.reserve(active_size_snapshot_ - visible_count_snapshot_);

        // Preserve the current visible-public order inside each visible block.
        for (std::size_t public_position = 0;
             public_position < visible_count_snapshot_;
             ++public_position) {
            const Index visible_public = static_cast<Index>(public_position);
            const Index stable =
                high_mesh_->visible_public_to_internal(visible_public);
            const bool is_new_cell =
                integral_->data().neighbour_address(stable) == Address{0};
            const bool is_dirty_cell =
                dirty_snapshot_[static_cast<std::size_t>(stable)] != std::uint8_t{0};

            if (is_new_cell || is_dirty_cell) {
                update_snapshot_[static_cast<std::size_t>(stable)] =
                    std::uint8_t{1};
            }

            const Index data_public = data_index_of_stable(stable);
            if (is_new_cell) {
                new_cells.push_back(data_public);
            } else if (is_dirty_cell) {
                dirty_old_cells.push_back(data_public);
            } else {
                clean_old_cells.push_back(data_public);
            }
        }

        // Reference nodes are geometry context only.  Keep them after every
        // visible cell, in stable insertion order, so their shifted coordinates
        // and raw identities remain reproducible and easy to inspect.
        for (Index stable = Index{0};
             static_cast<std::size_t>(stable) < stable_size_snapshot_;
             ++stable) {
            if (!high_mesh_->is_active_internal(stable) ||
                high_mesh_->is_visible_internal(stable)) {
                continue;
            }
            reference_cells.push_back(data_index_of_stable(stable));
        }

        new_count_ = new_cells.size();
        dirty_old_count_ = dirty_old_cells.size();

        view_to_data_.clear();
        view_to_data_.reserve(active_size_snapshot_);
        view_to_data_.insert(
            view_to_data_.end(), new_cells.begin(), new_cells.end());
        view_to_data_.insert(
            view_to_data_.end(), dirty_old_cells.begin(), dirty_old_cells.end());
        view_to_data_.insert(
            view_to_data_.end(), clean_old_cells.begin(), clean_old_cells.end());
        view_to_data_.insert(
            view_to_data_.end(), reference_cells.begin(), reference_cells.end());

        if (view_to_data_.size() != active_size_snapshot_) {
            throw std::logic_error(
                "HighVoronoi integration planner did not cover every active node");
        }

        data_to_view_.assign(active_size_snapshot_, Index{});
        for (std::size_t view_position = 0;
             view_position < view_to_data_.size();
             ++view_position) {
            const Index data_public = view_to_data_[view_position];
            data_to_view_.at(static_cast<std::size_t>(data_public)) =
                static_cast<Index>(view_position);
        }
    }

    void present_raw_cell(CellData& output) const {
        output.clear_presentation();
        output.volume_ = output.raw_.volume();
        output.source_neighbour_address_ =
            output.raw_.source_neighbour_address();
        output.neighbour_address_ = output.raw_.neighbour_address();
        output.bulk_integral_ = output.raw_.bulk_integral();

        const auto& raw_neighbours = output.raw_.neighbours();
        output.order_.clear();
        output.order_.reserve(raw_neighbours.size());

        // Translate each historical stable/internal neighbour into the temporary view.
        // Deleted ordinary cells have no current representation and are dropped together
        // with their aligned payload ordinal; boundary encodings are translated separately.
        for (std::size_t raw_ordinal = 0;
             raw_ordinal < raw_neighbours.size();
             ++raw_ordinal) {
            Index public_neighbour{};
            if (!map_internal_neighbour_to_view(
                    raw_neighbours[raw_ordinal],
                    public_neighbour)) {
                // Historical reference to a deleted ordinary node.
                continue;
            }
            output.order_.push_back(
                typename CellData::OrderEntry{public_neighbour, raw_ordinal});
        }

        // Sort only the presentation order. raw_ordinal travels with every neighbour so
        // area/interface payloads are shuffled by exactly the same permutation.
        std::stable_sort(
            output.order_.begin(),
            output.order_.end(),
            [](const auto& left, const auto& right) {
                return left.neighbour < right.neighbour;
            });

        output.neighbours_.resize(output.order_.size());
        for (std::size_t k = 0; k < output.order_.size(); ++k) {
            output.neighbours_[k] = output.order_[k].neighbour;
        }

        const auto& raw_area = output.raw_.area();
        if (!raw_area.empty()) {
            output.area_.resize(output.order_.size());
            for (std::size_t k = 0; k < output.order_.size(); ++k) {
                const std::size_t raw_ordinal = output.order_[k].raw_ordinal;
                if (raw_ordinal >= raw_area.size()) {
                    throw std::logic_error(
                        "Integral area record is not aligned with neighbour record");
                }
                output.area_[k] = raw_area[raw_ordinal];
            }
        }

        const auto& raw_interface = output.raw_.interface_integral();
        const std::size_t components = integral_->data().integral_components();
        if (!raw_interface.empty()) {
            output.interface_integral_.resize(
                output.order_.size() * components);
            for (std::size_t k = 0; k < output.order_.size(); ++k) {
                const std::size_t raw_ordinal = output.order_[k].raw_ordinal;
                const std::size_t raw_begin = raw_ordinal * components;
                const std::size_t view_begin = k * components;
                if (raw_begin + components > raw_interface.size()) {
                    throw std::logic_error(
                        "Integral interface record is not aligned with neighbour record");
                }
                std::copy_n(
                    raw_interface.begin() +
                        static_cast<std::ptrdiff_t>(raw_begin),
                    components,
                    output.interface_integral_.begin() +
                        static_cast<std::ptrdiff_t>(view_begin));
            }
        }
    }

    // Inverse of present_raw_cell(): write modified values back to the canonical raw
    // neighbour ordinals before IntegralData commits the immutable database records.
    void copy_presentation_to_raw(CellData& data) const {
        data.raw_.set_volume(data.volume_);
        if (integral_->data().stores_bulk_integral()) {
            data.raw_.bulk_integral() = data.bulk_integral_;
        }
        if (integral_->data().stores_area()) {
            if (data.area_.size() != data.order_.size()) {
                throw std::logic_error(
                    "Integration view area count does not match neighbours");
            }
            auto& raw_area = data.raw_.area();
            for (std::size_t k = 0; k < data.order_.size(); ++k) {
                raw_area.at(data.order_[k].raw_ordinal) = data.area_[k];
            }
        }
        if (integral_->data().stores_interface_integral()) {
            const std::size_t components = integral_->data().integral_components();
            if (data.interface_integral_.size() !=
                data.order_.size() * components) {
                throw std::logic_error(
                    "Integration view interface-integral layout mismatch");
            }
            auto& raw_interface = data.raw_.interface_integral();
            for (std::size_t k = 0; k < data.order_.size(); ++k) {
                const std::size_t raw_begin =
                    data.order_[k].raw_ordinal * components;
                const std::size_t view_begin = k * components;
                std::copy_n(
                    data.interface_integral_.begin() +
                        static_cast<std::ptrdiff_t>(view_begin),
                    components,
                    raw_interface.begin() +
                        static_cast<std::ptrdiff_t>(raw_begin));
            }
        }
    }

    [[nodiscard]] Index data_index_of_stable(Index stable) const {
        const std::optional<Index> data_public =
            data_mesh_.public_node_of_global_internal(stable);
        if (!data_public) {
            throw std::logic_error(
                "Active HighVoronoi stable node is missing from data_mesh()");
        }
        return *data_public;
    }

    [[nodiscard]] Index view_index_of_stable(Index stable) const {
        const Index data_public = data_index_of_stable(stable);
        return data_to_view_.at(static_cast<std::size_t>(data_public));
    }

    /** Map a visible/reference stable node to the stable visible integral owner. */
    [[nodiscard]] Index visible_owner_stable(Index stable) const {
        if (static_cast<std::size_t>(stable) >= stable_size_snapshot_) {
            throw std::out_of_range(
                "HighVoronoi stable node outside integration snapshot");
        }
        if (high_mesh_->is_reference_internal(stable)) {
            const std::optional<Index> reference =
                high_mesh_->reference_internal(stable);
            if (!reference) {
                throw std::logic_error(
                    "HighVoronoi reference node has no visible owner");
            }
            return *reference;
        }
        return stable;
    }

    [[nodiscard]] bool map_internal_neighbour_to_view(
        Index internal_neighbour,
        Index& public_neighbour) const {
        const std::size_t internal_position =
            static_cast<std::size_t>(internal_neighbour);

        // Ordinary generator: keep every active periodic reference distinct.
        // Historical entries to deleted generators are omitted with their
        // aligned area/interface payload occurrence.
        if (internal_position < stable_size_snapshot_) {
            const std::optional<Index> data_public =
                data_mesh_.public_node_of_global_internal(internal_neighbour);
            if (!data_public) {
                return false;
            }
            public_neighbour = data_to_view_.at(
                static_cast<std::size_t>(*data_public));
            return true;
        }

        // Boundary neighbours keep the common HighVoronoiCC high-end stable
        // encoding internally and become view.size()+plane in presentation.
        const Index invalid = (std::numeric_limits<Index>::max)();
        if (internal_neighbour == invalid) {
            throw std::logic_error(
                "Integral neighbour record contains the invalid index marker");
        }
        const Index plane = invalid - Index{1} - internal_neighbour;
        if (static_cast<std::size_t>(plane) >=
            static_cast<std::size_t>(internal_boundary().size())) {
            throw std::logic_error(
                "Integral neighbour record contains invalid boundary encoding");
        }
        public_neighbour = static_cast<Index>(
            size() + static_cast<std::size_t>(plane));
        return true;
    }

    void require_view_cell(Index cell) const {
        if (static_cast<std::size_t>(cell) >= size()) {
            throw std::out_of_range(
                "HighVoronoi integration view cell out of range");
        }
    }

    void require_visible_cell(Index cell) const {
        require_view_cell(cell);
        if (static_cast<std::size_t>(cell) >= visible_count()) {
            throw std::invalid_argument(
                "Periodic reference node is geometry context, not an integral cell");
        }
    }

    void require_update_cell(Index cell) const {
        require_visible_cell(cell);
        if (static_cast<std::size_t>(cell) >= update_count()) {
            throw std::invalid_argument(
                "CLEAN OLD visible cell is outside the integration update prefix");
        }
    }

    void verify_structure() const {
        if (static_cast<std::size_t>(high_mesh_->visible_public_count()) !=
                visible_count_snapshot_ ||
            static_cast<std::size_t>(high_mesh_->internal_node_count()) !=
                stable_size_snapshot_ ||
            static_cast<std::size_t>(data_mesh_.size()) !=
                active_size_snapshot_) {
            throw std::logic_error(
                "HighVoronoi mesh structure changed while IntegrationView exists");
        }
    }

    Integral* integral_ = nullptr;
    Mesh* high_mesh_ = nullptr;
    DataMesh data_mesh_;
    std::size_t visible_count_snapshot_ = 0;
    std::size_t stable_size_snapshot_ = 0;
    std::size_t active_size_snapshot_ = 0;
    std::size_t new_count_ = 0;
    std::size_t dirty_old_count_ = 0;
    std::vector<std::uint8_t> dirty_snapshot_;
    std::vector<std::uint8_t> update_snapshot_;
    std::vector<Index> view_to_data_;
    std::vector<Index> data_to_view_;
    std::optional<IndexView> index_view_;
    std::optional<MeshView> mesh_view_;

};

// Same make_integration_view() entry point as for an ordinary Voronoi mesh.
// The mapping policy selects this HighVoronoi planner without introducing a
// separate persistent HighVoronoiIntegral storage class.
template <
    class IntegralT,
    std::enable_if_t<
        std::is_same_v<
            typename IntegralT::Mesh::IndexMapping,
            ProjectedIndexMapping<typename IntegralT::Index>>,
        int> = 0>
[[nodiscard]] HighVoronoiIntegrationView<IntegralT>
make_integration_view(IntegralT& integral) {
    return HighVoronoiIntegrationView<IntegralT>(integral);
}

} // namespace highvoronoi


