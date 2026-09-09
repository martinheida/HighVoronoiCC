#pragma once

/**
 * @file integration_view.hpp
 * @brief Ordinary Voronoi integration view with finite/unbounded update planning.
 *
 * This is the C++ analogue of Julia's IntegralView, but adapted to the stable
 * internal numbering used by HighVoronoiCC.  The persistent IntegralData stays
 * in stable-internal cell numbering.  A temporary ReorderedMeshView presents
 * the ordinary Voronoi mesh as
 *
 *   [ FINITE NEW ][ FINITE DIRTY OLD ][ UNBOUNDED NEW/DIRTY ][ CLEAN OLD ]
 *
 * Only the finite prefix is handed to the concrete integration algorithm.
 * Unbounded updates are finalized afterwards with +infinity/NaN semantics and
 * this class applies the identical presentation mapping to integral data.
 * In particular neighbour indices are translated into the temporary view and
 * area/interface entries are shuffled by exactly the same ordinal permutation.
 * Thus
 *
 *   neighbours[k] <-> area[k] <-> interface_integral[k,*]
 *
 * remains an invariant under every integration reordering.
 *
 * This file contains only ordinary/bijective Voronoi planning.  HighVoronoi's
 * projected visible/reference geometry intentionally gets a separate planner
 * in a later step while reusing the same IntegralData and CellData contract.
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
class VoronoiIntegrationView final {
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
    using MeshView = ReorderedMeshView<Mesh, IndexView>;

    struct ReciprocalInterface {
        std::size_t cell = 0;
        std::size_t ordinal = 0;
    };
    using NodePoint = typename MeshView::NodePoint;

    static_assert(
        std::is_same_v<typename Mesh::IndexMapping, DenseIndexMapping<Index>>,
        "VoronoiIntegrationView is the ordinary bijective Voronoi planner; "
        "projected HighVoronoi meshes require their own integration planner.");

    /**
     * @brief Reusable cell buffer in the temporary integration numbering.
     *
     * This is deliberately a presentation/scratch object, not persistent
     * storage.  neighbours() is read-only.  Area and interface records have the
     * exact same ordinal semantics as their neighbour entries after reordering.
     */
    class CellData final {
        friend class VoronoiIntegrationView;

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
        friend class VoronoiIntegrationView;

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

        [[nodiscard]] VoronoiIntegrationView& integration_view() {
            require_prepared();
            return *owner_;
        }
        [[nodiscard]] const VoronoiIntegrationView& integration_view() const {
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
        decltype(auto) boundary() { return mesh().boundary(); }
        decltype(auto) boundary() const { return mesh().boundary(); }

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

        VoronoiIntegrationView* owner_ = nullptr;
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

    // Julia analogue: IntegralView + the update ordering created before _integrate().
    // The persistent integral stays in stable numbering; only this temporary view moves cells.
    explicit VoronoiIntegrationView(Integral& integral)
        : integral_(&integral),
          wrapped_mesh_(&integral.mesh()) {
        // Build [NEW][DIRTY OLD][CLEAN OLD], then reuse the existing ReorderedMeshView
        // so geometry and integral presentation share exactly one index permutation.
        integral_->synchronize_size();
        build_order();
        index_view_.emplace(
            wrapped_to_view_,
            view_to_wrapped_,
            static_cast<Index>(view_to_wrapped_.size()));
        mesh_view_.emplace(*wrapped_mesh_, *index_view_);
    }

    VoronoiIntegrationView(const VoronoiIntegrationView&) = delete;
    VoronoiIntegrationView& operator=(const VoronoiIntegrationView&) = delete;
    VoronoiIntegrationView(VoronoiIntegrationView&&) = delete;
    VoronoiIntegrationView& operator=(VoronoiIntegrationView&&) = delete;

    /**
     * Build one worker-local presentation of a contiguous slice of the global
     * finite integration prefix. The assigned master cells are moved to the front while
     * all remaining cells stay available as read-only/geometric context.
     *
     * IMPORTANT: this intentionally changes public neighbour indices/order.
     * Persistent neighbour records remain in canonical stable-internal order;
     * present_raw_cell() maps them into this worker's public numbering, sorts
     * only that presentation and carries raw_ordinal with every occurrence.
     * Area and interface-integral payloads therefore undergo exactly the same
     * permutation, including duplicate neighbour occurrences.
     *
     * Every returned view owns its own ReorderedMeshView and therefore its own
     * ExtendedVoronoiNodes activation scratch.  Persistent mesh/integral data
     * remain shared and are not published by the worker view.
     */
    [[nodiscard]] std::unique_ptr<VoronoiIntegrationView> make_worker_view(
        std::size_t update_begin,
        std::size_t update_end) const {
        verify_structure();
        if (update_begin > update_end || update_end > update_count()) {
            throw std::out_of_range(
                "Integration worker range is outside the update prefix");
        }
        return std::unique_ptr<VoronoiIntegrationView>(
            new VoronoiIntegrationView(
                *integral_,
                *this,
                update_begin,
                update_end));
    }

    /**
     * Resolve every mesh-neighbour record needed by the current update prefix
     * before parallel workers start.  VoronoiMesh owns persistent neighbour
     * workspace, so neighbour construction remains a serial planning step;
     * workers subsequently perform only immutable neighbour-database reads.
     */
    void ensure_update_neighbours_current() {
        verify_structure();
        for (std::size_t position = 0; position < update_count(); ++position) {
            const Index view_cell = static_cast<Index>(position);
            const Index wrapped_cell = wrapped_public_index(view_cell);
            Address address = wrapped_mesh_->neighbour_address(wrapped_cell);
            if (wrapped_mesh_->dirty(wrapped_cell) || address == Address{0}) {
                wrapped_mesh_->compute_neighbors(wrapped_cell);
                address = wrapped_mesh_->neighbour_address(wrapped_cell);
            }
            if (address == Address{0}) {
                throw std::logic_error(
                    "Integration parallel preflight did not obtain a neighbour record");
            }
        }
    }

    [[nodiscard]] Integral& integral() noexcept { return *integral_; }
    [[nodiscard]] const Integral& integral() const noexcept { return *integral_; }

    [[nodiscard]] MeshView& mesh() {
        verify_structure();
        return *mesh_view_;
    }
    [[nodiscard]] const MeshView& mesh() const {
        verify_structure();
        return *mesh_view_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return view_to_wrapped_.size();
    }

    [[nodiscard]] std::size_t new_count() const noexcept { return new_count_; }
    [[nodiscard]] std::size_t dirty_old_count() const noexcept {
        return dirty_old_count_;
    }
    [[nodiscard]] std::size_t clean_old_count() const noexcept {
        return size() - scheduled_update_count();
    }
    [[nodiscard]] std::size_t update_count() const noexcept {
        return new_count_ + dirty_old_count_;
    }

    /** Number of NEW/DIRTY unbounded cells excluded from the integrator pass. */
    [[nodiscard]] std::size_t unbounded_update_count() const noexcept {
        return unbounded_update_count_;
    }

    /** Complete NEW/DIRTY transaction prefix: finite updates followed by unbounded updates. */
    [[nodiscard]] std::size_t scheduled_update_count() const noexcept {
        return update_count() + unbounded_update_count();
    }

    /** Public view index of one unbounded update transaction. */
    [[nodiscard]] Index unbounded_update_cell(std::size_t ordinal) const {
        if (ordinal >= unbounded_update_count()) {
            throw std::out_of_range(
                "Unbounded integration update ordinal out of range");
        }
        return static_cast<Index>(update_count() + ordinal);
    }

    [[nodiscard]] Index wrapped_public_index(Index view_cell) const {
        require_view_cell(view_cell);
        return view_to_wrapped_[static_cast<std::size_t>(view_cell)];
    }

    [[nodiscard]] Index stable_internal_index(Index view_cell) const {
        const Index wrapped = wrapped_public_index(view_cell);
        return wrapped_mesh_->index_mapping().public_to_internal(wrapped);
    }

    [[nodiscard]] bool is_new(Index view_cell) const {
        require_view_cell(view_cell);
        return static_cast<std::size_t>(view_cell) < new_count_;
    }

    [[nodiscard]] bool is_dirty_old(Index view_cell) const {
        require_view_cell(view_cell);
        const std::size_t position = static_cast<std::size_t>(view_cell);
        return position >= new_count_ && position < update_count();
    }

    [[nodiscard]] bool was_dirty_stable(Index stable_cell) const {
        const std::size_t position = static_cast<std::size_t>(stable_cell);
        return position < dirty_snapshot_.size() && dirty_snapshot_[position] != 0;
    }

    /**
     * True iff this ordinary view cell belongs to the global NEW/DIRTY set
     * captured by the master integration plan.  For worker views this is
     * deliberately broader than the worker-local update prefix.
     */
    [[nodiscard]] bool is_update_cell(Index view_cell) const {
        require_view_cell(view_cell);
        return was_dirty_stable(stable_internal_index(view_cell));
    }

    /**
     * Locate the opposite occurrence of one interface inside the current
     * update pass.  Duplicate neighbour occurrences are matched by occurrence
     * number rather than by the first equal index.
     */
    template <class Pass>
    [[nodiscard]] std::optional<ReciprocalInterface>
    reciprocal_update_interface(
        const Pass& pass,
        std::size_t position,
        std::size_t ordinal) const {
        if (position >= pass.size()) {
            throw std::out_of_range(
                "Integration reciprocal source cell outside update pass");
        }

        const auto& current = pass.cell(position);
        const auto& neighbours = current.neighbours();
        if (ordinal >= neighbours.size()) {
            throw std::out_of_range(
                "Integration reciprocal source ordinal out of range");
        }

        const Index other = neighbours[ordinal];
        const std::size_t other_position = static_cast<std::size_t>(other);
        if (other_position >= pass.size()) {
            return std::nullopt;
        }

        std::size_t occurrence = 0;
        for (std::size_t k = 0; k <= ordinal; ++k) {
            if (neighbours[k] == other) {
                ++occurrence;
            }
        }

        const auto& other_neighbours = pass.cell(other_position).neighbours();
        for (std::size_t other_ordinal = 0;
             other_ordinal < other_neighbours.size();
             ++other_ordinal) {
            if (other_neighbours[other_ordinal] != current.cell()) {
                continue;
            }
            if (--occurrence == 0) {
                return ReciprocalInterface{other_position, other_ordinal};
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Read one committed integral cell in the temporary view numbering.
     *
     * Deleted ordinary neighbours from an old historical snapshot have no
     * current view representation and are omitted together with their aligned
     * area/interface entries. Boundary neighbours are translated from stable
     * high-end encoding to `view.size() + plane`.
     */
    [[nodiscard]] bool read_cell(Index view_cell, CellData& output) const {
        verify_structure();
        const Index stable_cell = stable_internal_index(view_cell);
        const bool complete = integral_->data().read_cell(
            stable_cell,
            output.raw_);
        present_raw_cell(output);
        return complete;
    }

    /** Prepare one prefix cell for a future concrete integrator. */
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

        // Julia order, step 2 and 3: read the old committed neighbour snapshot
        // and the raw integral cell data aligned with that old snapshot.
        update.had_complete_previous_data_ = integral_->data().read_cell(
            update.stable_cell_,
            update.data_.raw_);
        const Address source_address =
            update.data_.raw_.source_neighbour_address();
        update.old_internal_neighbours_.assign(
            update.data_.raw_.neighbours().begin(),
            update.data_.raw_.neighbours().end());

        // Julia order, step 4: obtain the current mesh neighbour snapshot in the
        // order in which the mesh/neighbour database provides it. The per-cell
        // presentation conversion and sorting happen only after this raw update
        // target is known.
        const Index wrapped_cell = wrapped_public_index(view_cell);
        Address target_address = wrapped_mesh_->neighbour_address(wrapped_cell);
        if (wrapped_mesh_->dirty(wrapped_cell) || target_address == Address{0}) {
            wrapped_mesh_->compute_neighbors(wrapped_cell);
            target_address = wrapped_mesh_->neighbour_address(wrapped_cell);
        }
        if (target_address == Address{0}) {
            throw std::logic_error(
                "Voronoi integration view did not obtain a neighbour record");
        }
        if (!integral_->data().neighbour_database().read(
                target_address,
                update.new_internal_neighbours_)) {
            throw std::logic_error(
                "Voronoi integration view could not read current neighbour record");
        }

        update.neighbour_snapshot_changed_ = source_address != target_address;
        update.neighbour_topology_changed_ =
            update.old_internal_neighbours_ != update.new_internal_neighbours_;

        // Julia set_neighbors analogue: migrate surviving per-interface payloads
        // from old -> new raw neighbours. This is still cell-local bookkeeping;
        // no global pre-sorting of all integral cells is performed.
        integral_->data().retarget_cell(
            update.data_.raw_,
            target_address,
            update.new_internal_neighbours_,
            update.raw_matched_,
            update.raw_area_scratch_,
            update.raw_interface_scratch_);

        // Cell-wide quantities cannot be reused merely because neighbour topology stayed
        // equal: a dirty cell may have moved vertices. Force volume/bulk recomputation.
        if (integral_->data().stores_volume()) {
            update.data_.raw_.set_volume(AreaScalar{});
        }
        if (integral_->data().stores_bulk_integral()) {
            std::fill(
                update.data_.raw_.bulk_integral().begin(),
                update.data_.raw_.bulk_integral().end(),
                IntegralScalar{});
        }

        // Convert target data into view numbering before applying invalidation.
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
        // Phase 4: decide interface-by-interface what survives. An old interface can be
        // reused only if it still exists and the opposite ordinary cell was not dirty
        // in the snapshot taken when this view was created. Boundary interfaces are redone.
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
                was_dirty_stable(raw_neighbour);
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

    /**
     * Finalize one prepared unbounded cell without invoking a concrete integrator.
     *
     * Geometry measure semantics are deliberately numeric and allocation-neutral:
     * volume is +infinity, an interface proven unbounded by an infinite edge is
     * +infinity, and every other not-computed numeric field is quiet NaN.  A
     * finite interface is copied from its finite reciprocal cell when that data
     * is available either in the current finite update pass or in committed CLEAN
     * state. Function integrals over unbounded domains remain NaN because their
     * convergence cannot be inferred from geometric unboundedness.
     *
     * Returns the number of interface occurrences set to +infinity.
     */
    template <class Pass>
    std::size_t finalize_unbounded_update(
        const Pass& pass,
        CellUpdate& update) {
        update.require_prepared();
        if (update.owner_ != this) {
            throw std::logic_error(
                "IntegrationView CellUpdate belongs to another view");
        }
        if (!is_unbounded_view_cell(update.cell_)) {
            throw std::invalid_argument(
                "Requested unbounded finalization for a bounded cell");
        }

        if ((integral_->data().stores_volume() ||
             integral_->data().stores_area()) &&
            !std::numeric_limits<AreaScalar>::has_infinity) {
            throw std::logic_error(
                "Unbounded Voronoi volume/area storage requires AreaScalar "
                "with infinity support");
        }
        if (integral_->data().stores_area() &&
            !std::numeric_limits<AreaScalar>::has_quiet_NaN) {
            throw std::logic_error(
                "Unbounded Voronoi area storage requires AreaScalar with quiet NaN support");
        }
        if ((integral_->data().stores_bulk_integral() ||
             integral_->data().stores_interface_integral()) &&
            !std::numeric_limits<IntegralScalar>::has_quiet_NaN) {
            throw std::logic_error(
                "Unbounded Voronoi function-integral storage requires IntegralScalar "
                "with quiet NaN support");
        }

        const AreaScalar area_nan =
            (std::numeric_limits<AreaScalar>::quiet_NaN)();
        const IntegralScalar integral_nan =
            (std::numeric_limits<IntegralScalar>::quiet_NaN)();

        if (integral_->data().stores_volume()) {
            update.data_.set_volume(
                (std::numeric_limits<AreaScalar>::infinity)());
        }
        if (integral_->data().stores_area()) {
            std::fill(
                update.data_.area_.begin(),
                update.data_.area_.end(),
                area_nan);
        }
        if (integral_->data().stores_bulk_integral()) {
            std::fill(
                update.data_.bulk_integral_.begin(),
                update.data_.bulk_integral_.end(),
                integral_nan);
        }
        if (integral_->data().stores_interface_integral()) {
            std::fill(
                update.data_.interface_integral_.begin(),
                update.data_.interface_integral_.end(),
                integral_nan);
        }

        std::size_t infinite_interfaces = 0;
        CellData clean_neighbour;
        const std::size_t components = integral_->data().integral_components();

        for (std::size_t ordinal = 0;
             ordinal < update.data_.neighbours_.size();
             ++ordinal) {
            const Index neighbour = update.data_.neighbours_[ordinal];

            if (is_unbounded_interface(update.cell_, neighbour)) {
                if (integral_->data().stores_area()) {
                    update.data_.area_.at(ordinal) =
                        (std::numeric_limits<AreaScalar>::infinity)();
                }
                ++infinite_interfaces;
                continue;
            }

            // Only a finite ordinary reciprocal cell can supply a finite
            // interface. Boundary mirrors and other unbounded cells have no
            // usable reciprocal integral record here.
            const std::size_t neighbour_position =
                static_cast<std::size_t>(neighbour);
            if (neighbour_position >= size() ||
                is_unbounded_view_cell(neighbour)) {
                continue;
            }

            const CellData* source_data = nullptr;
            std::optional<std::size_t> source_ordinal;

            if (neighbour_position < pass.size()) {
                const auto& reciprocal = pass.cell(neighbour_position);
                source_ordinal = reciprocal_neighbour_ordinal(
                    update.data_.neighbours_,
                    ordinal,
                    update.cell_,
                    reciprocal.neighbours());
                if (source_ordinal) {
                    source_data = &reciprocal.data();
                }
            } else if (read_cell(neighbour, clean_neighbour)) {
                source_ordinal = reciprocal_neighbour_ordinal(
                    update.data_.neighbours_,
                    ordinal,
                    update.cell_,
                    clean_neighbour.neighbours());
                if (source_ordinal) {
                    source_data = &clean_neighbour;
                }
            }

            if (source_data == nullptr || !source_ordinal) {
                continue;
            }

            if (integral_->data().stores_area() &&
                *source_ordinal < source_data->area().size()) {
                update.data_.area_.at(ordinal) =
                    source_data->area().at(*source_ordinal);
            }

            if (integral_->data().stores_interface_integral() && components != 0) {
                const std::size_t source_begin = *source_ordinal * components;
                const std::size_t target_begin = ordinal * components;
                if (source_begin + components <=
                        source_data->interface_integral().size() &&
                    target_begin + components <=
                        update.data_.interface_integral_.size()) {
                    std::copy_n(
                        source_data->interface_integral().begin() +
                            static_cast<std::ptrdiff_t>(source_begin),
                        components,
                        update.data_.interface_integral_.begin() +
                            static_cast<std::ptrdiff_t>(target_begin));
                }
            }
        }

        return infinite_interfaces;
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
     * Rebase one completed worker-local transaction into this master view.
     *
     * Worker views may use a different public permutation, but their raw
     * IntegralData payload is still aligned with the same stable neighbour
     * record.  Materialize the worker presentation back into raw order, copy
     * that transaction, then present it in the master's numbering for the
     * serial cleanup phase.  Nothing is written to persistent IntegralData.
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
                "Cannot rebase an update from a different integral");
        }

        const Index source_wrapped =
            source.owner_->wrapped_public_index(source.cell_);
        const Index target_wrapped = wrapped_public_index(master_view_cell);
        if (source_wrapped != target_wrapped) {
            throw std::logic_error(
                "Worker update does not match requested master cell");
        }

        // First restore the canonical raw-neighbour payload ordering.  The
        // worker transaction is dead after this call, so move its buffers into
        // the master transaction rather than copying potentially large vectors.
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

        // Re-present the same raw record in the master's public numbering.
        // This may change neighbour order again. raw_ordinal remains the bridge
        // that keeps neighbour/area/interface occurrences aligned.
        present_raw_cell(target.data_);

        target.matched_old_interface_.assign(
            target.data_.order_.size(),
            std::uint8_t{0});
        target.interface_recompute_.assign(
            target.data_.order_.size(),
            std::uint8_t{0});

        // Rebuild the per-interface flags from canonical raw metadata rather
        // than copying worker-presentation flags through another temporary
        // permutation. The criterion is identical to prepare_update().
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
                was_dirty_stable(raw_neighbour);
            target.interface_recompute_[ordinal] =
                recompute ? std::uint8_t{1} : std::uint8_t{0};
        }

        target.prepared_ = true;
        source.prepared_ = false;
    }

    /** Clear this integral's dirty bits and commit one global neighbour version. */
    void finish_update() {
        verify_structure();
        if (scheduled_update_count() == 0) {
            return;
        }
        // Clear integral-specific dirty bits only after the whole update prefix succeeded.
        // This mirrors Julia's fixed calculate/iterate set and keeps failures retryable.
        for (std::size_t k = 0; k < scheduled_update_count(); ++k) {
            const Index stable = stable_internal_index(static_cast<Index>(k));
            integral_->dirty_tracker()->set_dirty(
                static_cast<std::size_t>(stable),
                false);
        }
        (void)wrapped_mesh_->advance_neighbour_version();
    }

private:
    VoronoiIntegrationView(
        Integral& integral,
        const VoronoiIntegrationView& master,
        std::size_t update_begin,
        std::size_t update_end)
        : integral_(&integral),
          wrapped_mesh_(&integral.mesh()),
          public_size_snapshot_(master.public_size_snapshot_),
          stable_size_snapshot_(master.stable_size_snapshot_),
          dirty_snapshot_(master.dirty_snapshot_),
          unbounded_wrapped_cells_(master.unbounded_wrapped_cells_),
          unbounded_interface_pairs_(master.unbounded_interface_pairs_) {
        if (master.integral_ != integral_ || master.wrapped_mesh_ != wrapped_mesh_) {
            throw std::invalid_argument(
                "Integration worker view must derive from the same master integral");
        }
        if (update_begin > update_end || update_end > master.update_count()) {
            throw std::out_of_range(
                "Integration worker range is outside the master update prefix");
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

        view_to_wrapped_.reserve(public_size_snapshot_);
        // Assigned contiguous global range first.
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            master.view_to_wrapped_.begin() +
                static_cast<std::ptrdiff_t>(update_begin),
            master.view_to_wrapped_.begin() +
                static_cast<std::ptrdiff_t>(update_end));
        // Then every other finite global update cell. Unbounded update cells
        // already live behind the finite prefix and remain read-only context here.
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            master.view_to_wrapped_.begin(),
            master.view_to_wrapped_.begin() +
                static_cast<std::ptrdiff_t>(update_begin));
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            master.view_to_wrapped_.begin() +
                static_cast<std::ptrdiff_t>(update_end),
            master.view_to_wrapped_.begin() +
                static_cast<std::ptrdiff_t>(master.update_count()));
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            master.view_to_wrapped_.begin() +
                static_cast<std::ptrdiff_t>(master.update_count()),
            master.view_to_wrapped_.end());

        wrapped_to_view_.assign(public_size_snapshot_, Index{});
        for (std::size_t view_position = 0;
             view_position < view_to_wrapped_.size();
             ++view_position) {
            const Index wrapped = view_to_wrapped_[view_position];
            wrapped_to_view_[static_cast<std::size_t>(wrapped)] =
                static_cast<Index>(view_position);
        }

        index_view_.emplace(
            wrapped_to_view_,
            view_to_wrapped_,
            static_cast<Index>(view_to_wrapped_.size()));
        mesh_view_.emplace(*wrapped_mesh_, *index_view_);
    }

    void build_order() {
        public_size_snapshot_ = static_cast<std::size_t>(wrapped_mesh_->size());
        stable_size_snapshot_ =
            static_cast<std::size_t>(wrapped_mesh_->internal_size());
        if (integral_->data().size() != stable_size_snapshot_) {
            throw std::logic_error(
                "IntegralData size does not match stable mesh size");
        }

        view_to_wrapped_.clear();
        view_to_wrapped_.reserve(public_size_snapshot_);
        dirty_snapshot_.assign(stable_size_snapshot_, std::uint8_t{0});

        build_unbounded_state();

        // Partition current visible cells without touching persistent numbering.
        // Finite NEW/DIRTY cells form the actual integrator prefix. Unbounded
        // NEW/DIRTY cells follow that prefix and are finalized numerically without
        // invoking Polygon/FastPolygon/MonteCarlo/etc. CLEAN cells remain context.
        std::vector<Index> new_cells;
        std::vector<Index> dirty_old_cells;
        std::vector<Index> unbounded_new_cells;
        std::vector<Index> unbounded_dirty_old_cells;
        std::vector<Index> clean_old_cells;
        new_cells.reserve(public_size_snapshot_);
        dirty_old_cells.reserve(public_size_snapshot_);
        unbounded_new_cells.reserve(public_size_snapshot_);
        unbounded_dirty_old_cells.reserve(public_size_snapshot_);
        clean_old_cells.reserve(public_size_snapshot_);

        for (std::size_t position = 0;
             position < public_size_snapshot_;
             ++position) {
            const Index wrapped_public = static_cast<Index>(position);
            const Index stable =
                wrapped_mesh_->index_mapping().public_to_internal(wrapped_public);
            const std::size_t stable_position = static_cast<std::size_t>(stable);
            if (stable_position >= stable_size_snapshot_) {
                throw std::logic_error(
                    "Ordinary Voronoi public mapping contains invalid stable index");
            }

            const bool is_new_cell =
                integral_->data().neighbour_address(stable) == Address{0};
            const bool is_dirty_cell =
                integral_->dirty_tracker()->dirty(stable_position);
            if (is_new_cell || is_dirty_cell) {
                dirty_snapshot_[stable_position] = std::uint8_t{1};
            }

            const bool unbounded =
                !unbounded_wrapped_cells_.empty() &&
                unbounded_wrapped_cells_[static_cast<std::size_t>(wrapped_public)] !=
                    std::uint8_t{0};
            if (is_new_cell) {
                (unbounded ? unbounded_new_cells : new_cells).push_back(wrapped_public);
            } else if (is_dirty_cell) {
                (unbounded ? unbounded_dirty_old_cells : dirty_old_cells)
                    .push_back(wrapped_public);
            } else {
                clean_old_cells.push_back(wrapped_public);
            }
        }

        new_count_ = new_cells.size();
        dirty_old_count_ = dirty_old_cells.size();
        unbounded_update_count_ =
            unbounded_new_cells.size() + unbounded_dirty_old_cells.size();

        // [FINITE NEW][FINITE DIRTY][UNBOUNDED NEW/DIRTY][CLEAN]. Only the
        // first two blocks are presented to the concrete integration algorithm.
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            new_cells.begin(),
            new_cells.end());
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            dirty_old_cells.begin(),
            dirty_old_cells.end());
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            unbounded_new_cells.begin(),
            unbounded_new_cells.end());
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            unbounded_dirty_old_cells.begin(),
            unbounded_dirty_old_cells.end());
        view_to_wrapped_.insert(
            view_to_wrapped_.end(),
            clean_old_cells.begin(),
            clean_old_cells.end());

        wrapped_to_view_.assign(public_size_snapshot_, Index{});
        for (std::size_t view_position = 0;
             view_position < view_to_wrapped_.size();
             ++view_position) {
            const Index wrapped = view_to_wrapped_[view_position];
            wrapped_to_view_[static_cast<std::size_t>(wrapped)] =
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

    void build_unbounded_state() {
        unbounded_wrapped_cells_.clear();
        unbounded_interface_pairs_.clear();

        const auto infinite_edges = wrapped_mesh_->infinite_edges();
        if (infinite_edges.empty()) {
            return;
        }

        unbounded_wrapped_cells_.assign(
            public_size_snapshot_,
            std::uint8_t{0});

        for (const auto& edge : infinite_edges) {
            for (const Index generator : edge.sigma) {
                const std::size_t position =
                    static_cast<std::size_t>(generator);
                if (position < public_size_snapshot_) {
                    unbounded_wrapped_cells_[position] = std::uint8_t{1};
                }
            }

            for (std::size_t i = 0; i < edge.sigma.size(); ++i) {
                for (std::size_t j = i + 1; j < edge.sigma.size(); ++j) {
                    Index first = edge.sigma[i];
                    Index second = edge.sigma[j];
                    if (second < first) {
                        std::swap(first, second);
                    }
                    // At least one ordinary cell must own the queried interface.
                    if (static_cast<std::size_t>(first) >= public_size_snapshot_ &&
                        static_cast<std::size_t>(second) >= public_size_snapshot_) {
                        continue;
                    }
                    unbounded_interface_pairs_.emplace_back(first, second);
                }
            }
        }

        std::sort(
            unbounded_interface_pairs_.begin(),
            unbounded_interface_pairs_.end());
        unbounded_interface_pairs_.erase(
            std::unique(
                unbounded_interface_pairs_.begin(),
                unbounded_interface_pairs_.end()),
            unbounded_interface_pairs_.end());
    }

    [[nodiscard]] bool is_unbounded_view_cell(Index view_cell) const {
        require_view_cell(view_cell);
        const Index wrapped = wrapped_public_index(view_cell);
        return !unbounded_wrapped_cells_.empty() &&
               unbounded_wrapped_cells_.at(
                   static_cast<std::size_t>(wrapped)) != std::uint8_t{0};
    }

    [[nodiscard]] Index wrapped_interface_index(Index view_index) const {
        const std::size_t value = static_cast<std::size_t>(view_index);
        if (value < size()) {
            return wrapped_public_index(view_index);
        }

        const std::size_t plane = value - size();
        if (plane >= static_cast<std::size_t>(wrapped_mesh_->boundary().size())) {
            return view_index;
        }
        return static_cast<Index>(public_size_snapshot_ + plane);
    }

    [[nodiscard]] bool is_unbounded_interface(
        Index view_cell,
        Index view_neighbour) const {
        Index first = wrapped_interface_index(view_cell);
        Index second = wrapped_interface_index(view_neighbour);
        if (second < first) {
            std::swap(first, second);
        }
        return std::binary_search(
            unbounded_interface_pairs_.begin(),
            unbounded_interface_pairs_.end(),
            std::make_pair(first, second));
    }

    template <class SourceNeighbours, class TargetNeighbours>
    [[nodiscard]] static std::optional<std::size_t>
    reciprocal_neighbour_ordinal(
        const SourceNeighbours& source_neighbours,
        std::size_t source_ordinal,
        Index source_cell,
        const TargetNeighbours& target_neighbours) {
        if (source_ordinal >= source_neighbours.size()) {
            return std::nullopt;
        }
        const Index neighbour = source_neighbours[source_ordinal];
        std::size_t occurrence = 0;
        for (std::size_t k = 0; k <= source_ordinal; ++k) {
            if (source_neighbours[k] == neighbour) {
                ++occurrence;
            }
        }

        for (std::size_t target_ordinal = 0;
             target_ordinal < target_neighbours.size();
             ++target_ordinal) {
            if (target_neighbours[target_ordinal] != source_cell) {
                continue;
            }
            if (--occurrence == 0) {
                return target_ordinal;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool map_internal_neighbour_to_view(
        Index internal_neighbour,
        Index& public_neighbour) const {
        const std::size_t internal_position =
            static_cast<std::size_t>(internal_neighbour);
        // Ordinary neighbour: stable internal -> current wrapped public -> integration view.
        if (internal_position < stable_size_snapshot_) {
            const std::optional<Index> wrapped_public =
                wrapped_mesh_->index_mapping().internal_to_public(
                    internal_neighbour);
            if (!wrapped_public) {
                return false;
            }
            public_neighbour = wrapped_to_view_.at(
                static_cast<std::size_t>(*wrapped_public));
            return true;
        }

        // Boundary neighbour: decode the stable high-end plane marker and expose it as
        // view.size()+plane, matching the public mesh neighbour convention.
        const Index invalid = (std::numeric_limits<Index>::max)();
        if (internal_neighbour == invalid) {
            throw std::logic_error(
                "Integral neighbour record contains the invalid index marker");
        }
        const Index plane = invalid - Index{1} - internal_neighbour;
        if (static_cast<std::size_t>(plane) >=
            static_cast<std::size_t>(wrapped_mesh_->boundary().size())) {
            throw std::logic_error(
                "Integral neighbour record contains invalid boundary encoding");
        }
        public_neighbour = static_cast<Index>(
            public_size_snapshot_ + static_cast<std::size_t>(plane));
        return true;
    }

    void require_view_cell(Index cell) const {
        if (static_cast<std::size_t>(cell) >= size()) {
            throw std::out_of_range("Integration view cell out of range");
        }
    }

    void require_update_cell(Index cell) const {
        require_view_cell(cell);
        if (static_cast<std::size_t>(cell) >= scheduled_update_count()) {
            throw std::invalid_argument(
                "Clean OLD cell is outside the integration transaction prefix");
        }
    }

    void verify_structure() const {
        if (static_cast<std::size_t>(wrapped_mesh_->size()) !=
                public_size_snapshot_ ||
            static_cast<std::size_t>(wrapped_mesh_->internal_size()) !=
                stable_size_snapshot_) {
            throw std::logic_error(
                "Voronoi mesh structure changed while IntegrationView exists");
        }
    }

    Integral* integral_ = nullptr;
    Mesh* wrapped_mesh_ = nullptr;
    std::size_t public_size_snapshot_ = 0;
    std::size_t stable_size_snapshot_ = 0;
    std::size_t new_count_ = 0;
    std::size_t dirty_old_count_ = 0;
    std::size_t unbounded_update_count_ = 0;
    std::vector<std::uint8_t> dirty_snapshot_;
    std::vector<std::uint8_t> unbounded_wrapped_cells_;
    std::vector<std::pair<Index, Index>> unbounded_interface_pairs_;
    std::vector<Index> view_to_wrapped_;
    std::vector<Index> wrapped_to_view_;
    std::optional<IndexView> index_view_;
    std::optional<MeshView> mesh_view_;
};

template <
    class IntegralT,
    std::enable_if_t<
        std::is_same_v<
            typename IntegralT::Mesh::IndexMapping,
            DenseIndexMapping<typename IntegralT::Index>>,
        int> = 0>
[[nodiscard]] VoronoiIntegrationView<IntegralT>
make_integration_view(IntegralT& integral) {
    return VoronoiIntegrationView<IntegralT>(integral);
}

namespace detail {

template <class View, class Point, class = void>
struct HasIntegrationEvaluationWrap : std::false_type {};

template <class View, class Point>
struct HasIntegrationEvaluationWrap<
    View,
    Point,
    std::void_t<decltype(std::declval<const View&>().wrap_evaluation_point(
        std::declval<Point&>()))>> : std::true_type {};

/**
 * Geometry remains in the integration view's Euclidean construction domain.
 * Only function-evaluation points may need a view-specific projection, e.g.
 * HighVoronoi periodic wrapping back into the external fundamental domain.
 */
template <class View, class Point>
void wrap_integration_evaluation_point(
    const View& view,
    Point& point) {
    if constexpr (HasIntegrationEvaluationWrap<View, Point>::value) {
        view.wrap_evaluation_point(point);
    } else {
        (void)view;
        (void)point;
    }
}

} // namespace detail

} // namespace highvoronoi
