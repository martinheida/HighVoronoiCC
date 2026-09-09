#pragma once

/**
 * @file voronoi_integral.hpp
 * @brief Persistent integral state attached to one mesh, without update policy.
 *
 * `VoronoiIntegral` is deliberately a storage/ownership facade only.  It owns
 * the integral result databases and one integral-specific neighbour-dirty
 * tracker requested from the mesh, but it no longer decides which cells are
 * integrated, how they are reordered, or how geometry is presented to an
 * integrator.  Those responsibilities belong to an integration view/planner.
 *
 * Persistent IntegralData slots are indexed by the mesh's stable internal
 * ordinary-node index.  This is essential because public numbering may change
 * after deletion or through temporary mesh views while historical neighbour
 * records and integral records must retain stable identities.
 */

#include <highvoronoi/integration/integral_data.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace highvoronoi {

template <class MeshT,
          typename AreaScalarT = double,
          typename IntegralScalarT = double>
class VoronoiIntegral final {
public:
    using Mesh = MeshT;
    using Index = typename Mesh::Index;
    using AreaScalar = AreaScalarT;
    using IntegralScalar = IntegralScalarT;
    using Address = std::size_t;

private:
    using NeighbourDatabaseRef =
        decltype(std::declval<Mesh&>().neighbour_database());

public:
    using NeighbourDatabase = std::remove_cv_t<
        std::remove_reference_t<NeighbourDatabaseRef>>;
    using LockType = typename NeighbourDatabase::LockType;
    using Data = IntegralData<
        LockType,
        Index,
        AreaScalar,
        IntegralScalar,
        NeighbourDatabase>;
    using DirtyTrackerHandle = decltype(
        std::declval<Mesh&>().request_neighbour_dirty_tracker(true));

    static_assert(
        std::is_integral_v<Index> && std::is_unsigned_v<Index>,
        "VoronoiIntegral requires an unsigned mesh Index");

    // This object owns persistent result/history state only. It intentionally does not
    // decide integration order; make_integration_view(...) supplies that policy.
    VoronoiIntegral(
        Mesh& mesh,
        std::size_t integral_components,
        IntegralDataOptions options = {},
        std::size_t area_database_unit_length = std::size_t{65536},
        std::size_t integral_database_unit_length = std::size_t{65536})
        : mesh_(&mesh),
          data_(
              mesh.neighbour_database(),
              static_cast<std::size_t>(mesh.internal_size()),
              integral_components,
              options,
              area_database_unit_length,
              integral_database_unit_length),
          // The tracker snapshots mesh-side neighbour dirtiness for this integral.
          // Recomputing mesh neighbours may clear mesh dirty bits without losing ours.
          dirty_tracker_(mesh.request_neighbour_dirty_tracker(true)) {
        if (!dirty_tracker_) {
            throw std::logic_error(
                "Mesh returned a null neighbour dirty tracker");
        }
    }

    VoronoiIntegral(const VoronoiIntegral&) = delete;
    VoronoiIntegral& operator=(const VoronoiIntegral&) = delete;
    VoronoiIntegral(VoronoiIntegral&&) = delete;
    VoronoiIntegral& operator=(VoronoiIntegral&&) = delete;

    [[nodiscard]] Mesh& mesh() noexcept { return *mesh_; }
    [[nodiscard]] const Mesh& mesh() const noexcept { return *mesh_; }

    [[nodiscard]] Data& data() noexcept { return data_; }
    [[nodiscard]] const Data& data() const noexcept { return data_; }

    [[nodiscard]] const DirtyTrackerHandle& dirty_tracker() const noexcept {
        return dirty_tracker_;
    }

    /** Current number of visible/public cells of the attached mesh. */
    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(mesh_->size());
    }

    /** Current number of persistent stable internal ordinary-node slots. */
    [[nodiscard]] std::size_t stable_size() const noexcept {
        return static_cast<std::size_t>(mesh_->internal_size());
    }

    /**
     * @brief Grow persistent result arrays to the stable mesh slot count.
     *
     * Structural mesh mutation must not run concurrently with this operation.
     * Shrinking is intentionally not expected for an ordinary persistent mesh:
     * deletion removes public visibility but does not renumber stable slots.
     */
    void synchronize_size() {
        const std::size_t stable_count = stable_size();
        if (data_.size() != stable_count) {
            data_.resize(stable_count);
        }
    }

private:
    Mesh* mesh_ = nullptr;
    Data data_;
    DirtyTrackerHandle dirty_tracker_;
};

} // namespace highvoronoi
