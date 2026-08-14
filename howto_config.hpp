#pragma once

/**
 * @file howto_config.hpp
 * @brief Editable configuration guide for the current HighVoronoiCC core.
 *
 * This file is intentionally written as C++ instead of prose documentation.
 * Open it in an editor, copy the parts you need, and change the marked lines.
 * It documents the current C++17 configuration interface for:
 *
 * - SingleThread / MultiThread;
 * - persistent database QueueHash;
 * - cell-local QueueHash;
 * - cell-local EdgeHash;
 * - HVDataBase;
 * - VoronoiMesh;
 * - search tree and RayCaster;
 * - ComputeVoronoi.
 *
 * SerialMesh, refinement and periodic-composite construction are deliberately
 * not covered here yet. They belong to the next architecture layer.
 *
 * IMPORTANT:
 * This is a HOW-TO header, not a header that applications need to include.
 * The recommended use is to copy the Config block or the complete example into
 * the application that constructs a HighVoronoi mesh.
 */

#include <highvoronoi/parameters.hpp>

#include <highvoronoi/detail/hvdatabase.hpp>

#include <highvoronoi/geometry/boundary.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/nanoflann_search_tree_crtp.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/geometry/voronoi_nodes.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi_howto_config {

// ============================================================================
// 1. EDIT THIS BLOCK
// ============================================================================
//
// This block collects the choices a normal user is most likely to change.
// The rest of the file explains every choice in detail.
//
// The helper functions deliberately use `auto`.  This makes the selected type
// available through decltype(...) without repeating it elsewhere.
//
struct Config {
    // ------------------------------------------------------------------------
    // Numeric types and dimension
    // ------------------------------------------------------------------------

    using NodeScalar = double;
    using VertexScalar = double;
    using Index = std::uint32_t;

    static constexpr int Dimension = 3;

    // VoronoiMesh requires an unsigned Index.  Therefore choose Index
    // explicitly.  The formal DataBaseParams<> default is currently
    // std::int64_t and is NOT directly suitable for VoronoiMesh.


    // ------------------------------------------------------------------------
    // Threading
    // ------------------------------------------------------------------------
    // Library default for both axes: SingleThread{}.
    //
    // Alternatives:
    //     return highvoronoi::MultiThread{4};
    //
    // MeshThreading:
    //     independent SystematicVoronoi mesh branches.
    //
    // CastThreading:
    //     several geometry workers inside one SystematicVoronoi branch.
    //
    // Both axes may be MultiThread at the same time.

    inline static auto mesh_threading_value = highvoronoi::SingleThread{};
    inline static auto cast_threading_value = highvoronoi::SingleThread{};

    using MeshThreading =
        std::decay_t<decltype(mesh_threading_value)>;
    using CastThreading =
        std::decay_t<decltype(cast_threading_value)>;

    [[nodiscard]] static MeshThreading mesh_threading() {
        return mesh_threading_value;
    }

    [[nodiscard]] static CastThreading cast_threading() {
        return cast_threading_value;
    }

    // A parallel ComputeVoronoi requires a ReadWriteLock database.  Keeping
    // this selection derived from the two threading choices avoids a common
    // configuration error.
    using DatabaseLock = std::conditional_t<
        MeshThreading::is_multithreaded || CastThreading::is_multithreaded,
        highvoronoi::ReadWriteLock,
        highvoronoi::EmptyLock>;


    // ------------------------------------------------------------------------
    // Hash generator
    // ------------------------------------------------------------------------
    // Default in DataBaseParams and EdgeBufferParams:
    //     highvoronoi::FNV64_128HashGenerator
    //
    // Common alternative:
    //     highvoronoi::Murmur128HashGenerator<>
    //
    // More alternatives are documented below.

    using HashGenerator = highvoronoi::FNV64_128HashGenerator;


    // ------------------------------------------------------------------------
    // Persistent DATABASE hash
    // ------------------------------------------------------------------------
    // This QueueHash detects duplicate persistent records in HVDataBase.
    //
    // Library default:
    //     highvoronoi::DirectHash{8}
    //
    // Alternatives:
    //     highvoronoi::StaticHash<16>{1024}
    //     highvoronoi::DynamicHash<>{4, 10000, 1024}
    //
    // StaticHash<N> is particularly useful for parallel construction because
    // the N subtables have independent locks.

    inline static auto database_hash_value = highvoronoi::DirectHash{1024};

    using DatabaseHashMode =
        std::decay_t<decltype(database_hash_value)>;

    [[nodiscard]] static DatabaseHashMode database_hash() {
        return database_hash_value;
    }


    // ------------------------------------------------------------------------
    // Cell-local VERTEX QUEUE hash
    // ------------------------------------------------------------------------
    // This is NOT the database hash.  It belongs to one SystematicVoronoi and
    // is reset for every cell.  It claims vertex signatures for the current
    // work queue.
    //
    // ComputeVoronoi's type-level default is a DataBaseParams using:
    //     FNV64_128HashGenerator
    //     DirectHash
    //     QueueTable
    //
    // NOTE ABOUT CONTENTION:
    // VoronoiVertexQueue currently places one outer QueueLock around claim,
    // enqueue, pop and reset, and therefore instantiates its internal QueueHash
    // with EmptyLock.  StaticHash<N> still partitions the hash storage, but it
    // does NOT remove contention on that outer queue lock in the current code.

    inline static auto queue_hash_value = highvoronoi::DirectHash{256};

    using QueueHashMode =
        std::decay_t<decltype(queue_hash_value)>;

    [[nodiscard]] static QueueHashMode queue_hash() {
        return queue_hash_value;
    }


    // ------------------------------------------------------------------------
    // Cell-local EDGE hash
    // ------------------------------------------------------------------------
    // This tracks edge occurrences inside one SystematicVoronoi.
    //
    // Library default:
    //     highvoronoi::DirectHash{8}
    //
    // For parallel construction StaticHash<N> is often attractive because
    // different first edge indices are routed to independent locked subtables.
    // EdgeHash and the shared FEIStorageCache use VoronoiThreading::RWLock,
    // i.e. ReadWriteLock whenever MeshThreading OR CastThreading is parallel.

    inline static auto edge_hash_value = highvoronoi::DirectHash{512};
    // Parallel-oriented alternative:
    // inline static auto edge_hash_value = highvoronoi::StaticHash<16>{512};

    using EdgeHashMode =
        std::decay_t<decltype(edge_hash_value)>;

    [[nodiscard]] static EdgeHashMode edge_hash() {
        return edge_hash_value;
    }


    // ------------------------------------------------------------------------
    // Database storage block size
    // ------------------------------------------------------------------------
    // HVDataBase has no library default for this value: it is a constructor
    // argument and must be supplied explicitly.
    //
    // The unit is 16-bit storage words, NOT bytes.
    // 32768 is an example value, not a mandated HighVoronoi default.

    static constexpr std::size_t DatabaseBlockUnits = 32768;


    // ------------------------------------------------------------------------
    // Search backend
    // ------------------------------------------------------------------------
    // KDSearch defaults:
    //     leaf_max_size      = 10
    //     build_thread_count = 1
    //
    // build_thread_count controls construction of the nanoflann KD-tree.  It
    // is independent of MeshThreading and CastThreading.
    //
    // Alternative:
    //     return highvoronoi::geometry::BruteForceSearch{};

    inline static auto search_value =
        highvoronoi::geometry::KDSearch{10, 1};

    using SearchKeyword =
        std::decay_t<decltype(search_value)>;

    [[nodiscard]] static SearchKeyword search() {
        return search_value;
    }


    // ------------------------------------------------------------------------
    // RayCaster
    // ------------------------------------------------------------------------
    // RaycastParameters<> defaults to InRangeRaycast and double.
    // InRangeRaycast is the general path for potentially degenerate vertices.
    // ClassicRaycast is the alternative classical NN-based path.

    inline static auto raycast_parameters_value = [] {
        highvoronoi::RaycastParameters<
            highvoronoi::InRangeRaycast,
            VertexScalar> parameters{};

        // All values below are already the defaults for double and are written
        // out here so they can be seen and edited in one place.
        parameters.variance_tolerance = VertexScalar{1e-15};
        parameters.break_tolerance = VertexScalar{1e-5};
        parameters.boundary_node_tolerance = VertexScalar{1e-7};
        parameters.plane_tolerance = VertexScalar{1e-12};
        parameters.ray_tolerance = VertexScalar{1e-12};
        parameters.rank_tolerance = VertexScalar{1e-12};
        parameters.verification_absolute_tolerance = VertexScalar{1e-10};
        parameters.verification_relative_tolerance = VertexScalar{1e-8};
        parameters.classic_relative_error_trigger = VertexScalar{1e-10};
        parameters.classic_absolute_error_trigger = VertexScalar{1e-8};
        parameters.inrange_t_slack = VertexScalar{1e-7};

        return parameters;
    }();

    using RaycastParameters =
        std::decay_t<decltype(raycast_parameters_value)>;

    [[nodiscard]] static RaycastParameters raycast_parameters() {
        return raycast_parameters_value;
    }


    // ------------------------------------------------------------------------
    // Compute range
    // ------------------------------------------------------------------------
    // std::nullopt means all cells.
    // An Index value is an INCLUSIVE last public cell index.

    [[nodiscard]] static std::optional<Index> range_end() {
        return std::nullopt;
    }


    // ------------------------------------------------------------------------
    // Low-level queue-table implementation
    // ------------------------------------------------------------------------
    // QueueTable is the current default and resolves to QueueHashTable_2.
    // To select the original implementation, replace QueueTable below with
    // ClassicQueueTable in the two parameter aliases that use it.

    template<class Lock, class Generator>
    using DatabaseQueueTable = highvoronoi::QueueTable<Lock, Generator>;

    template<class Lock, class Generator>
    using WorkQueueTable = highvoronoi::QueueTable<Lock, Generator>;


    // ------------------------------------------------------------------------
    // Types derived from the choices above
    // ------------------------------------------------------------------------

    using DatabaseParameters = highvoronoi::DataBaseParams<
        VertexScalar,
        Index,
        HashGenerator,
        DatabaseHashMode,
        DatabaseQueueTable>;

    using QueueParameters = highvoronoi::DataBaseParams<
        VertexScalar,
        Index,
        HashGenerator,
        QueueHashMode,
        WorkQueueTable>;

    using EdgeParameters = highvoronoi::EdgeBufferParams<
        HashGenerator,
        EdgeHashMode,
        highvoronoi::EdgeTable>;

    using Database = highvoronoi::HVDataBase<
        DatabaseLock,
        DatabaseParameters>;

    using Nodes = highvoronoi::VoronoiNodes<
        NodeScalar,
        Dimension,
        Index>;

    using Boundary = highvoronoi::Boundary<
        Dimension,
        NodeScalar,
        Index>;

    using Mesh = highvoronoi::VoronoiMesh<
        NodeScalar,
        Dimension,
        Database>;

    [[nodiscard]] static DatabaseParameters database_parameters() {
        return DatabaseParameters{database_hash()};
    }

    [[nodiscard]] static QueueParameters queue_parameters() {
        return QueueParameters{queue_hash()};
    }

    [[nodiscard]] static EdgeParameters edge_parameters() {
        return EdgeParameters{edge_hash()};
    }
};


// ============================================================================
// 2. THREADING REFERENCE
// ============================================================================
//
// SingleThread
// ------------
//     highvoronoi::SingleThread{}
//
//     thread_count() == 1
//     RWLock == EmptyLock
//
// MultiThread
// -----------
//     highvoronoi::MultiThread{N}
//
//     N must be >= 1.
//     RWLock == ReadWriteLock
//
// ComputeVoronoi has two independent axes:
//
//     MeshThreading = SingleThread, CastThreading = SingleThread
//         one branch, one worker
//
//     MeshThreading = MultiThread{M}, CastThreading = SingleThread
//         M mesh branches, one worker in each branch
//
//     MeshThreading = SingleThread, CastThreading = MultiThread{C}
//         one mesh branch, C workers
//
//     MeshThreading = MultiThread{M}, CastThreading = MultiThread{C}
//         M mesh branches, C workers in each branch
//
// Any parallel combination requires Mesh::Database::LockType == ReadWriteLock.
// ComputeVoronoi checks this at compile time.
//
// MeshThreading creates reordered mesh views internally.  Each branch gets its
// own KD-tree through rebind().  Within one branch worker RayCasters use
// safe_copy(): they share that branch's immutable KD-tree backend and duplicate
// only worker-local / active-extended-node state.


// ============================================================================
// 3. HASH GENERATOR REFERENCE
// ============================================================================
//
// Default generator:
//
//     using H = highvoronoi::FNV64_128HashGenerator;
//
// Built-in alternatives include:
//
//     using H = highvoronoi::Murmur128HashGenerator<>;
//
//     using H = highvoronoi::UInt64HashGenerator<
//         highvoronoi::XXHash64<>,
//         highvoronoi::MurmurHash64<>>;
//
//     using H = highvoronoi::UInt64HashGenerator<
//         highvoronoi::FNV1a64<>,
//         highvoronoi::SipHash64<>>;
//
// Additional fingerprints may be appended without changing the hosted probe
// sequence:
//
//     using H = highvoronoi::ExtendedHashGenerator<
//         highvoronoi::Murmur128HashGenerator<>,
//         highvoronoi::XXHash64<>>;
//
// FNV1a64, XXHash64, MurmurHash64 and SipHash64 have compile-time seed/key
// parameters.  Murmur128HashGenerator has a compile-time uint32 seed.


// ============================================================================
// 4. HASH CONTAINER REFERENCE
// ============================================================================
//
// DirectHash
// ----------
//     highvoronoi::DirectHash{capacity}
//
//     One table.
//     Default capacity: 8.
//
// StaticHash<N>
// -------------
//     highvoronoi::StaticHash<16>{capacity}
//
//     N fixed independent tables.
//     Routing: key[0] % N.
//     Default capacity per subtable: 8.
//     No outer resize lock because the std::array never changes size.
//
// DynamicHash<ResizeLock>
// -----------------------
//     highvoronoi::DynamicHash<>{initial_tables, block_size, capacity}
//
//     Routing: key[0] / block_size.
//     The vector grows when a required table does not yet exist.
//
//     ResizeLock defaults to void.  `void` means: use the same Lock selected
//     for the contained hash tables.  An explicit lock type may be supplied as
//     DynamicHash<SomeLock>.
//
// IMPORTANT CURRENT DEFAULT DETAIL:
//     DynamicHash's member initializers currently read (1, 100, 8), but its
//     constructor default arguments are (1, 1, 1), and the constructor assigns
//     those arguments.  Therefore the EFFECTIVE `DynamicHash{}` values in the
//     current implementation are:
//
//         initial_table_count = 1
//         block_size          = 1
//         hash_capacity       = 1
//
//     Until that interface is cleaned up, specify all three DynamicHash values
//     explicitly in user configuration.
//
// Capacities are initial capacities.  The individual hash tables can rehash.


// ============================================================================
// 5. PERSISTENT DATABASE
// ============================================================================
//
// DataBaseParams template parameters:
//
//     highvoronoi::DataBaseParams<
//         Scalar,              // default: double
//         Index,               // formal default: std::int64_t
//         HashGenerator,       // default: FNV64_128HashGenerator
//         ContainerMode,       // default: DirectHash
//         QueueTableTemplate   // default: QueueTable
//     >
//
// QueueTable resolves to QueueHashTable_2.
// ClassicQueueTable selects the original simpler queue table.
//
// For VoronoiMesh always choose an unsigned Index explicitly.
//
// HVDataBase itself is:
//
//     highvoronoi::HVDataBase<Lock, DataBaseParams>
//
// and is constructed as:
//
//     auto parameters = Config::database_parameters();
//     auto database = std::make_shared<Config::Database>(
//         Config::DatabaseBlockUnits,
//         parameters);
//
// The block length counts uint16 storage units.  There is no constructor
// default for it.
//
// The persistent database hash detects duplicate vertices and duplicate
// infinite-edge facet records.  StaticHash<N> can reduce lock contention here
// because its subtables have independent locks.


// ============================================================================
// 6. COMPUTE QUEUE HASH
// ============================================================================
//
// ComputeVoronoi's QueueParameters have the same DataBaseParams shape because
// the existing QueueHash type adapter is reused, although this queue does not
// store Scalar data in HVDataBase.
//
// The queue hash is local to one SystematicVoronoi and is cleared for every
// cell.  It decides which thread first claims a vertex signature.
//
// Current implementation detail:
//     VoronoiVertexQueue owns one outer QueueLock around the queue state and
//     constructs its internal QueueHash with EmptyLock.  Therefore StaticHash
//     does not currently make claim/pop operations themselves concurrent.


// ============================================================================
// 7. EDGE HASH
// ============================================================================
//
// EdgeBufferParams template parameters:
//
//     highvoronoi::EdgeBufferParams<
//         HashGenerator,       // default: FNV64_128HashGenerator
//         ContainerMode,       // default: DirectHash
//         EdgeTableTemplate    // default: EdgeTable
//     >
//
// The EdgeHash is shared by all worker/prototype EdgeIterators of one
// SystematicVoronoi.  Its lock is selected from complete Voronoi threading:
// ReadWriteLock whenever MeshThreading or CastThreading is parallel.
//
// Therefore StaticHash<N> is useful here: independent edge subtables reduce
// the chance that simultaneous operations contend for the same lock.


// ============================================================================
// 8. VORONOI MESH
// ============================================================================
//
// Current user-facing mesh type:
//
//     highvoronoi::VoronoiMesh<NodeScalar, Dimension, Database>
//
// The Database determines VertexScalar and Index.  NodeScalar is independent,
// although using the same scalar type for nodes and vertices is simplest.
//
// Stored nodes:
//
//     Config::Nodes nodes{node_count};
//     Config::Nodes::Point p;
//     p << 0.0, 0.0, 0.0;
//     nodes.set(Config::Index{0}, p);
//
// For runtime dimension use highvoronoi::Dynamic as Dimension and construct
// VoronoiNodes(length, dimension).
//
// Unbounded domain:
//
//     Config::Boundary boundary{};
//
// Axis-aligned bounded box with NON-PERIODIC faces:
//
//     Config::Boundary::Point dimensions;
//     dimensions << 10.0, 10.0, 10.0;
//
//     Config::Boundary::Point offset;
//     offset << -5.0, -5.0, -5.0;
//
//     auto boundary = Config::Boundary::cuboid(
//         dimensions,
//         offset,
//         std::vector<Config::Index>{});
//
// IMPORTANT:
// Boundary::cuboid(..., periodic_axes = std::nullopt) follows the Julia
// convention and makes ALL axes periodic.  Pass an explicit empty vector for a
// normal non-periodic cuboid.
//
// Periodic-composite construction is intentionally postponed to SerialMesh and
// is not part of this basic VoronoiMesh how-to.
//
// Mesh construction:
//
//     Config::Mesh mesh(
//         std::move(nodes),
//         std::move(boundary),
//         database);


// ============================================================================
// 9. SEARCH TREE
// ============================================================================
//
// KD-tree backend:
//
//     auto tree = highvoronoi::geometry::make_search_tree(
//         mesh,
//         highvoronoi::geometry::KDSearch{10, 1});
//
// KDSearch defaults:
//     leaf_max_size      = 10
//     build_thread_count = 1
//
// Exact reference backend:
//
//     auto tree = highvoronoi::geometry::make_search_tree(
//         mesh,
//         highvoronoi::geometry::BruteForceSearch{});
//
// safe_copy() and rebind() mean different things:
//
//     safe_copy()
//         same mesh branch, same persistent KD backend, independent mutable
//         ExtendedNodes/search state; used for worker RayCasters.
//
//     rebind(new_mesh_view)
//         another mesh/view numbering, therefore another KD index; used by
//         ComputeVoronoi for MeshThreading branches.


// ============================================================================
// 10. RAYCAST PARAMETERS
// ============================================================================
//
// Default template:
//
//     highvoronoi::RaycastParameters<
//         highvoronoi::InRangeRaycast,
//         double>
//
// Methods:
//     InRangeRaycast   default; supports potentially degenerate endpoints.
//     ClassicRaycast   classical iterative nearest-neighbour path.
//
// Defaults for double:
//
//     variance_tolerance               = 1e-15
//     break_tolerance                  = 1e-5
//     boundary_node_tolerance          = 1e-7
//     plane_tolerance                  = 1e-12
//     ray_tolerance                    = 1e-12
//     rank_tolerance                   = 1e-12
//     verification_absolute_tolerance  = 1e-10
//     verification_relative_tolerance  = 1e-8
//     classic_relative_error_trigger   = 1e-10
//     classic_absolute_error_trigger   = 1e-8
//     inrange_t_slack                  = 1e-7
//
// Float-specific defaults differ for:
//
//     variance_tolerance = 5e-10
//     plane_tolerance    = 1e-5
//
// All other listed defaults remain the same.
//
// Construction:
//
//     auto raycaster = highvoronoi::make_raycaster(
//         tree,
//         Config::raycast_parameters());


// ============================================================================
// 11. COMPUTEVORONOI
// ============================================================================
//
// Type:
//
//     highvoronoi::ComputeVoronoi<
//         Mesh,
//         RayCaster,
//         MeshThreading,       // default template type: SingleThread
//         CastThreading,       // default template type: SingleThread
//         QueueParameters,     // default: DataBaseParams<Mesh scalar/index>
//         EdgeParameters       // default: EdgeBufferParams<>
//     >
//
// Constructor argument order:
//
//     ComputeVoronoi(
//         mesh,
//         raycaster_prototype,
//         mesh_threading,
//         cast_threading,
//         range_end,           // std::nullopt = all cells; otherwise inclusive
//         queue_parameters,
//         edge_parameters)
//
// Then:
//
//     compute.compute();
//
// MeshThreading branches and their ReorderedMeshViews are built internally.
// The user supplies only the main VoronoiMesh and one RayCaster prototype.


// ============================================================================
// 12. COMPLETE EXAMPLE USING THE Config BLOCK ABOVE
// ============================================================================
//
// The function accepts an already constructed mesh so that application code
// retains ownership and may inspect/use the mesh after computation.
//
#if 0 // Copy this complete example into an application .cpp when needed.
inline void compute(Config::Mesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        Config::search());

    auto raycaster = highvoronoi::make_raycaster(
        tree,
        Config::raycast_parameters());

    using RayCaster = decltype(raycaster);

    using Compute = highvoronoi::ComputeVoronoi<
        Config::Mesh,
        RayCaster,
        Config::MeshThreading,
        Config::CastThreading,
        Config::QueueParameters,
        Config::EdgeParameters>;

    Compute compute_voronoi(
        mesh,
        raycaster,
        Config::mesh_threading(),
        Config::cast_threading(),
        Config::range_end(),
        Config::queue_parameters(),
        Config::edge_parameters());

    compute_voronoi.compute();
}
#endif


// Example database construction used before Config::Mesh construction:
//
//     auto database_parameters = Config::database_parameters();
//
//     auto database = std::make_shared<Config::Database>(
//         Config::DatabaseBlockUnits,
//         database_parameters);
//
//     Config::Nodes nodes{number_of_nodes};
//     ... fill nodes ...
//
//     Config::Boundary boundary{};       // unbounded
//     // or construct an explicit bounded boundary
//
//     Config::Mesh mesh(
//         std::move(nodes),
//         std::move(boundary),
//         database);
//
//     highvoronoi_howto_config::compute(mesh);

} // namespace highvoronoi_howto_config
