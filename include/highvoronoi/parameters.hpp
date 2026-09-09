
#pragma once
/**
 * @file parameters.hpp
 * @brief Configuration parameters for HighVoronoi data structures.
 *
 * @section parameter_configuration Parameter configuration
 *
 * `DataBaseParams` and `EdgeBufferParams` combine:
 *
 * 1. compile-time type selection, and
 * 2. runtime values needed to construct the selected hash structures.
 *
 * ---------------------------------------------------------------------------
 * DataBaseParams
 * ---------------------------------------------------------------------------
 *
 * @code{.cpp}
 * DataBaseParams<
 *     Scalar,
 *     Index,
 *     HashGenerator,
 *     ContainerMode,
 *     QueueTable
 * >
 * @endcode
 *
 * - `Scalar`
 *      Floating-point type used by the geometric database, for example
 *      `float` or `double`. It does not affect the hash-table structure.
 *
 * - `Index`
 *      Integer type used for indices, for example `std::uint32_t` or
 *      `std::uint64_t`. It does not affect the hash-table structure.
 *
 * - `HashGenerator`
 *      Converts a key into the fingerprint and probe sequence used by every
 *      contained hash table.
 *
 * - `QueueTable`
 *      Selects the implementation of one individual queue hash table.
 *
 *      `QueueTable` selects the newer implementation with separate reader,
 *      writer, and rehash synchronization.
 *
 *      `ClassicQueueTable` selects the original, simpler implementation.
 *
 * - `ContainerMode`
 *      Determines whether one table or several independent tables are created.
 *
 * ---------------------------------------------------------------------------
 * Container modes
 * ---------------------------------------------------------------------------
 *
 * Direct table:
 *
 * @code{.cpp}
 * DirectHash{hash_capacity}
 * @endcode
 *
 * Creates one hash table. Every key is stored in this table.
 *
 * `hash_capacity` is the initial number of slots in that table.
 *
 *
 * Static container:
 *
 * @code{.cpp}
 * StaticHash<N>{hash_capacity}
 * @endcode
 *
 * Creates a fixed `std::array` containing `N` independent hash tables.
 *
 * The selected table is:
 *
 * @code{.cpp}
 * table_index = key[0] % N;
 * @endcode
 *
 * Every contained table starts with `hash_capacity` slots.
 *
 * A larger `N` generally reduces contention and the number of entries per
 * table, but increases the number of allocated table objects.
 *
 *
 * Dynamic container:
 *
 * @code{.cpp}
 * DynamicHash<>{
 *     initial_table_count,
 *     block_size,
 *     hash_capacity
 * }
 * @endcode
 *
 * Creates a dynamically growing vector of independent hash tables.
 *
 * The selected table is:
 *
 * @code{.cpp}
 * table_index = key[0] / block_size;
 * @endcode
 *
 * - `initial_table_count`
 *      Number of hash tables created initially.
 *
 * - `block_size`
 *      Number of consecutive `key[0]` values assigned to one table.
 *
 *      Example with `block_size == 1000`:
 *
 *      - `key[0] == 0 ... 999`       -> table 0
 *      - `key[0] == 1000 ... 1999`  -> table 1
 *      - `key[0] == 2000 ... 2999`  -> table 2
 *
 * - `hash_capacity`
 *      Initial number of slots in every newly created hash table.
 *
 * The vector grows automatically when the required table does not yet exist.
 * A smaller `block_size` creates more tables and usually reduces contention.
 * A larger `block_size` creates fewer tables but places more keys in each one.
 *
 * ---------------------------------------------------------------------------
 * EdgeBufferParams
 * ---------------------------------------------------------------------------
 *
 * @code{.cpp}
 * EdgeBufferParams<
 *     HashGenerator,
 *     ContainerMode,
 *     EdgeTable
 * >
 * @endcode
 *
 * This has the same hash-generator and container semantics as
 * `DataBaseParams`, but configures edge hash tables instead of queue hash
 * tables. It has no `Scalar` or `Index` parameter.
 *
 * ---------------------------------------------------------------------------
 * Threading
 * ---------------------------------------------------------------------------
 *
 * The lock type is selected separately through the threading policy:
 *
 * - `SingleThread` uses `EmptyLock`.
 * - `MultiThread` uses `ReadWriteLock`.
 *
 * The parameter adapters combine the selected lock with the parameter object
 * and produce the final queue or edge type.
 *
 * ---------------------------------------------------------------------------
 * Complete example
 * ---------------------------------------------------------------------------
 *
 * @code{.cpp}
 * using Generator = ExtendedHashGenerator<
 *     Murmur128HashGenerator<>,
 *     XXHash64<>
 * >;
 *
 * using DatabaseParameters = DataBaseParams<
 *     double,
 *     std::uint32_t,
 *     Generator,
 *     StaticHash<8>,
 *     QueueTable
 * >;
 *
 * DatabaseParameters database_parameters{
 *     StaticHash<8>{
 *         1024
 *     }
 * };
 *
 * using EdgeParameters = EdgeBufferParams<
 *     Generator,
 *     DynamicHash<>,
 *     EdgeTable
 * >;
 *
 * EdgeParameters edge_parameters{
 *     DynamicHash<>{
 *         4,
 *         10000,
 *         256
 *     }
 * };
 *
 * MultiThread threading{8};
 * @endcode
 *
 * This configuration produces:
 *
 * - geometric values stored as `double`,
 * - indices stored as `std::uint32_t`,
 * - eight fixed queue hash tables,
 * - queue routing through `key[0] % 8`,
 * - 1024 initial slots in each queue table,
 * - a dynamic vector of edge hash tables,
 * - initially four edge tables,
 * - edge routing through `key[0] / 10000`,
 * - 256 initial slots in every edge table,
 * - Murmur128 plus XXHash fingerprints,
 * - read/write locking selected by `MultiThread`.
 *
 * @code{.cpp}
 * using Params = DataBaseParams<
 *     double,
 *     std::uint32_t,
 *     Generator,
 *     StaticHash<8>
 * >;
 * @endcode
 */

#include <highvoronoi/storage/hash/edge_hash_table.hpp>
#include <highvoronoi/storage/hash/hash_functions.hpp>
#include <highvoronoi/storage/hash/hash_generators.hpp>
#include <highvoronoi/core/detail/locks.hpp>
#include <highvoronoi/storage/hash/queue_hash_table.hpp>
#include <highvoronoi/storage/hash/queue_hash_table_2.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <utility>

namespace highvoronoi {

// ============================================================================
// Threading policies and locks
// ============================================================================

/**
 * @brief Single-thread execution policy.
 *
 * Uses EmptyLock internally and reports one worker thread.
 */
using SingleThread = detail::SingleThread;

/**
 * @brief Multi-thread execution policy.
 *
 * Uses ReadWriteLock internally. The thread count is passed to the constructor.
 *
 * @code{.cpp}
 * highvoronoi::MultiThread threading(8);
 * @endcode
 */
using MultiThread = detail::MultiThread;

/** @brief Public alias for the no-op lock. */
using EmptyLock = detail::EmptyLock;

/** @brief Public alias for the read/write lock. */
using ReadWriteLock = detail::ReadWriteLock;


// ============================================================================
// Hash functions
// ============================================================================

/**
 * @brief FNV-1a 64-bit hash function.
 *
 * Prime and OffsetBasis can be replaced at compile time.
 */
template<
    std::uint64_t Prime = 1099511628211ULL,
    std::uint64_t OffsetBasis = 14695981039346656037ULL
>
using FNV1a64 = detail::FNV1a64<Prime, OffsetBasis>;

/** @brief XXHash64 with a compile-time seed. */
template<std::uint64_t Seed = 0>
using XXHash64 = detail::XXHash64<Seed>;

/** @brief MurmurHash64A with a compile-time seed. */
template<std::uint64_t Seed = 0>
using MurmurHash64 = detail::MurmurHash64<Seed>;

/** @brief MurmurHash3 x64 128-bit hash returning two uint64 values. */
template<std::uint32_t Seed = 0>
using MurmurHash_x64_128 = detail::MurmurHash_x64_128<Seed>;

/** @brief SipHash-2-4 with compile-time key values. */
template<
    std::uint64_t K0 = 0,
    std::uint64_t K1 = 0
>
using SipHash64 = detail::SipHash64<K0, K1>;


// ============================================================================
// Hash generators
// ============================================================================

/**
 * @brief Hash generator constructed from at least two uint64 hash functions.
 *
 * The first hash determines the initial table index. The second hash determines
 * the probing step. Additional hashes are stored only for fingerprint
 * comparison.
 *
 * @code{.cpp}
 * using Generator = highvoronoi::UInt64HashGenerator<
 *     highvoronoi::XXHash64<>,
 *     highvoronoi::MurmurHash64<>,
 *     highvoronoi::FNV1a64<>
 * >;
 * @endcode
 */
template<
    class FirstHash,
    class SecondHash,
    class... MoreHashes
>
using UInt64HashGenerator =
    detail::UInt64HashGenerator<
        FirstHash,
        SecondHash,
        MoreHashes...
    >;

/**
 * @brief Extend an existing hash generator with additional fingerprints.
 *
 * The hosted generator continues to determine the probing sequence. Appendix
 * hashes are used only for equality comparison.
 *
 * @code{.cpp}
 * using Generator = highvoronoi::ExtendedHashGenerator<
 *     highvoronoi::Murmur128HashGenerator<>,
 *     highvoronoi::XXHash64<>
 * >;
 * @endcode
 */
template<
    class Generator,
    class... AppendixHashes
>
using ExtendedHashGenerator =
    detail::ExtendedHashGenerator<
        Generator,
        AppendixHashes...
    >;

/**
 * @brief Hash generator using MurmurHash3 x64 128.
 *
 * The two returned uint64 values determine the initial index and probing step.
 */
template<std::uint32_t Seed = 0>
using Murmur128HashGenerator =
    detail::Murmur128HashGenerator<Seed>;

/**
 * @brief Original HighVoronoi generator combining 128-bit and 64-bit FNV.
 */
using FNV64_128HashGenerator =
    detail::FNV64_128HashGenerator;


// ============================================================================
// Container modes
// ============================================================================

/**
 * @brief Use one hash table directly.
 *
 * @param hash_capacity Initial capacity of the hash table.
 */
struct DirectHash {
    std::size_t hash_capacity{8};

    explicit DirectHash(
        std::size_t capacity = 8
    ) noexcept
        : hash_capacity(capacity)
    {}
};


/**
 * @brief Use a fixed array of N independent hash tables.
 *
 * Keys are routed by:
 *
 *     key[0] % N
 *
 * @param hash_capacity Initial capacity of every contained hash table.
 */
template<std::size_t N>
struct StaticHash {
    static_assert(N > 0, "StaticHash<N> requires N > 0");

    std::size_t hash_capacity{8};

    explicit StaticHash(
        std::size_t capacity = 8
    ) noexcept
        : hash_capacity(capacity)
    {}
};


/**
 * @brief Use a dynamically growing vector of independent hash tables.
 *
 * Keys are routed by:
 *
 *     key[0] / block_size
 *
 * @param initial_table_count Initial number of contained hash tables.
 * @param block_size          Number of first-key values routed to one table.
 * @param hash_capacity       Initial capacity of every contained hash table.
 */
template<class ResizeLock = void>
struct DynamicHash {
    std::size_t initial_table_count{1};
    std::uint64_t block_size{100};
    std::size_t hash_capacity{8};

    DynamicHash(
        std::size_t initial_count = 1,
        std::uint64_t block = 1,
        std::size_t capacity = 1
    )
        : initial_table_count(initial_count),
          block_size(block),
          hash_capacity(capacity)
    {
        if (block_size == 0) {
            throw std::invalid_argument(
                "DynamicHash block_size must be greater than zero"
            );
        }
    }
};

// ============================================================================
// Selectable hash-table implementations
// ============================================================================

/**
 * @brief Default queue-table implementation.
 *
 * Resolves to QueueHashTable_2.
 */
template<
    class Lock,
    class HashGenerator
>
using QueueTable =
    detail::QueueHashTable_2<
        Lock,
        HashGenerator
    >;

/**
 * @brief Original, simpler queue-table implementation.
 */
template<
    class Lock,
    class HashGenerator
>
using ClassicQueueTable =
    detail::QueueHashTable<
        Lock,
        HashGenerator
    >;

/**
 * @brief Default edge-table implementation.
 */
template<
    class Lock,
    class HashGenerator
>
using EdgeTable =
    detail::EdgeHashTable<
        Lock,
        HashGenerator
    >;


// ============================================================================
// Database parameters
// ============================================================================

/**
 * @brief Compile-time configuration of the main HighVoronoi database.
 *
 * @tparam ScalarType
 *     Floating-point type used for geometric values.
 *
 * @tparam IndexType
 *     Integral type used for indices.
 *
 * @tparam HashGeneratorType
 *     Hash generator used by database queue hashes.
 *
 * @tparam ContainerModeType
 *     DirectHash, StaticHash<N>, or DynamicHash<>.
 *
 * @tparam QueueTableTemplate
 *     Concrete queue-table implementation. The default is QueueTable, which
 *     resolves to QueueHashTable_2.
 *
 * Default configuration (`double`, `std::uint32_t`, original FNV64/128
 * generator, one DirectHash table, current QueueTable implementation):
 *
 * @code{.cpp}
 * using Params = highvoronoi::DataBaseParams<>;
 * @endcode
 *
 * Static container:
 *
 * @code{.cpp}
 * using Params = highvoronoi::DataBaseParams<
 *     float,
 *     std::uint32_t,
 *     highvoronoi::Murmur128HashGenerator<>,
 *     highvoronoi::StaticHash<8>
 * >;
 * @endcode
 *
 * Custom generator and dynamic container:
 *
 * @code{.cpp}
 * using Generator = highvoronoi::ExtendedHashGenerator<
 *     highvoronoi::Murmur128HashGenerator<>,
 *     highvoronoi::XXHash64<>
 * >;
 *
 * using Params = highvoronoi::DataBaseParams<
 *     double,
 *     std::uint64_t,
 *     Generator,
 *     highvoronoi::DynamicHash<>,
 *     highvoronoi::QueueTable
 * >;
 * @endcode
 */
template<
    class ScalarType = double,
    class IndexType = std::uint32_t,
    class HashGeneratorType = FNV64_128HashGenerator,
    class ContainerModeType = DirectHash,
    template<class, class> class QueueTableTemplate = QueueTable
>
struct DataBaseParams {
    static_assert(
        std::is_floating_point_v<ScalarType>,
        "DataBaseParams ScalarType must be floating-point"
    );

    static_assert(
        std::is_integral_v<IndexType> &&
        !std::is_same_v<std::remove_cv_t<IndexType>, bool>,
        "DataBaseParams IndexType must be a non-bool integer"
    );

    using Scalar = ScalarType;
    using Index = IndexType;
    using HashGenerator = HashGeneratorType;
    using ContainerMode = ContainerModeType;

    template<class Lock, class Generator>
    using QueueTable =
        QueueTableTemplate<Lock, Generator>;

    ContainerMode queue_hash{};

    explicit DataBaseParams(
        ContainerMode mode = {}
    )
        : queue_hash(std::move(mode))
    {}
};


// ============================================================================
// Edge-buffer parameters
// ============================================================================

/**
 * @brief Compile-time configuration of the temporary edge buffer.
 *
 * @tparam HashGeneratorType
 *     Hash generator used by edge hashes.
 *
 * @tparam ContainerModeType
 *     DirectHash, StaticHash<N>, or DynamicHash<>.
 *
 * @tparam EdgeTableTemplate
 *     Concrete edge-table implementation.
 *
 * Default configuration:
 *
 * @code{.cpp}
 * using Params = highvoronoi::EdgeBufferParams<>;
 * @endcode
 *
 * Static edge-buffer container:
 *
 * @code{.cpp}
 * using Params = highvoronoi::EdgeBufferParams<
 *     highvoronoi::Murmur128HashGenerator<>,
 *     highvoronoi::StaticHash<16>
 * >;
 * @endcode
 */
template<
    class HashGeneratorType = FNV64_128HashGenerator,
    class ContainerModeType = DirectHash,
    template<class, class> class EdgeTableTemplate = EdgeTable
>
struct EdgeBufferParams {
    using HashGenerator = HashGeneratorType;
    using ContainerMode = ContainerModeType;

    template<class Lock, class Generator>
    using EdgeTable =
        EdgeTableTemplate<Lock, Generator>;

    ContainerMode edge_hash{};

    explicit EdgeBufferParams(
        ContainerMode mode = {}
    )
        : edge_hash(std::move(mode))
    {}
};

} // namespace highvoronoi



// ============================================================================
// Ray-caster selection and tolerances
// ============================================================================

namespace highvoronoi {

/** @brief Select the classical nearest-neighbour RayCaster. */
struct ClassicRaycast {};

/** @brief Select the in-range RayCaster for potentially degenerate vertices. */
struct InRangeRaycast {};

/**
 * @brief Select the Julia-faithful fused KD-tree / raycast algorithm.
 *
 * This corresponds to HighVoronoi.jl `Raycast_Combined`. The specialised
 * nanoflann traversal may move the current ray endpoint and restart from the
 * KD-tree root while preserving caller-owned visited-subtree state.
 */
struct CombinedRaycast {};

/**
 * @brief Common ray-caster parameters.
 *
 * The public default method is CombinedRaycast. ClassicRaycast and
 * InRangeRaycast remain explicit alternatives. CombinedRaycast itself uses
 * its robust fallback policy by default.
 */
template <class MethodT = CombinedRaycast, class ScalarT = double>
struct RaycastParameters {
    static_assert(
        std::is_floating_point_v<ScalarT>,
        "RaycastParameters ScalarT must be floating-point.");

    using Method = MethodT;
    using Scalar = ScalarT;

    Method method{};

    Scalar variance_tolerance =
        std::is_same_v<Scalar, float> ? Scalar{5e-10} : Scalar{1e-15};
    Scalar break_tolerance = Scalar{1e-5};
    Scalar boundary_node_tolerance = Scalar{1e-7};
    Scalar plane_tolerance =
        std::is_same_v<Scalar, float> ? Scalar{1e-5} : Scalar{1e-12};
    Scalar ray_tolerance = Scalar{1e-12};

    // Rank/verification controls for verify_vertex().
    Scalar rank_tolerance = Scalar{1e-12};
    Scalar verification_absolute_tolerance = Scalar{1e-10};
    Scalar verification_relative_tolerance = Scalar{1e-8};

    // Julia uses semantic verify_vertex() for descent bootstrap only.  Keep
    // the stronger per-WalkRay verification available as an opt-in diagnostic,
    // but do not pay for it in the normal construction hot path.
    bool verify_walk_vertices = false;

    // Vertex corrector:
    // - pivot ratio below vertex_condition_tolerance -> direct Float128 fallback;
    // - otherwise reuse one double ColPivHouseholderQR factorization until
    //   |delta| / |r_initial-p_0| is small enough;
    // - failure to converge within the bounded iteration count -> Float128.
    Scalar vertex_condition_tolerance =
        std::is_same_v<Scalar, float> ? Scalar{1e-4} : Scalar{1e-8};
    Scalar vertex_correction_relative_tolerance =
        std::is_same_v<Scalar, float> ? Scalar{1e-5} : Scalar{1e-12};
    std::size_t vertex_correction_max_iterations = 3;

    // Julia Raycast_Original switches to full correction above these errors.
    Scalar classic_relative_error_trigger = Scalar{1e-10};
    Scalar classic_absolute_error_trigger = Scalar{1e-8};

    // Julia Raycast_Non_General uses upper_t += 10E-8.
    Scalar inrange_t_slack = Scalar{1e-7};
};

} // namespace highvoronoi

