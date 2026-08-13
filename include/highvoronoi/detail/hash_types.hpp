#pragma once

#include <highvoronoi/parameters.hpp>
#include <highvoronoi/detail/hash_table_containers.hpp>

#include <cstddef>
#include <type_traits>

namespace highvoronoi::detail {

// ============================================================================
// QueueHash selection
// ============================================================================

template<
    class ContainerMode,
    class Lock,
    class HashGenerator,
    template<class, class> class QueueTable
>
struct QueueHashSelector;


template<
    class Lock,
    class HashGenerator,
    template<class, class> class QueueTable
>
struct QueueHashSelector<
    DirectHash,
    Lock,
    HashGenerator,
    QueueTable
> {
    using type =
        QueueTable<Lock, HashGenerator>;
};


template<
    std::size_t N,
    class Lock,
    class HashGenerator,
    template<class, class> class QueueTable
>
struct QueueHashSelector<
    StaticHash<N>,
    Lock,
    HashGenerator,
    QueueTable
> {
    using table_type =
        QueueTable<Lock, HashGenerator>;

    using type =
        StaticQueueHashContainer<table_type, N>;
};


template<
    class ResizeLock,
    class Lock,
    class HashGenerator,
    template<class, class> class QueueTable
>
struct QueueHashSelector<
    DynamicHash<ResizeLock>,
    Lock,
    HashGenerator,
    QueueTable
> {
    using table_type =
        QueueTable<Lock, HashGenerator>;

    using resize_lock_type =
        std::conditional_t<
            std::is_void_v<ResizeLock>,
            Lock,
            ResizeLock
        >;

    using type =
        DynamicQueueHashContainer<
            table_type,
            resize_lock_type
        >;
};


// ============================================================================
// EdgeHash selection
// ============================================================================

template<
    class ContainerMode,
    class Lock,
    class HashGenerator,
    template<class, class> class EdgeTable
>
struct EdgeHashSelector;


template<
    class Lock,
    class HashGenerator,
    template<class, class> class EdgeTable
>
struct EdgeHashSelector<
    DirectHash,
    Lock,
    HashGenerator,
    EdgeTable
> {
    using type =
        EdgeTable<Lock, HashGenerator>;
};


template<
    std::size_t N,
    class Lock,
    class HashGenerator,
    template<class, class> class EdgeTable
>
struct EdgeHashSelector<
    StaticHash<N>,
    Lock,
    HashGenerator,
    EdgeTable
> {
    using table_type =
        EdgeTable<Lock, HashGenerator>;

    using type =
        StaticEdgeHashContainer<table_type, N>;
};


template<
    class ResizeLock,
    class Lock,
    class HashGenerator,
    template<class, class> class EdgeTable
>
struct EdgeHashSelector<
    DynamicHash<ResizeLock>,
    Lock,
    HashGenerator,
    EdgeTable
> {
    using table_type =
        EdgeTable<Lock, HashGenerator>;

    using resize_lock_type =
        std::conditional_t<
            std::is_void_v<ResizeLock>,
            Lock,
            ResizeLock
        >;

    using type =
        DynamicEdgeHashContainer<
            table_type,
            resize_lock_type
        >;
};

} // namespace highvoronoi::detail


namespace highvoronoi {

// ============================================================================
// Public concrete hash types
// ============================================================================

template<
    class Lock = EmptyLock,
    class HashGenerator = FNV64_128HashGenerator,
    class ContainerMode = DirectHash,
    template<class, class> class QueueTableTemplate = QueueTable
>
using QueueHash =
    typename detail::QueueHashSelector<
        ContainerMode,
        Lock,
        HashGenerator,
        QueueTableTemplate
    >::type;


template<
    class Lock = EmptyLock,
    class HashGenerator = FNV64_128HashGenerator,
    class ContainerMode = DirectHash,
    template<class, class> class EdgeTableTemplate = EdgeTable
>
using EdgeHash =
    typename detail::EdgeHashSelector<
        ContainerMode,
        Lock,
        HashGenerator,
        EdgeTableTemplate
    >::type;

} // namespace highvoronoi


namespace highvoronoi::detail {

// ============================================================================
// Lock + parameter-object adapters
// ============================================================================

/**
 * @brief Construct the database QueueHash type from a lock and DataBaseParams.
 *
 * @code{.cpp}
 * using Params = highvoronoi::DataBaseParams<>;
 *
 * using Queue =
 *     highvoronoi::detail::QueueHashFromParams_t<
 *         highvoronoi::ReadWriteLock,
 *         Params
 *     >;
 * @endcode
 */
template<
    class Lock,
    class Params
>
struct QueueHashFromParams {
    using type =
        highvoronoi::QueueHash<
            Lock,
            typename Params::HashGenerator,
            typename Params::ContainerMode,
            Params::template QueueTable
        >;
};


template<
    class Lock,
    class Params
>
using QueueHashFromParams_t =
    typename QueueHashFromParams<
        Lock,
        Params
    >::type;


/**
 * @brief Construct the EdgeHash type from a lock and EdgeBufferParams.
 *
 * @code{.cpp}
 * using Params = highvoronoi::EdgeBufferParams<>;
 *
 * using Edges =
 *     highvoronoi::detail::EdgeHashFromParams_t<
 *         highvoronoi::ReadWriteLock,
 *         Params
 *     >;
 * @endcode
 */
template<
    class Lock,
    class Params
>
struct EdgeHashFromParams {
    using type =
        highvoronoi::EdgeHash<
            Lock,
            typename Params::HashGenerator,
            typename Params::ContainerMode,
            Params::template EdgeTable
        >;
};


template<
    class Lock,
    class Params
>
using EdgeHashFromParams_t =
    typename EdgeHashFromParams<
        Lock,
        Params
    >::type;

} // namespace highvoronoi::detail


// ============================================================================
// Make Hash
// ============================================================================


namespace highvoronoi::detail {

template<class HashType>
[[nodiscard]] HashType make_hash_object(
    const DirectHash& params)
{
    return HashType(params.hash_capacity);
}


template<class HashType, std::size_t N>
[[nodiscard]] HashType make_hash_object(
    const StaticHash<N>& params)
{
    return HashType(params.hash_capacity);
}


template<class HashType, class ResizeLock>
[[nodiscard]] HashType make_hash_object(
    const DynamicHash<ResizeLock>& params)
{
    return HashType(
        params.initial_table_count,
        params.block_size,
        params.hash_capacity
    );
}

} // namespace highvoronoi::detail


// ============================================================================
// Make Queue
// ============================================================================

namespace highvoronoi::detail {

template<class Lock, class Params>
[[nodiscard]] QueueHashFromParams_t<Lock, Params>
make_queue_hash(const Params& params)
{
    using Queue =
        QueueHashFromParams_t<Lock, Params>;

    return make_hash_object<Queue>(
        params.queue_hash
    );
}

} // namespace highvoronoi::detail

// ============================================================================
// Make Edge
// ============================================================================

namespace highvoronoi::detail {

template<class Lock, class Params>
[[nodiscard]] EdgeHashFromParams_t<Lock, Params>
make_edge_hash(const Params& params)
{
    using Edge =
        EdgeHashFromParams_t<Lock, Params>;

    return make_hash_object<Edge>(
        params.edge_hash
    );
}

} // namespace highvoronoi::detail
