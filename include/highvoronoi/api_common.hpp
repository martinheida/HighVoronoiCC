#pragma once

/**
 * @file api_common.hpp
 * @brief Shared Level-1/Level-2 convenience configuration.
 *
 * Level 1 intentionally hides database, lock, hash, search-tree and ray-caster
 * plumbing. Level 2 exposes the ray-cast method and hash policy while still
 * constructing the same Level-3 engine classes internally.
 */

#include <highvoronoi/parameters.hpp>
#include <highvoronoi/core/boundary.hpp>
#include <highvoronoi/core/voronoi_nodes.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

namespace api_detail {

template <class T>
inline constexpr bool is_supported_threading_v =
    std::is_same_v<std::decay_t<T>, SingleThread> ||
    std::is_same_v<std::decay_t<T>, MultiThread>;

template <class Threading>
struct DefaultPersistentHashContainer;

template <>
struct DefaultPersistentHashContainer<SingleThread> {
    using type = DirectHash;
};

template <>
struct DefaultPersistentHashContainer<MultiThread> {
    using type = StaticHash<16>;
};

template <class Threading>
using DefaultPersistentHashContainerT =
    typename DefaultPersistentHashContainer<Threading>::type;

template <class Container>
struct HashContainerFactory {
    [[nodiscard]] static Container make(std::size_t) {
        return Container{};
    }
};

template <>
struct HashContainerFactory<DirectHash> {
    [[nodiscard]] static DirectHash make(std::size_t total_capacity) {
        return DirectHash{(std::max)(std::size_t{8}, total_capacity)};
    }
};

template <std::size_t N>
struct HashContainerFactory<StaticHash<N>> {
    [[nodiscard]] static StaticHash<N> make(std::size_t total_capacity) {
        const std::size_t per_table = (std::max)(
            std::size_t{8},
            (total_capacity + N - std::size_t{1}) / N);
        return StaticHash<N>{per_table};
    }
};

[[nodiscard]] inline std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* message) {
    if (right != 0 && left > (std::numeric_limits<std::size_t>::max)() / right) {
        throw std::length_error(message);
    }
    return left * right;
}

[[nodiscard]] inline std::size_t recommended_hash_capacity(
    std::size_t point_count,
    std::size_t dimension) {
    // This is deliberately only an initial-capacity heuristic. All current
    // queue/edge tables grow when required. The estimate avoids starting large
    // jobs at the tiny infrastructure default without pretending to predict
    // the number of high-dimensional Voronoi vertices exactly.
    const std::size_t geometry_factor = (std::max)(
        std::size_t{16},
        checked_product(
            (std::max)(dimension, std::size_t{1}),
            std::size_t{8},
            "HighVoronoi API hash-capacity heuristic overflow."));
    return (std::max)(
        std::size_t{1024},
        checked_product(
            (std::max)(point_count, std::size_t{1}),
            geometry_factor,
            "HighVoronoi API hash-capacity heuristic overflow."));
}

[[nodiscard]] inline std::size_t recommended_queue_capacity(
    std::size_t dimension) {
    return (std::max)(std::size_t{256}, dimension * std::size_t{64});
}

template <class Threading>
[[nodiscard]] std::size_t resolve_thread_count(
    std::size_t preferred,
    std::optional<std::size_t> override_count = std::nullopt) {
    static_assert(
        is_supported_threading_v<Threading>,
        "Level-1/2 API supports SingleThread or MultiThread. Use Level 3 for custom threading policies.");

    if constexpr (std::is_same_v<Threading, SingleThread>) {
        (void)preferred;
        (void)override_count;
        return std::size_t{1};
    } else {
        const std::size_t count = override_count.value_or(preferred);
        if (count == 0) {
            throw std::invalid_argument(
                "MultiThread Level-1/2 API requires at least one thread.");
        }
        return count;
    }
}

template <class Threading>
[[nodiscard]] Threading make_threading(std::size_t count) {
    if constexpr (std::is_same_v<Threading, SingleThread>) {
        (void)count;
        return SingleThread{};
    } else {
        return MultiThread{count};
    }
}

template <int Dim>
[[nodiscard]] std::size_t resolve_dimension(std::size_t requested) {
    if constexpr (Dim == Dynamic) {
        if (requested == 0) {
            throw std::invalid_argument(
                "Dynamic Level-1/2 mesh construction requires dimension > 0.");
        }
        return requested;
    } else {
        if (requested != 0 && requested != static_cast<std::size_t>(Dim)) {
            throw std::invalid_argument(
                "Runtime dimension disagrees with the fixed template dimension.");
        }
        return static_cast<std::size_t>(Dim);
    }
}

template <class Index>
[[nodiscard]] Index checked_index(std::size_t value, const char* message) {
    if (value > static_cast<std::size_t>((std::numeric_limits<Index>::max)())) {
        throw std::length_error(message);
    }
    return static_cast<Index>(value);
}

template <int Dim, class Index>
[[nodiscard]] VoronoiNodes<double, Dim, Index> make_nodes(
    const double* points,
    std::size_t point_count,
    std::size_t runtime_dimension) {
    if (point_count != 0 && points == nullptr) {
        throw std::invalid_argument(
            "Level-1/2 mesh construction received a null points pointer.");
    }

    const Index count = checked_index<Index>(
        point_count,
        "Point count does not fit into the configured mesh index type.");
    const Index dimension = checked_index<Index>(
        runtime_dimension,
        "Dimension does not fit into the configured mesh index type.");

    VoronoiNodes<double, Dim, Index> nodes = [&] {
        if constexpr (Dim == Dynamic) {
            return VoronoiNodes<double, Dim, Index>(count, dimension);
        } else {
            (void)dimension;
            return VoronoiNodes<double, Dim, Index>(count);
        }
    }();

    for (Index node = Index{0}; node < count; ++node) {
        nodes.set(
            node,
            points + static_cast<std::size_t>(node) * runtime_dimension);
    }
    return nodes;
}

template <int Dim, class SourceScalar, class SourceIndex, class TargetIndex>
[[nodiscard]] Boundary<Dim, double, TargetIndex> convert_boundary(
    const Boundary<Dim, SourceScalar, SourceIndex>& source,
    std::size_t runtime_dimension) {
    using TargetBoundary = Boundary<Dim, double, TargetIndex>;
    using TargetPlane = typename TargetBoundary::PlaneType;

    if (!source.empty() && source.dimension() != runtime_dimension) {
        throw std::invalid_argument(
            "Boundary dimension does not match the requested mesh dimension.");
    }

    std::vector<TargetPlane> planes;
    planes.reserve(static_cast<std::size_t>(source.size()));
    for (const auto& plane : source.planes()) {
        std::optional<TargetIndex> partner;
        if (const auto source_partner = plane.periodic_partner()) {
            partner = checked_index<TargetIndex>(
                static_cast<std::size_t>(*source_partner),
                "Boundary plane index does not fit into uint32_t.");
        }
        planes.emplace_back(
            plane.base(),
            plane.normal(),
            plane.condition(),
            partner);
    }
    return TargetBoundary(std::move(planes), source.convex());
}



} // namespace api_detail

/**
 * @brief Level-2 construction policy shared by ordinary and HighVoronoi APIs.
 *
 * Level 1 creates this policy automatically. Level 2 lets users choose the
 * ray-cast method, hash generator and persistent/edge hash containers without
 * manually spelling database, lock, search-tree, RayCaster or ComputeVoronoi
 * types. Level 3 remains the existing explicit API.
 */
template <
    class ThreadingT = SingleThread,
    class RaycastMethodT = CombinedRaycast,
    class HashGeneratorT = FNV64_128HashGenerator,
    class DatabaseHashContainerT =
        api_detail::DefaultPersistentHashContainerT<ThreadingT>,
    class EdgeHashContainerT =
        api_detail::DefaultPersistentHashContainerT<ThreadingT>>
struct VoronoiConfig {
    static_assert(
        api_detail::is_supported_threading_v<ThreadingT>,
        "VoronoiConfig Level 2 supports SingleThread or MultiThread. Use Level 3 for custom threading policies.");

    using Threading = ThreadingT;
    using RaycastMethod = RaycastMethodT;
    using HashGenerator = HashGeneratorT;
    using DatabaseHashContainer = DatabaseHashContainerT;
    using EdgeHashContainer = EdgeHashContainerT;

    using RayParameters = RaycastParameters<RaycastMethod, double>;

    std::size_t thread_count = 1;
    std::size_t database_block_units = 65536;

    geometry::KDSearch search{8, 1};
    RayParameters raycast{};
    DatabaseHashContainer database_hash{};
    DirectHash queue_hash{};
    EdgeHashContainer edge_hash{};

    bool verbose = false;

    VoronoiConfig() = default;

    explicit VoronoiConfig(Threading threading)
        : thread_count(threading.thread_count()) {}
};

template <class T>
struct IsVoronoiConfig : std::false_type {};

template <class... Ts>
struct IsVoronoiConfig<VoronoiConfig<Ts...>> : std::true_type {};

template <class T>
inline constexpr bool IsVoronoiConfigV = IsVoronoiConfig<std::decay_t<T>>::value;

/** Build the automatically sized Level-1 configuration explicitly. */
template <
    class Threading = SingleThread,
    class RaycastMethod = CombinedRaycast,
    class HashGenerator = FNV64_128HashGenerator,
    class DatabaseHashContainer =
        api_detail::DefaultPersistentHashContainerT<Threading>,
    class EdgeHashContainer =
        api_detail::DefaultPersistentHashContainerT<Threading>>
[[nodiscard]] VoronoiConfig<
    Threading,
    RaycastMethod,
    HashGenerator,
    DatabaseHashContainer,
    EdgeHashContainer>
make_voronoi_config(
    std::size_t point_count,
    std::size_t dimension,
    Threading threading = Threading{}) {
    VoronoiConfig<
        Threading,
        RaycastMethod,
        HashGenerator,
        DatabaseHashContainer,
        EdgeHashContainer> config(threading);

    const std::size_t persistent_capacity =
        api_detail::recommended_hash_capacity(point_count, dimension);
    config.database_hash =
        api_detail::HashContainerFactory<DatabaseHashContainer>::make(
            persistent_capacity);
    config.edge_hash =
        api_detail::HashContainerFactory<EdgeHashContainer>::make(
            persistent_capacity);
    config.queue_hash = DirectHash{
        api_detail::recommended_queue_capacity(dimension)};
    return config;
}

} // namespace highvoronoi
