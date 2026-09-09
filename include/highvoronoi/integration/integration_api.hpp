#pragma once

/**
 * @file integration_api.hpp
 * @brief Level-1 and Level-2 facade for persistent Voronoi integration.
 *
 * Level 1 keeps one persistent integral owner and uses FastPolygon by default:
 *
 *   auto integral = voronoi_integral(mesh);
 *   integrate(integral);
 *
 * A scalar function can be attached directly:
 *
 *   auto integral = voronoi_integral(mesh, function);
 *   integrate(integral);
 *
 * Level 2 keeps the same persistent owner but exposes result-storage options,
 * algorithm choice and the existing serial/parallel execution policies.
 * Level 3 remains the native VoronoiIntegral / algorithm / Integrator API.
 */

#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/integration/integrator.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>
#include <highvoronoi/integration/monte_carlo_integrator.hpp>
#include <highvoronoi/integration/heuristic_mc_integrator.hpp>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace highvoronoi {

/**
 * @brief Persistent-storage configuration used by the Level-2 integration facade.
 *
 * The facade default is geometry-only storage: volume and interface area are
 * enabled, function-integral arrays are disabled. This differs deliberately
 * from the low-level IntegralDataOptions aggregate default.
 */
struct IntegrationConfig {
    IntegralDataOptions data_options{true, true, false, false};
    std::size_t integral_components = 0;
    std::size_t area_database_unit_length = std::size_t{65536};
    std::size_t integral_database_unit_length = std::size_t{65536};
};

/** Level-2 tag selecting the cached deterministic FastPolygon algorithm. */
struct FastPolygon final {};

/** Level-2 tag selecting the deterministic reference Polygon algorithm. */
struct Polygon final {};

/** Level-2 tag selecting Monte-Carlo integration with explicit options. */
struct MonteCarlo final {
    MonteCarloOptions options{};
};

/**
 * Level-2 tag selecting Monte-Carlo geometry plus heuristic function quadrature.
 * A function-bearing facade integral is required.
 */
struct HeuristicMC final {
    MonteCarloOptions options{};
};

namespace api_detail {

struct NoIntegrationFunction final {};

template <class T, class = void>
struct HasLevel3Mesh : std::false_type {};

template <class T>
struct HasLevel3Mesh<
    T,
    std::void_t<decltype(std::declval<T&>().level3_mesh())>>
    : std::true_type {};

template <class MeshLike>
decltype(auto) integration_native_mesh(MeshLike& mesh) {
    if constexpr (HasLevel3Mesh<MeshLike>::value) {
        return (mesh.level3_mesh());
    } else {
        return (mesh);
    }
}

template <class Function, class Point, class = void>
struct IsScalarIntegrationFunction : std::false_type {};

template <class Function, class Point>
struct IsScalarIntegrationFunction<
    Function,
    Point,
    std::void_t<std::invoke_result_t<const Function&, const Point&>>>
    : std::bool_constant<std::is_arithmetic_v<
          std::decay_t<std::invoke_result_t<const Function&, const Point&>>>> {};

template <class T>
struct IsIntegrationTag : std::false_type {};
template <>
struct IsIntegrationTag<FastPolygon> : std::true_type {};
template <>
struct IsIntegrationTag<Polygon> : std::true_type {};
template <>
struct IsIntegrationTag<MonteCarlo> : std::true_type {};
template <>
struct IsIntegrationTag<HeuristicMC> : std::true_type {};

template <class T>
inline constexpr bool is_integration_tag_v =
    IsIntegrationTag<std::decay_t<T>>::value;

inline IntegrationConfig function_integration_config(
    std::size_t components) {
    if (components == 0) {
        throw std::invalid_argument(
            "Function integration requires at least one component");
    }
    IntegrationConfig config;
    config.data_options = IntegralDataOptions{true, true, true, true};
    config.integral_components = components;
    return config;
}

inline void validate_integration_config(const IntegrationConfig& config) {
    if ((config.data_options.bulk_integral ||
         config.data_options.interface_integral) &&
        config.integral_components == 0) {
        throw std::invalid_argument(
            "Enabled function-integral storage requires integral_components > 0");
    }
}

} // namespace api_detail

/**
 * @brief Persistent Level-1/2 integral owner.
 *
 * This wrapper owns exactly one native VoronoiIntegral and, optionally, one
 * user function. The native integral keeps its dedicated mesh dirty tracker, so
 * repeated integrate() calls after refine/remove retain incremental semantics.
 */
template <class MeshT, class FunctionT = api_detail::NoIntegrationFunction>
class ApiVoronoiIntegral final {
public:
    using Mesh = MeshT;
    using Function = FunctionT;
    using NativeIntegral = VoronoiIntegral<Mesh, double, double>;
    using Index = typename NativeIntegral::Index;
    using AreaScalar = typename NativeIntegral::AreaScalar;
    using IntegralScalar = typename NativeIntegral::IntegralScalar;
    using Data = typename NativeIntegral::Data;
    using CellData = typename Data::CellData;

    static constexpr bool has_function =
        !std::is_same_v<Function, api_detail::NoIntegrationFunction>;

    ApiVoronoiIntegral(
        Mesh& mesh,
        IntegrationConfig config,
        Function function = Function{})
        : config_(std::move(config)),
          function_(std::move(function)),
          integral_(
              mesh,
              config_.integral_components,
              config_.data_options,
              config_.area_database_unit_length,
              config_.integral_database_unit_length) {
        api_detail::validate_integration_config(config_);
    }

    ApiVoronoiIntegral(const ApiVoronoiIntegral&) = delete;
    ApiVoronoiIntegral& operator=(const ApiVoronoiIntegral&) = delete;
    ApiVoronoiIntegral(ApiVoronoiIntegral&&) = delete;
    ApiVoronoiIntegral& operator=(ApiVoronoiIntegral&&) = delete;

    [[nodiscard]] NativeIntegral& level3_integral() noexcept {
        return integral_;
    }
    [[nodiscard]] const NativeIntegral& level3_integral() const noexcept {
        return integral_;
    }

    [[nodiscard]] Mesh& mesh() noexcept { return integral_.mesh(); }
    [[nodiscard]] const Mesh& mesh() const noexcept { return integral_.mesh(); }
    [[nodiscard]] Data& data() noexcept { return integral_.data(); }
    [[nodiscard]] const Data& data() const noexcept { return integral_.data(); }
    [[nodiscard]] const IntegrationConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] std::size_t size() const noexcept { return integral_.size(); }

    template <class F = Function,
              std::enable_if_t<!std::is_same_v<
                  F,
                  api_detail::NoIntegrationFunction>, int> = 0>
    [[nodiscard]] F& function() noexcept {
        return function_;
    }

    template <class F = Function,
              std::enable_if_t<!std::is_same_v<
                  F,
                  api_detail::NoIntegrationFunction>, int> = 0>
    [[nodiscard]] const F& function() const noexcept {
        return function_;
    }

    /**
     * Read one current public cell without exposing stable internal numbering.
     * The payload is still the native CellData, preserving the zero-overhead
     * aligned neighbours/area/interface layout.
     */
    [[nodiscard]] bool read_cell(Index public_cell, CellData& output) const {
        if (static_cast<std::size_t>(public_cell) >= size()) {
            throw std::out_of_range(
                "Public integral cell index is outside the current mesh");
        }
        const Index stable =
            mesh().index_mapping().public_to_internal(public_cell);
        return data().read_cell(stable, output);
    }

private:
    IntegrationConfig config_;
    Function function_;
    NativeIntegral integral_;
};

/** Level 1: persistent geometry integral, FastPolygon selected by integrate(). */
template <class MeshLike>
[[nodiscard]] auto voronoi_integral(MeshLike& mesh_like) {
    auto& mesh = api_detail::integration_native_mesh(mesh_like);
    using Mesh = std::remove_reference_t<decltype(mesh)>;
    using ApiIntegral = ApiVoronoiIntegral<Mesh>;
    return ApiIntegral(mesh, IntegrationConfig{});
}

/** Level 2: persistent integral with explicit storage/components configuration. */
template <class MeshLike>
[[nodiscard]] auto voronoi_integral(
    MeshLike& mesh_like,
    IntegrationConfig config) {
    api_detail::validate_integration_config(config);
    auto& mesh = api_detail::integration_native_mesh(mesh_like);
    using Mesh = std::remove_reference_t<decltype(mesh)>;
    using ApiIntegral = ApiVoronoiIntegral<Mesh>;
    return ApiIntegral(mesh, std::move(config));
}

/**
 * Level 1: attach one scalar function. Bulk and interface function integrals
 * are both enabled with one component and FastPolygon remains the default.
 */
template <
    class MeshLike,
    class Function,
    class NativeMesh = std::remove_reference_t<decltype(
        api_detail::integration_native_mesh(std::declval<MeshLike&>()))>,
    std::enable_if_t<api_detail::IsScalarIntegrationFunction<
        std::decay_t<Function>, typename NativeMesh::NodePoint>::value, int> = 0>
[[nodiscard]] auto voronoi_integral(
    MeshLike& mesh_like,
    Function&& function) {
    auto& mesh = api_detail::integration_native_mesh(mesh_like);
    using StoredFunction = std::decay_t<Function>;
    using ApiIntegral = ApiVoronoiIntegral<NativeMesh, StoredFunction>;
    return ApiIntegral(
        mesh,
        api_detail::function_integration_config(std::size_t{1}),
        std::forward<Function>(function));
}

/**
 * Level 1/2: attach a vector-valued function with an explicit component count.
 * Function-integral storage is enabled automatically.
 */
template <class MeshLike, class Function>
[[nodiscard]] auto voronoi_integral(
    MeshLike& mesh_like,
    std::size_t integral_components,
    Function&& function) {
    auto& mesh = api_detail::integration_native_mesh(mesh_like);
    using Mesh = std::remove_reference_t<decltype(mesh)>;
    using StoredFunction = std::decay_t<Function>;
    using ApiIntegral = ApiVoronoiIntegral<Mesh, StoredFunction>;
    return ApiIntegral(
        mesh,
        api_detail::function_integration_config(integral_components),
        std::forward<Function>(function));
}

/**
 * Level 2: function-bearing persistent integral with fully explicit storage
 * configuration. integral_components belongs to the config.
 */
template <class MeshLike, class Function>
[[nodiscard]] auto voronoi_integral(
    MeshLike& mesh_like,
    IntegrationConfig config,
    Function&& function) {
    api_detail::validate_integration_config(config);
    auto& mesh = api_detail::integration_native_mesh(mesh_like);
    using Mesh = std::remove_reference_t<decltype(mesh)>;
    using StoredFunction = std::decay_t<Function>;
    using ApiIntegral = ApiVoronoiIntegral<Mesh, StoredFunction>;
    return ApiIntegral(
        mesh,
        std::move(config),
        std::forward<Function>(function));
}

namespace api_detail {

template <class ApiIntegral>
[[nodiscard]] auto make_facade_algorithm(
    ApiIntegral& owner,
    FastPolygon) {
    if constexpr (ApiIntegral::has_function) {
        return make_fast_polygon_algorithm(
            owner.level3_integral(),
            owner.function());
    } else {
        return make_fast_polygon_algorithm(owner.level3_integral());
    }
}

template <class ApiIntegral>
[[nodiscard]] auto make_facade_algorithm(
    ApiIntegral& owner,
    Polygon) {
    if constexpr (ApiIntegral::has_function) {
        return make_polygon_algorithm(
            owner.level3_integral(),
            owner.function());
    } else {
        return make_polygon_algorithm(owner.level3_integral());
    }
}

template <class ApiIntegral>
[[nodiscard]] auto make_facade_algorithm(
    ApiIntegral& owner,
    MonteCarlo choice) {
    if constexpr (ApiIntegral::has_function) {
        using Function = typename ApiIntegral::Function;
        return MonteCarloAlgorithm<Function, Function>(
            owner.function(),
            owner.function(),
            choice.options);
    } else {
        return MonteCarloAlgorithm<>(choice.options);
    }
}

template <class ApiIntegral>
[[nodiscard]] auto make_facade_algorithm(
    ApiIntegral& owner,
    HeuristicMC choice) {
    static_assert(
        ApiIntegral::has_function,
        "HeuristicMC requires a function-bearing ApiVoronoiIntegral");
    return make_heuristic_mc_algorithm(
        owner.level3_integral(),
        owner.function(),
        choice.options);
}

} // namespace api_detail

/** Level 1: FastPolygon + serial execution. */
template <class Mesh, class Function>
IntegrationReport integrate(ApiVoronoiIntegral<Mesh, Function>& owner) {
    return integrate(owner, FastPolygon{}, SerialIntegrationExecution{});
}

/** Level 2: select a facade algorithm tag; serial execution remains default. */
template <
    class Mesh,
    class Function,
    class AlgorithmTag,
    std::enable_if_t<api_detail::is_integration_tag_v<AlgorithmTag>, int> = 0>
IntegrationReport integrate(
    ApiVoronoiIntegral<Mesh, Function>& owner,
    AlgorithmTag algorithm_tag) {
    return integrate(
        owner,
        std::move(algorithm_tag),
        SerialIntegrationExecution{});
}

/** Level 2: select both facade algorithm and integration execution policy. */
template <
    class Mesh,
    class Function,
    class AlgorithmTag,
    class Execution,
    std::enable_if_t<api_detail::is_integration_tag_v<AlgorithmTag>, int> = 0>
IntegrationReport integrate(
    ApiVoronoiIntegral<Mesh, Function>& owner,
    AlgorithmTag algorithm_tag,
    Execution execution) {
    auto algorithm = api_detail::make_facade_algorithm(
        owner,
        std::move(algorithm_tag));
    return highvoronoi::integrate(
        owner.level3_integral(),
        algorithm,
        std::move(execution));
}

/**
 * Level-3 bridge: run an explicitly constructed native algorithm against the
 * native integral owned by the facade without giving up persistent ownership.
 */
template <
    class Mesh,
    class Function,
    class Algorithm,
    std::enable_if_t<!api_detail::is_integration_tag_v<Algorithm>, int> = 0>
IntegrationReport integrate(
    ApiVoronoiIntegral<Mesh, Function>& owner,
    Algorithm& algorithm) {
    return highvoronoi::integrate(owner.level3_integral(), algorithm);
}

template <
    class Mesh,
    class Function,
    class Algorithm,
    class Execution,
    std::enable_if_t<!api_detail::is_integration_tag_v<Algorithm>, int> = 0>
IntegrationReport integrate(
    ApiVoronoiIntegral<Mesh, Function>& owner,
    Algorithm& algorithm,
    Execution execution) {
    return highvoronoi::integrate(
        owner.level3_integral(),
        algorithm,
        std::move(execution));
}

} // namespace highvoronoi
