#pragma once

#include <highvoronoi/integration/detail/area_database.hpp>
#include <highvoronoi/integration/detail/hybrid_numeric_database.hpp>
#include <highvoronoi/integration/detail/integral_database.hpp>
#include <highvoronoi/integration/integral_engine.hpp>

namespace highvoronoi::detail {

template <typename AreaScalar, typename Index>
struct AreaEngineAccess {
    using Engine = AreaEngineInterface<AreaScalar, Index>;
    using Address = std::size_t;

    [[nodiscard]] static Address capacity(const Engine& engine) noexcept {
        return engine.area_address_capacity();
    }

    [[nodiscard]] static bool read(
        const Engine& engine,
        Address local_address,
        std::vector<AreaScalar>& values) {
        return engine.read_area(local_address, values);
    }
};

template <typename IntegralScalar, typename Index>
struct IntegralEngineAccess {
    using Engine = IntegralEngineInterface<IntegralScalar, Index>;
    using Address = std::size_t;

    [[nodiscard]] static Address capacity(const Engine& engine) noexcept {
        return engine.integral_address_capacity();
    }

    [[nodiscard]] static bool read(
        const Engine& engine,
        Address local_address,
        std::vector<IntegralScalar>& values) {
        return engine.read_integral(local_address, values);
    }
};

/** Stored + virtual engine area records in one immutable logical address space. */
template <class Lock, typename AreaScalar, typename Index>
using HybridAreaDatabase = HybridNumericDatabase<
    Lock,
    AreaScalar,
    AreaDatabase<Lock, AreaScalar>,
    AreaEngineInterface<AreaScalar, Index>,
    AreaEngineAccess<AreaScalar, Index>>;

/** Stored + virtual engine integral records in one immutable logical address space. */
template <class Lock, typename IntegralScalar, typename Index>
using HybridIntegralDatabase = HybridNumericDatabase<
    Lock,
    IntegralScalar,
    IntegralDatabase<Lock, IntegralScalar>,
    IntegralEngineInterface<IntegralScalar, Index>,
    IntegralEngineAccess<IntegralScalar, Index>>;

} // namespace highvoronoi::detail
