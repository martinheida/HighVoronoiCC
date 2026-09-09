


#pragma once

#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/remove_voronoi.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace voronoi_incremental_test {

using Scalar = double;
using Index = std::uint32_t;

inline constexpr Scalar VerificationTolerance = Scalar{1e-12};
inline constexpr Scalar MinimumNodeDistance = Scalar{0.20};
inline constexpr std::size_t DatabaseUnits = 65536;
inline constexpr std::size_t HashCapacity = 8192;

inline std::size_t performed_checks = 0;
inline std::size_t failed_checks = 0;

inline void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

inline void section(std::string_view name) {
    std::cout << "\n============================================================\n"
              << name << '\n'
              << "============================================================\n";
}

template <int Dimension>
struct Types {
    using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
    using EdgeParameters = highvoronoi::EdgeBufferParams<>;
    using Database = highvoronoi::HVDataBase<
        highvoronoi::EmptyLock,
        DatabaseParameters,
        Dimension>;
    using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
    using Nodes = typename Mesh::InternalNodes;
    using Point = typename Mesh::NodePoint;
    using Boundary = typename Mesh::BoundaryType;
    using RayParameters = highvoronoi::RaycastParameters<
        highvoronoi::InRangeRaycast,
        Scalar>;
    // This regression intentionally exercises InRangeRaycast. Since the
    // library-wide default is CombinedRaycast, bind the non-default ray
    // parameter type explicitly instead of relying on Refine/Remove defaults.
    using Refine = highvoronoi::RefineVoronoi<
        Mesh,
        highvoronoi::geometry::KDSearch,
        RayParameters>;
    using Remove = highvoronoi::RemoveVoronoi<
        Mesh,
        highvoronoi::geometry::KDSearch,
        RayParameters>;
    using AffectedVector = typename Refine::AffectedVector;

    static_assert(
        std::is_same_v<AffectedVector, typename Remove::AffectedVector>,
        "RemoveVoronoi and RefineVoronoi must use one compatible AFFECTED type.");
};

template <int Dimension>
typename Types<Dimension>::DatabaseParameters database_parameters() {
    using Parameters = typename Types<Dimension>::DatabaseParameters;
    return Parameters{highvoronoi::DirectHash{HashCapacity}};
}

template <int Dimension>
typename Types<Dimension>::EdgeParameters edge_parameters() {
    using Parameters = typename Types<Dimension>::EdgeParameters;
    return Parameters{highvoronoi::DirectHash{HashCapacity}};
}

template <int Dimension>
typename Types<Dimension>::RayParameters ray_parameters() {
    typename Types<Dimension>::RayParameters parameters;
    parameters.variance_tolerance = Scalar{9e-14};
    return parameters;
}

template <int Dimension>
typename Types<Dimension>::Boundary make_boundary() {
    using Point = typename Types<Dimension>::Point;
    using Boundary = typename Types<Dimension>::Boundary;

    const Point dimensions = Point::Constant(Scalar{8});
    const Point offset = Point::Constant(Scalar{-4});
    return Boundary::cuboid(dimensions, offset, std::vector<Index>{});
}

template <int Dimension>
std::shared_ptr<typename Types<Dimension>::Database> make_database() {
    using Database = typename Types<Dimension>::Database;
    return std::make_shared<Database>(
        DatabaseUnits,
        database_parameters<Dimension>());
}

template <int Dimension>
std::vector<typename Types<Dimension>::Point> make_random_points(
    Index count,
    std::uint64_t seed) {

    using Point = typename Types<Dimension>::Point;

    std::mt19937_64 random(seed);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{-2.4},
        Scalar{2.4});

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(count));

    while (points.size() < static_cast<std::size_t>(count)) {
        Point candidate;
        for (int coordinate_index = 0;
             coordinate_index < Dimension;
             ++coordinate_index) {
            candidate[coordinate_index] = coordinate(random);
        }

        bool separated = true;
        for (const Point& existing : points) {
            if ((candidate - existing).norm() < MinimumNodeDistance) {
                separated = false;
                break;
            }
        }

        if (separated) {
            points.push_back(candidate);
        }
    }

    return points;
}

template <int Dimension>
std::unique_ptr<typename Types<Dimension>::Mesh> make_mesh(
    const std::vector<typename Types<Dimension>::Point>& points) {

    using Mesh = typename Types<Dimension>::Mesh;
    using Nodes = typename Types<Dimension>::Nodes;

    Nodes nodes(static_cast<Index>(points.size()));
    for (Index index = Index{0};
         index < static_cast<Index>(points.size());
         ++index) {
        nodes.set(index, points[static_cast<std::size_t>(index)]);
    }

    return std::make_unique<Mesh>(
        std::move(nodes),
        make_boundary<Dimension>(),
        make_database<Dimension>());
}

template <int Dimension>
void compute_full(typename Types<Dimension>::Mesh& mesh) {
    using Mesh = typename Types<Dimension>::Mesh;
    using DatabaseParameters = typename Types<Dimension>::DatabaseParameters;
    using EdgeParameters = typename Types<Dimension>::EdgeParameters;

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});
    auto raycaster = highvoronoi::make_raycaster(
        tree,
        ray_parameters<Dimension>());

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        DatabaseParameters,
        EdgeParameters>;

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());
    compute.compute();
}

inline std::vector<Index> deletion_indices(Index count, unsigned variant = 0) {
    std::vector<Index> result;

    auto add = [&](Index value) {
        if (value >= count) {
            return;
        }
        if (std::find(result.begin(), result.end(), value) == result.end()) {
            result.push_back(value);
        }
    };

    if (variant == 0) {
        add(Index{1});
        add(static_cast<Index>(count / Index{3}));
        add(static_cast<Index>((Index{2} * count) / Index{3}));
        add(static_cast<Index>(count - Index{2}));
    } else {
        add(Index{2});
        add(static_cast<Index>(count / Index{2}));
        add(static_cast<Index>(count - Index{3}));
    }

    std::sort(result.begin(), result.end());
    return result;
}

template <class Point>
void erase_public_points(
    std::vector<Point>& points,
    std::vector<Index> deleted_public) {

    std::sort(deleted_public.begin(), deleted_public.end(), std::greater<Index>{});
    deleted_public.erase(
        std::unique(deleted_public.begin(), deleted_public.end()),
        deleted_public.end());

    for (const Index public_index : deleted_public) {
        points.erase(points.begin() + static_cast<std::ptrdiff_t>(public_index));
    }
}

template <class Point>
void append_points(
    std::vector<Point>& target,
    const std::vector<Point>& appended) {
    target.insert(target.end(), appended.begin(), appended.end());
}

template <class Point>
std::vector<Point> slice(
    const std::vector<Point>& points,
    std::size_t first,
    std::size_t count) {

    return std::vector<Point>(
        points.begin() + static_cast<std::ptrdiff_t>(first),
        points.begin() + static_cast<std::ptrdiff_t>(first + count));
}


template <int Dimension>
bool verify_complete_state(
    typename Types<Dimension>::Mesh& mesh,
    std::string_view label) {

    const auto report = highvoronoi::verify_mesh_complete(
        mesh,
        VerificationTolerance,
        false);

    if (report.consistency.valid() && report.complete()) {
        return true;
    }

    std::cerr << "\n      incomplete baseline/state for " << label
              << ": consistency=" << report.consistency.valid()
              << " complete=" << report.complete() << '\n';
    (void)highvoronoi::verify_mesh_complete(
        mesh,
        VerificationTolerance,
        true);
    return false;
}

template <class AffectedVector>
std::size_t affected_count(const AffectedVector& affected) {
    std::size_t count = 0;
    for (std::size_t index = 0; index < affected.size(); ++index) {
        if (affected.test(index)) {
            ++count;
        }
    }
    return count;
}

template <int Dimension>
bool compare_neighbourhoods(
    typename Types<Dimension>::Mesh& actual,
    typename Types<Dimension>::Mesh& reference,
    std::string_view label) {

    using Mesh = typename Types<Dimension>::Mesh;
    using Index = typename Mesh::Index;

    if (actual.size() != reference.size()) {
        std::cerr << "\n      neighbour comparison for " << label
                  << " has different public node counts: actual="
                  << actual.size() << " reference=" << reference.size() << '\n';
        return false;
    }

    std::vector<Index> actual_neighbours;
    std::vector<Index> reference_neighbours;

    for (Index cell = Index{0}; cell < actual.size(); ++cell) {
        actual.compute_neighbors(cell);
        reference.compute_neighbors(cell);

        const bool actual_current =
            actual.neighbours(cell, actual_neighbours);
        const bool reference_current =
            reference.neighbours(cell, reference_neighbours);

        if (!actual_current || !reference_current) {
            std::cerr << "\n      neighbour comparison for " << label
                      << " left cell " << cell << " dirty after recomputation\n";
            return false;
        }

        // Public mappings after deletion need not preserve the internal sorted
        // order. Sort for comparison only; duplicates deliberately survive
        // because neighbour lists represent facets, not a mathematical set.
        std::sort(actual_neighbours.begin(), actual_neighbours.end());
        std::sort(reference_neighbours.begin(), reference_neighbours.end());

        if (actual_neighbours == reference_neighbours) {
            continue;
        }

        std::cerr << "\n      neighbour mismatch for " << label
                  << " at public cell " << cell << '\n'
                  << "      actual:    ";
        for (const Index neighbour : actual_neighbours) {
            std::cerr << neighbour << ' ';
        }
        std::cerr << "\n      reference: ";
        for (const Index neighbour : reference_neighbours) {
            std::cerr << neighbour << ' ';
        }
        std::cerr << '\n';
        return false;
    }

    return true;
}

template <int Dimension>
bool validate_against_fresh_reference(
    typename Types<Dimension>::Mesh& actual,
    const std::vector<typename Types<Dimension>::Point>& expected_points,
    std::string_view label) {

    auto reference = make_mesh<Dimension>(expected_points);
    compute_full<Dimension>(*reference);

    const auto actual_complete = highvoronoi::verify_mesh_complete(
        actual,
        VerificationTolerance,
        false);
    const auto reference_complete = highvoronoi::verify_mesh_complete(
        *reference,
        VerificationTolerance,
        false);
    const auto comparison = highvoronoi::compare_meshes(
        actual,
        *reference,
        Scalar{0},
        false);
    const bool neighbours_equal = compare_neighbourhoods<Dimension>(
        actual,
        *reference,
        label);

    const bool success =
        actual_complete.consistency.valid() &&
        actual_complete.complete() &&
        reference_complete.consistency.valid() &&
        reference_complete.complete() &&
        comparison.equal() &&
        neighbours_equal;

    if (success) {
        return true;
    }

    std::cerr << "\n      diagnostic for " << label << '\n'
              << "      actual: consistency="
              << actual_complete.consistency.valid()
              << " complete=" << actual_complete.complete()
              << " finite edge endpoints="
              << actual_complete.unique_finite_edge_endpoints
              << " infinite edges=" << actual_complete.infinite_edges
              << '\n'
              << "      reference: consistency="
              << reference_complete.consistency.valid()
              << " complete=" << reference_complete.complete()
              << '\n'
              << "      compare_meshes errors="
              << comparison.error_count() << '\n'
              << "      neighbours equal="
              << (neighbours_equal ? "yes" : "NO") << '\n';

    if (!actual_complete.consistency.valid() || !actual_complete.complete()) {
        (void)highvoronoi::verify_mesh_complete(
            actual,
            VerificationTolerance,
            true);
    }
    if (!comparison.equal()) {
        (void)highvoronoi::compare_meshes(
            actual,
            *reference,
            Scalar{0},
            true);
    }

    return false;
}

template <int Dimension>
std::uint64_t case_seed(
    std::uint64_t workflow_salt,
    Index initial_count) {

    const std::uint64_t dimension_salt =
        Dimension == 3
            ? 0x3344494d00000000ULL
            : 0x3444494d00000000ULL;

    return dimension_salt ^
           workflow_salt ^
           (static_cast<std::uint64_t>(initial_count) << 17U);
}

template <int Dimension>
std::string case_label(std::string_view workflow, Index initial_count) {
    return std::string(workflow) +
           " D=" + std::to_string(Dimension) +
           " N=" + std::to_string(initial_count);
}

} // namespace voronoi_incremental_test




