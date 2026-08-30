
#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/edge_iterator.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp> 
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

// ============================================================================
// One global type configuration for the complete test
// ============================================================================

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 4;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using ExtendedNodes = Mesh::ExtendedNodes;
using Point = Mesh::VertexPoint;
using Sigma = Mesh::Sigma;
using Iterator = highvoronoi::EdgeIterator<ExtendedNodes>;
using Candidate = Iterator::Candidate;
using FEIAction = Iterator::FEIAction;
using OnQueueEdges = Iterator::OnQueueEdges;
using OnSysVoronoi = Iterator::OnSysVoronoi;

inline constexpr Scalar tolerance = 2.0e-10;
std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

// ============================================================================
// Minimal test tools
// ============================================================================

void check(bool condition, std::string_view description) {
    ++performed_checks;

    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

std::shared_ptr<Database> make_database() {
    const DatabaseParameters parameters{
        highvoronoi::DirectHash{256}
    };
    return std::make_shared<Database>(256, parameters);
}

Point zero_point() {
    return Point::Zero();
}

Point cube_point(Index index) {
    Point result;

    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        const int bit = Dimension - 1 - coordinate;
        result[coordinate] =
            ((index >> bit) & Index{1}) != Index{0}
                ? Scalar{1}
                : Scalar{-1};
    }

    return result;
}

Mesh make_general_mesh() {
    constexpr Index node_count = Index{Dimension + 1};
    Nodes nodes(node_count);

    std::mt19937_64 random(0x4752414d5152ULL);
    std::normal_distribution<Scalar> normal(Scalar{0}, Scalar{1});

    for (Index index = 0; index < node_count; ++index) {
        Point point;
        do {
            for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
                point[coordinate] = normal(random);
            }
        } while (point.norm() < Scalar{0.1});

        point.normalize();
        nodes.set(index, point);
    }

    return Mesh(std::move(nodes), make_database());
}

Mesh make_cube_mesh() {
    constexpr Index node_count = Index{1} << Dimension;
    Nodes nodes(node_count);

    for (Index index = 0; index < node_count; ++index) {
        nodes.set(index, cube_point(index));
    }

    return Mesh(std::move(nodes), make_database());
}

Sigma full_signature(Index count) {
    Sigma sigma;
    sigma.reserve(static_cast<std::size_t>(count));

    for (Index index = 0; index < count; ++index) {
        sigma.push_back(index);
    }

    return sigma;
}

Mesh::VertexRecord load_vertex(Mesh& mesh, Index cell) {
    for (const auto& vertex : mesh.vertices(cell)) {
        return vertex;
    }
    throw std::runtime_error("Expected stored Voronoi vertex.");
}

void print_indices(const std::vector<Index>& values) {
    std::cout << '(';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << values[i];
    }
    std::cout << ')';
}

void print_sigma(const Sigma& sigma) {
    std::cout << "  sigma = ";
    print_indices(std::vector<Index>(sigma.begin(), sigma.end()));
    std::cout << '\n';
}

void print_edge(const Candidate& edge) {
    std::cout << "      minimal=";
    print_indices(edge.indices);
    std::cout << " full=";
    print_indices(edge.full_indices);
    std::cout << " skip=" << edge.skip;
    std::cout << " u=(";

    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        if (coordinate != 0) {
            std::cout << ',';
        }
        std::cout << std::setprecision(7) << edge.direction[coordinate];
    }

    std::cout << ") du=" << std::setprecision(3)
              << edge.cycle_error << '\n';
}

const char* action_name(FEIAction action) {
    switch (action) {
    case FEIAction::None:
        return "no FEI";
    case FEIAction::Reinitialized:
        return "FEI REINITIALIZED";
    case FEIAction::Loaded:
        return "FEI LOADED";
    case FEIAction::SkippedByCellOwnership:
        return "FEI SKIPPED: cell cannot own an edge";
    }
    return "unknown";
}

std::vector<Candidate> collect_queue_edges(
    Iterator& iterator,
    const Sigma& sigma,
    const Point& vertex,
    Index cell) {

    std::cout << "    [CALL] OnQueueEdges for cell " << cell << '\n';
    iterator.reset(sigma, vertex, cell, OnQueueEdges{});
    std::cout << "           " << action_name(iterator.last_fei_action()) << '\n';

    std::vector<Candidate> result;
    while (const auto edge = iterator.next()) {
        result.emplace_back(*edge);
        print_edge(result.back());
    }

    std::cout << "           -> " << result.size() << " edge(s)\n";
    return result;
}

std::vector<Candidate> collect_sys_edges(
    Iterator& iterator,
    const Sigma& sigma,
    const Point& vertex,
    Index cell) {

    std::cout << "    [CALL] OnSysVoronoi for cell " << cell << '\n';
    iterator.reset(sigma, vertex, cell, OnSysVoronoi{});
    std::cout << "           " << action_name(iterator.last_fei_action()) << '\n';

    std::vector<Candidate> result;
    while (const auto edge = iterator.next()) {
        result.emplace_back(*edge);
        print_edge(result.back());
    }

    std::cout << "           -> " << result.size() << " edge(s)\n";
    return result;
}

bool same_direction(const Point& left, const Point& right) {
    return (left - right).norm() <= tolerance;
}

bool same_candidate(const Candidate& left, const Candidate& right) {
    return left.indices == right.indices &&
           left.full_indices == right.full_indices &&
           left.skip == right.skip &&
           same_direction(left.direction, right.direction) &&
           std::abs(left.cycle_error - right.cycle_error) <= tolerance;
}

bool same_edges(
    const std::vector<Candidate>& left,
    const std::vector<Candidate>& right) {

    if (left.size() != right.size()) {
        return false;
    }

    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!same_candidate(left[i], right[i])) {
            return false;
        }
    }

    return true;
}

void check_cell_ownership(
    const std::vector<Candidate>& edges,
    Index cell) {

    for (const Candidate& edge : edges) {
        check(
            !edge.full_indices.empty(),
            "returned edge has a non-empty full signature");

        if (edge.full_indices.empty()) {
            continue;
        }

        const Index owner = *std::min_element(
            edge.full_indices.begin(),
            edge.full_indices.end());

        check(
            owner == cell,
            "returned edge is owned by the active cell");
    }
}

// ============================================================================
// General position: only cells 0 and 1 own edges
// ============================================================================

void test_general_cellwise_iteration() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] general position: cell-wise iteration\n";
    std::cout << "============================================================\n";

    Mesh mesh = make_general_mesh();
    const Sigma sigma = full_signature(Index{Dimension + 1});
    const Point vertex = zero_point();

    const auto address = mesh.store_vertex(vertex, sigma);
    check(address != 0, "store general-position vertex");

    const auto stored = load_vertex(mesh, Index{0});
    print_sigma(stored.sigma);

    Iterator queue_iterator(mesh.concrete_extended_nodes());
    auto sys_iterator = queue_iterator.create_shared();

    const std::vector<std::size_t> expected_counts{4, 1, 0, 0, 0};
    std::vector<std::vector<Index>> all_edges;

    for (Index cell = 0; cell < Index{Dimension + 1}; ++cell) {
        std::cout << "\n  [CELL " << cell << "]\n";

        const auto queue_edges = collect_queue_edges(
            queue_iterator,
            stored.sigma,
            stored.position,
            cell);

        const auto sys_edges = collect_sys_edges(
            *sys_iterator,
            stored.sigma,
            stored.position,
            cell);

        check(
            queue_iterator.last_fei_action() == FEIAction::None &&
                sys_iterator->last_fei_action() == FEIAction::None,
            "general-position iteration does not touch FEIStorage");

        check(
            same_edges(queue_edges, sys_edges),
            "OnQueueEdges and OnSysVoronoi return identical edges");

        check(
            queue_edges.size() == expected_counts[static_cast<std::size_t>(cell)],
            "active cell returns the expected number of owned edges");

        check_cell_ownership(queue_edges, cell);

        for (const Candidate& edge : queue_edges) {
            auto key = edge.indices;
            std::sort(key.begin(), key.end());
            all_edges.push_back(std::move(key));
        }
    }

    std::sort(all_edges.begin(), all_edges.end());
    check(
        all_edges.size() == static_cast<std::size_t>(Dimension + 1),
        "all cells together return exactly d+1 general edges");

    check(
        std::adjacent_find(all_edges.begin(), all_edges.end()) == all_edges.end(),
        "no general edge is returned for two different cells");

    check(
        queue_iterator.storage_cache().empty(),
        "general-position test creates no FEIStorage cache entry");
}

// ============================================================================
// Degenerate cube: Queue recomputes, Sys immediately reloads the same FEI
// ============================================================================

void test_degenerate_cellwise_shared_fei() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] degenerate cube: shared FEI cache + cell ownership\n";
    std::cout << "============================================================\n";

    constexpr Index node_count = Index{1} << Dimension;

    Mesh mesh = make_cube_mesh();
    const Sigma sigma = full_signature(node_count);
    const Point vertex = zero_point();

    const auto address = mesh.store_vertex(vertex, sigma);
    check(address != 0, "store degenerate cube vertex");

    const auto stored = load_vertex(mesh, Index{0});
    print_sigma(stored.sigma);

    Iterator queue_iterator(mesh.concrete_extended_nodes());
    auto sys_iterator = queue_iterator.create_shared();

    check(
        queue_iterator.storage_cache_handle() ==
            sys_iterator->storage_cache_handle(),
        "both EdgeIterator instances share exactly one FEIStorageCache");

    const std::vector<std::size_t> expected_counts{
        4, 1, 1, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 0
    };

    const std::vector<std::vector<Index>> expected_all_edges{
        {Index{0}, Index{1}, Index{2}, Index{4}},
        {Index{0}, Index{1}, Index{2}, Index{8}},
        {Index{0}, Index{1}, Index{4}, Index{8}},
        {Index{0}, Index{2}, Index{4}, Index{8}},
        {Index{1}, Index{3}, Index{5}, Index{9}},
        {Index{2}, Index{3}, Index{6}, Index{10}},
        {Index{4}, Index{5}, Index{6}, Index{12}},
        {Index{8}, Index{9}, Index{10}, Index{12}}
    };

    std::vector<std::vector<Index>> all_edges;

    for (Index cell = 0; cell < node_count; ++cell) {
        std::cout << "\n  [CELL " << cell << "]\n";
        std::cout << "    order: OnQueueEdges -> OnSysVoronoi\n";

        const auto queue_edges = collect_queue_edges(
            queue_iterator,
            stored.sigma,
            stored.position,
            cell);

        const bool ownership_shortcut =
            static_cast<std::size_t>(cell) >
            static_cast<std::size_t>(node_count) -
                static_cast<std::size_t>(Dimension);

        const FEIAction expected_queue_action = ownership_shortcut
            ? FEIAction::SkippedByCellOwnership
            : FEIAction::Reinitialized;

        check(
            queue_iterator.last_fei_action() == expected_queue_action,
            ownership_shortcut
                ? "OnQueueEdges stops before FEI initialization for this cell"
                : "OnQueueEdges invalidates and recomputes FEI initialization");

        const auto* after_queue = queue_iterator.storage_cache().find(stored.sigma);
        check(after_queue != nullptr, "FEI cache contains the vertex signature");
        if (after_queue != nullptr) {
            if (ownership_shortcut) {
                check(
                    !after_queue->initialized(),
                    "ownership shortcut leaves the invalidated FEI uninitialized");
            } else {
                check(
                    after_queue->initialized(),
                    "FEI is initialized after OnQueueEdges");
                check(
                    after_queue->stored_cell() == cell,
                    "FEI cache records the currently initialized cell");
            }
        }

        const auto sys_edges = collect_sys_edges(
            *sys_iterator,
            stored.sigma,
            stored.position,
            cell);

        const FEIAction expected_sys_action = ownership_shortcut
            ? FEIAction::SkippedByCellOwnership
            : FEIAction::Loaded;

        check(
            sys_iterator->last_fei_action() == expected_sys_action,
            ownership_shortcut
                ? "OnSysVoronoi uses the same immediate ownership shortcut"
                : "OnSysVoronoi loads the FEI produced by OnQueueEdges");

        check(
            same_edges(queue_edges, sys_edges),
            "OnQueueEdges and OnSysVoronoi return identical edges in identical order");

        check(
            queue_edges.size() == expected_counts[static_cast<std::size_t>(cell)],
            "active cube cell returns the expected number of owned edges");

        check_cell_ownership(queue_edges, cell);

        for (const Candidate& edge : queue_edges) {
            auto key = edge.indices;
            std::sort(key.begin(), key.end());
            all_edges.push_back(std::move(key));
        }
    }

    std::sort(all_edges.begin(), all_edges.end());
    auto expected = expected_all_edges;
    std::sort(expected.begin(), expected.end());

    check(
        queue_iterator.storage_cache().size() == 1,
        "one degenerate vertex signature owns exactly one shared FEIStorage");

    check(
        all_edges == expected,
        "all active cells together return exactly the eight cube edges");

    check(
        std::adjacent_find(all_edges.begin(), all_edges.end()) == all_edges.end(),
        "no cube edge is returned for two different cells");
}


// ============================================================================
// Degenerate boundary vertex: the last possible owner must not be skipped
// ============================================================================

void test_degenerate_boundary_last_owner() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] degenerate boundary vertex: last possible owner\n";
    std::cout << "============================================================\n";

    using Mesh3 = highvoronoi::VoronoiMesh<Scalar, 3, Database>;
    using Nodes3 = Mesh3::InternalNodes;
    using Point3 = Mesh3::NodePoint;
    using Boundary3 = Mesh3::BoundaryType;
    using Sigma3 = Mesh3::Sigma;
    using Iterator3 = highvoronoi::EdgeIterator<Mesh3::ExtendedNodes>;
    using FEIAction3 = typename Iterator3::FEIAction;

    constexpr Index side = Index{4};
    constexpr Index node_count = side * side * side;

    const auto point3 = [](Scalar x, Scalar y, Scalar z) {
        Point3 result;
        result << x, y, z;
        return result;
    };

    Nodes3 nodes(node_count);
    Index flat = Index{0};
    for (Index z = Index{0}; z < side; ++z) {
        for (Index y = Index{0}; y < side; ++y) {
            for (Index x = Index{0}; x < side; ++x) {
                nodes.set(
                    flat++,
                    point3(
                        Scalar{0.2} + Scalar{0.2} * x,
                        Scalar{0.2} + Scalar{0.2} * y,
                        Scalar{0.2} + Scalar{0.2} * z));
            }
        }
    }

    const Boundary3 boundary = Boundary3::cuboid(
        point3(Scalar{2}, Scalar{2}, Scalar{2}),
        point3(Scalar{-1}, Scalar{-1}, Scalar{-1}),
        std::vector<Index>{});

    Mesh3 mesh(std::move(nodes), boundary, make_database());
    auto& extended = mesh.concrete_extended_nodes();
    Iterator3 iterator(extended);

    // Exact boundary vertex found in the Cartesian completeness regression.
    // Boundary generator 69 is the lower z-plane mirror generator.
    const Sigma3 sigma{Index{0}, Index{1}, Index{4}, Index{5}, Index{69}};
    const Point3 vertex = point3(Scalar{0.3}, Scalar{0.3}, Scalar{-1});
    const std::vector<Index> active_boundary{Index{69}};

    const std::vector<Index> cells{Index{0}, Index{1}, Index{4}, Index{5}};
    const std::vector<std::size_t> expected_counts{3, 1, 1, 0};

    bool target_found = false;

    for (std::size_t position = 0; position < cells.size(); ++position) {
        const Index cell = cells[position];

        std::cout << "\n  [CELL " << cell << "]\n";
        extended.activate_cell(cell, active_boundary);
        iterator.reset(
            sigma,
            vertex,
            cell,
            typename Iterator3::OnQueueEdges{});

        std::cout << "    FEI action: "
                  << static_cast<int>(iterator.last_fei_action()) << '\n';

        std::size_t edge_count = 0;
        while (const auto edge = iterator.next()) {
            ++edge_count;

            std::vector<Index> minimal(
                edge->indices().begin(),
                edge->indices().end());
            std::vector<Index> full(
                edge->full_indices().begin(),
                edge->full_indices().end());

            std::cout << "      minimal=";
            print_indices(minimal);
            std::cout << " full=";
            print_indices(full);
            std::cout << " skip=" << edge->skip() << '\n';

            if (cell == Index{4} &&
                full == std::vector<Index>{Index{4}, Index{5}, Index{69}} &&
                edge->skip() == Index{0}) {
                target_found = true;
            }
        }

        check(
            edge_count == expected_counts[position],
            "boundary regression cell returns the expected number of owned edges");

        if (cell == Index{4}) {
            check(
                iterator.last_fei_action() != FEIAction3::SkippedByCellOwnership,
                "cell 4 is not rejected by the ownership shortcut");
        }
    }

    check(
        target_found,
        "cell 4 emits full edge {4,5,69} with skip 0");
}

} // namespace

int main() {
    test_general_cellwise_iteration();
    test_degenerate_cellwise_shared_fei();
    test_degenerate_boundary_last_owner();

    std::cout << "\n============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    std::cout << "============================================================\n";

    return failed_checks == 0 ? 0 : 1;
}
