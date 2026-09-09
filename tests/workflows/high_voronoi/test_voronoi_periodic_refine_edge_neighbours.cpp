

#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/high_voronoi_compute_mesh.hpp>
#include <highvoronoi/algorithm/high_voronoi/high_voronoi_incremental_backend.hpp>
#include <highvoronoi/algorithm/incremental/incremental_voronoi_backend.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;
inline constexpr Index InitialVisibleNodeCount = Index{27};

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;

using HighMesh = highvoronoi::HighVoronoiMesh<Scalar, Dimension, Database>;
using ComputeRoundState = highvoronoi::HighVoronoiComputeRoundState<HighMesh, highvoronoi::detail::BitVector>;
using ComputeMesh = highvoronoi::HighVoronoiComputeMesh<HighMesh, highvoronoi::detail::BitVector>;
using OrdinaryMesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using OrdinaryNodes = OrdinaryMesh::InternalNodes;

using Point = HighMesh::NodePoint;
using VertexPoint = HighMesh::VertexPoint;
using Boundary = HighMesh::BoundaryType;
using ShiftMask = HighMesh::ShiftMask;
using PeriodicRequest = std::pair<Index, Index>;
using Sigma = std::vector<Index>;
using Address = std::size_t;

using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast,
    Scalar>;

using HighRefiner = highvoronoi::RefineVoronoi<
    HighMesh,
    highvoronoi::geometry::KDSearch,
    RayParameters,
    highvoronoi::SingleThread,
    highvoronoi::SingleThread,
    DatabaseParameters,
    EdgeParameters>;

inline constexpr std::size_t DatabaseUnits = 65536;
inline constexpr std::size_t HashCapacity = 65536;
inline constexpr Scalar RayVarianceTolerance = Scalar{9e-14};
inline constexpr Scalar VerificationTolerance = Scalar{1e-12};
inline constexpr Scalar VertexMatchTolerance = Scalar{2e-9};
inline constexpr Scalar MinimumNodeDistance = Scalar{0.055};
inline constexpr Scalar BoundaryTolerance = Scalar{1e-10};

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }
    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

void section(std::string_view title) {
    std::cout << "\n============================================================\n"
              << title << '\n'
              << "============================================================\n";
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

DatabaseParameters database_parameters() {
    return DatabaseParameters{highvoronoi::DirectHash{HashCapacity}};
}

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(DatabaseUnits, database_parameters());
}

RayParameters ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = RayVarianceTolerance;
    return parameters;
}

EdgeParameters edge_parameters() {
    return EdgeParameters{highvoronoi::DirectHash{HashCapacity}};
}

Boundary periodic_unit_cube() {
    return Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{Index{0}, Index{1}});
}

std::vector<Point> future_refinement_points() {
    return {
        point(0.008, 0.18, 0.37),
        point(0.992, 0.73, 0.63),
        point(0.43, 0.009, 0.84)
    };
}

bool far_enough(
    const Point& candidate,
    const std::vector<Point>& points,
    Scalar minimum_distance = MinimumNodeDistance) {
    for (const Point& existing : points) {
        if ((candidate - existing).norm() < minimum_distance) {
            return false;
        }
    }
    return true;
}

std::vector<Point> make_initial_visible_points() {
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(InitialVisibleNodeCount));

    const std::vector<Point> future = future_refinement_points();
    std::mt19937_64 random(0x494e4352454d454eULL); // "INCREMEN"
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{0.025}, Scalar{0.975});

    while (points.size() < static_cast<std::size_t>(InitialVisibleNodeCount)) {
        Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));

        if (!far_enough(candidate, points)) continue;
        if (!far_enough(candidate, future, Scalar{0.085})) continue;
        points.push_back(candidate);
    }
    return points;
}

std::unique_ptr<HighMesh> make_empty_high_mesh() {
    return std::make_unique<HighMesh>(
        Index{Dimension},
        periodic_unit_cube(),
        std::in_place,
        DatabaseUnits,
        database_parameters());
}

void append_visible_points(HighMesh& mesh, const std::vector<Point>& points) {
    for (const Point& value : points) {
        (void)mesh.append_visible_node(value);
    }
}

OrdinaryNodes make_ordinary_nodes(const std::vector<Point>& points) {
    OrdinaryNodes nodes(static_cast<Index>(points.size()), Index{Dimension});
    for (Index i = Index{0}; i < static_cast<Index>(points.size()); ++i) {
        nodes.set(i, points[static_cast<std::size_t>(i)]);
    }
    return nodes;
}

std::unique_ptr<OrdinaryMesh> make_ordinary_mesh(
    const std::vector<Point>& points,
    const Boundary& boundary) {
    return std::make_unique<OrdinaryMesh>(
        make_ordinary_nodes(points),
        boundary,
        make_database());
}

void compute_full(OrdinaryMesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters());

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        OrdinaryMesh,
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
        database_parameters(),
        edge_parameters());
    compute.compute();
}

// -----------------------------------------------------------------------------
// Periodic-reference planning copied from the production HighVoronoi logic.
// This is used only to obtain the exact first reference block and its final
// internal-boundary geometry.
// -----------------------------------------------------------------------------

struct PlannedReference {
    Index reference_internal = Index{0};
    ShiftMask shift;
    Point point;
};

bool planned_contains(
    const std::vector<PlannedReference>& planned,
    Index reference,
    const ShiftMask& shift) {
    return std::any_of(
        planned.begin(),
        planned.end(),
        [&](const PlannedReference& item) {
            return item.reference_internal == reference && item.shift == shift;
        });
}

void enumerate_shift_combinations(
    const HighMesh& mesh,
    Index reference,
    const std::vector<Index>& candidates,
    std::size_t position,
    ShiftMask& current,
    highvoronoi::detail::BitVector& taboo,
    std::vector<PlannedReference>& output) {
    if (position == candidates.size()) {
        const bool nonempty = std::any_of(
            current.begin(), current.end(),
            [](std::uint8_t value) { return value != 0; });
        if (!nonempty ||
            mesh.find_reference_copy(reference, current).has_value() ||
            planned_contains(output, reference, current)) {
            return;
        }

        Point value;
        mesh.internal_nodes().copy_node(reference, value.data());
        value += mesh.periodic_shift(current);
        output.push_back(PlannedReference{reference, current, value});
        return;
    }

    enumerate_shift_combinations(
        mesh, reference, candidates, position + 1, current, taboo, output);

    const Index plane = candidates[position];
    if (taboo.test(static_cast<std::size_t>(plane))) return;

    const auto partner = mesh.external_boundary()[plane].periodic_partner();
    if (!partner) {
        throw std::logic_error(
            "periodic request selected a non-periodic boundary plane");
    }

    current[static_cast<std::size_t>(plane)] = std::uint8_t{1};
    const bool partner_was_taboo =
        taboo.test(static_cast<std::size_t>(*partner));
    taboo.set(static_cast<std::size_t>(*partner));

    enumerate_shift_combinations(
        mesh, reference, candidates, position + 1, current, taboo, output);

    if (!partner_was_taboo) {
        taboo.reset(static_cast<std::size_t>(*partner));
    }
    current[static_cast<std::size_t>(plane)] = std::uint8_t{0};
}

std::vector<PlannedReference> materialize_periodic_requests(
    const HighMesh& mesh,
    const std::vector<PeriodicRequest>& requests) {
    const Index node_count = mesh.internal_node_count();
    const Index plane_count = mesh.external_boundary().size();

    std::vector<std::vector<Index>> requested_planes(
        static_cast<std::size_t>(node_count));

    for (const auto& [reference, plane] : requests) {
        if (reference >= node_count || plane >= plane_count ||
            !mesh.is_active_internal(reference) ||
            !mesh.is_visible_internal(reference) ||
            !mesh.external_boundary()[plane].is_periodic()) {
            continue;
        }
        requested_planes[static_cast<std::size_t>(reference)].push_back(plane);
    }

    std::vector<PlannedReference> result;
    for (Index reference = Index{0}; reference < node_count; ++reference) {
        auto& candidates = requested_planes[static_cast<std::size_t>(reference)];
        if (candidates.empty()) continue;

        std::sort(candidates.begin(), candidates.end());
        candidates.erase(
            std::unique(candidates.begin(), candidates.end()),
            candidates.end());

        ShiftMask current(
            static_cast<std::size_t>(plane_count),
            std::uint8_t{0});
        highvoronoi::detail::BitVector taboo(
            static_cast<std::size_t>(plane_count));
        enumerate_shift_combinations(
            mesh,
            reference,
            candidates,
            std::size_t{0},
            current,
            taboo,
            result);
    }
    return result;
}

std::vector<Point> planned_points(const std::vector<PlannedReference>& planned) {
    std::vector<Point> result;
    result.reserve(planned.size());
    for (const auto& item : planned) result.push_back(item.point);
    return result;
}

std::vector<Index> append_exact_reference_block(
    HighMesh& mesh,
    const std::vector<PlannedReference>& planned) {
    std::vector<Index> result;
    result.reserve(planned.size());
    for (const auto& item : planned) {
        result.push_back(mesh.append_reference_node_internal(
            item.reference_internal,
            item.shift));
    }
    return result;
}

// -----------------------------------------------------------------------------
// Test-local periodic ordinary refinement.
//
// Production RefineVoronoi deliberately rejects a periodic ordinary boundary.
// For this diagnostic we want exactly the same ordinary incremental algorithm
// on the fixed HighVoronoi internal boundary, so this class copies the ordinary
// RefineVoronoi core and omits only Backend::validate(mesh).
// -----------------------------------------------------------------------------

class PeriodicOrdinaryRefiner final {
public:
    using Mesh = OrdinaryMesh;
    using MeshThreading = highvoronoi::SingleThread;
    using CastThreading = highvoronoi::SingleThread;
    using AffectedVector = highvoronoi::IncrementalVoronoiAffectedVector<
        MeshThreading,
        CastThreading>;
    using Backend = highvoronoi::IncrementalVoronoiBackend<Mesh, AffectedVector>;
    using State = typename Backend::State;
    using IncrementalMesh = typename Backend::ComputeMesh;

    struct Report {
        std::size_t appended_nodes = 0;
        std::size_t new_cell_vertices = 0;
        std::size_t affected_nodes = 0;
        std::size_t invalidated_old_vertices = 0;
    };

    PeriodicOrdinaryRefiner(Mesh& mesh, std::vector<Point> new_nodes)
        : mesh_(mesh),
          new_nodes_(std::move(new_nodes)),
          old_internal_count_(Backend::internal_size(mesh)),
          affected_(static_cast<std::size_t>(old_internal_count_)) {}

    void compute() {
        appended_internal_nodes_ = Backend::append_visible_nodes(mesh_, new_nodes_);
        report_.appended_nodes = appended_internal_nodes_.size();
        if (appended_internal_nodes_.empty()) return;

        highvoronoi::detail::ensure_affected_size(
            affected_,
            Backend::internal_size(mesh_));

        State state = Backend::make_state(
            mesh_,
            old_internal_count_,
            affected_);
        Backend::set_new_phase(state);

        IncrementalMesh computation(
            mesh_,
            state,
            appended_internal_nodes_);

        report_.new_cell_vertices = highvoronoi::detail::run_incremental_compute(
            computation,
            static_cast<Index>(appended_internal_nodes_.size()),
            highvoronoi::geometry::KDSearch{8, 1},
            ray_parameters(),
            MeshThreading{},
            CastThreading{},
            database_parameters(),
            edge_parameters());

        invalidate_old_vertices(computation, state);

        report_.affected_nodes =
            highvoronoi::detail::active_incremental_affected_nodes(
                mesh_,
                affected_,
                old_internal_count_).size();
    }

    [[nodiscard]] const Report& report() const noexcept { return report_; }
    [[nodiscard]] const std::vector<Index>& appended_internal_nodes() const noexcept {
        return appended_internal_nodes_;
    }

private:
    bool old_vertex_is_candidate(
        const Mesh::Sigma& internal_sigma,
        const State& state) const {
        bool has_ordinary = false;
        for (const Index generator : internal_sigma) {
            if (generator >= state.internal_count()) break;
            has_ordinary = true;
            if (!Backend::is_active_internal(mesh_, generator) ||
                !state.is_old_node(generator) ||
                !state.is_affected(generator)) {
                return false;
            }
        }
        return has_ordinary;
    }

    void invalidate_old_vertices(
        IncrementalMesh& computation,
        const State& state) {
        const Index new_count = static_cast<Index>(appended_internal_nodes_.size());
        if (new_count == Index{0}) return;

        auto tree = highvoronoi::geometry::make_search_tree(
            computation,
            highvoronoi::geometry::KDSearch{8, 1});
        auto search_data = tree.make_backend_data();

        VertexPoint position;
        Point generator_point;
        Mesh::Sigma sigma;

        const std::vector<Index> affected_old =
            highvoronoi::detail::active_incremental_affected_nodes(
                mesh_,
                affected_,
                old_internal_count_);

        for (const Index primary_internal : affected_old) {
            const auto& addresses = Backend::primary_addresses(
                mesh_,
                primary_internal);
            const std::size_t address_count = addresses.size();

            for (std::size_t p = 0; p < address_count; ++p) {
                const Address address = addresses[p];
                sigma.clear();
                Backend::read_internal_vertex(mesh_, address, position, sigma);
                if (sigma.empty() || !old_vertex_is_candidate(sigma, state)) {
                    continue;
                }

                Index first_generator = state.internal_count();
                for (const Index generator : sigma) {
                    if (generator < state.internal_count()) {
                        first_generator = generator;
                        break;
                    }
                }
                if (first_generator == state.internal_count()) continue;

                Backend::copy_internal_node(
                    mesh_,
                    first_generator,
                    generator_point);

                long double radius_squared = 0.0L;
                for (Index coordinate = Index{0};
                     coordinate < mesh_.dimension();
                     ++coordinate) {
                    const long double delta =
                        static_cast<long double>(
                            position[static_cast<Eigen::Index>(coordinate)]) -
                        static_cast<long double>(
                            generator_point[static_cast<Eigen::Index>(coordinate)]);
                    radius_squared += delta * delta;
                }
                const Scalar radius = static_cast<Scalar>(
                    std::sqrt(radius_squared));

                tree.write_point(search_data, position);
                const auto skip_not_new = [new_count](Index compute_public) {
                    return compute_public >= new_count;
                };
                const auto nearest = tree.nn(search_data, skip_not_new);
                if (!nearest) continue;

                if (static_cast<Scalar>(nearest->distance) + Scalar{1e-10} >= radius) {
                    continue;
                }

                if (Backend::erase_internal_vertex(mesh_, address, sigma)) {
                    ++report_.invalidated_old_vertices;
                }
            }
        }

        if (report_.invalidated_old_vertices != 0) {
            Backend::compact_vertex_lists(mesh_);
        }
    }

    Mesh& mesh_;
    std::vector<Point> new_nodes_;
    Index old_internal_count_ = Index{0};
    std::vector<Index> appended_internal_nodes_;
    AffectedVector affected_;
    Report report_{};
};

// -----------------------------------------------------------------------------
// Raw snapshots and comparison. Same node numbering, same boundary. For the
// HighVoronoi comparison only periodic-internal-boundary vertices are filtered,
// because HighVoronoi deliberately does not persist them.
// -----------------------------------------------------------------------------

struct VertexRecord {
    Sigma sigma;
    VertexPoint position;
};

struct MeshSnapshot {
    std::vector<std::vector<VertexRecord>> cells;
    std::size_t filtered_periodic_boundary_occurrences = 0;
};

std::string sigma_string(const Sigma& sigma) {
    std::ostringstream out;
    out << '{';
    for (std::size_t i = 0; i < sigma.size(); ++i) {
        if (i != 0) out << ',';
        out << sigma[i];
    }
    out << '}';
    return out.str();
}

bool touches_periodic_boundary(
    const Sigma& sigma,
    Index ordinary_count,
    const Boundary& boundary) {
    for (const Index generator : sigma) {
        if (generator < ordinary_count) continue;
        const Index plane = static_cast<Index>(generator - ordinary_count);
        if (plane < boundary.size() && boundary.at(plane).is_periodic()) {
            return true;
        }
    }
    return false;
}

template <class MeshLike>
MeshSnapshot capture_snapshot(
    const MeshLike& mesh,
    const Boundary& boundary,
    bool filter_periodic_boundary) {
    MeshSnapshot result;
    result.cells.resize(static_cast<std::size_t>(mesh.size()));
    const Index ordinary_count = mesh.size();

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        auto& output = result.cells[static_cast<std::size_t>(cell)];
        for (const auto& vertex : mesh.vertices(cell)) {
            Sigma sigma(vertex.sigma.begin(), vertex.sigma.end());
            if (filter_periodic_boundary &&
                touches_periodic_boundary(sigma, ordinary_count, boundary)) {
                ++result.filtered_periodic_boundary_occurrences;
                continue;
            }
            output.push_back(VertexRecord{std::move(sigma), vertex.position});
        }
    }
    return result;
}

bool same_vertex(const VertexRecord& left, const VertexRecord& right) {
    return left.sigma == right.sigma &&
           (left.position - right.position).norm() <= VertexMatchTolerance;
}

std::vector<VertexRecord> unique_vertices(
    const std::vector<VertexRecord>& values) {
    std::vector<VertexRecord> result;
    for (const auto& value : values) {
        const auto found = std::find_if(
            result.begin(), result.end(),
            [&](const VertexRecord& existing) {
                return same_vertex(existing, value);
            });
        if (found == result.end()) result.push_back(value);
    }
    return result;
}

struct Comparison {
    bool match = true;
    std::size_t left_occurrences = 0;
    std::size_t right_occurrences = 0;
    std::size_t matched_occurrences = 0;
    std::vector<VertexRecord> left_only_occurrences;
    std::vector<VertexRecord> right_only_occurrences;
};

Comparison compare_snapshots(
    const MeshSnapshot& left,
    const MeshSnapshot& right) {
    Comparison result;
    if (left.cells.size() != right.cells.size()) {
        result.match = false;
        return result;
    }

    for (std::size_t cell = 0; cell < left.cells.size(); ++cell) {
        const auto& a = left.cells[cell];
        const auto& b = right.cells[cell];
        result.left_occurrences += a.size();
        result.right_occurrences += b.size();

        std::vector<std::uint8_t> used_b(b.size(), 0);
        std::vector<std::uint8_t> used_a(a.size(), 0);

        for (std::size_t i = 0; i < a.size(); ++i) {
            for (std::size_t j = 0; j < b.size(); ++j) {
                if (used_b[j] != 0 || !same_vertex(a[i], b[j])) continue;
                used_a[i] = 1;
                used_b[j] = 1;
                ++result.matched_occurrences;
                break;
            }
        }

        for (std::size_t i = 0; i < a.size(); ++i) {
            if (used_a[i] == 0) {
                result.match = false;
                result.left_only_occurrences.push_back(a[i]);
            }
        }
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (used_b[j] == 0) {
                result.match = false;
                result.right_only_occurrences.push_back(b[j]);
            }
        }
    }
    return result;
}

void print_comparison(
    std::string_view name,
    const MeshSnapshot& candidate,
    const MeshSnapshot& truth,
    const Comparison& comparison) {
    const auto unique_candidate = unique_vertices(comparison.left_only_occurrences);
    const auto unique_truth = unique_vertices(comparison.right_only_occurrences);

    std::cout << "\n    " << name << '\n'
              << "      candidate occurrences:     " << comparison.left_occurrences << '\n'
              << "      truth occurrences:         " << comparison.right_occurrences << '\n'
              << "      matched occurrences:       " << comparison.matched_occurrences << '\n'
              << "      candidate-only occurrences:" << comparison.left_only_occurrences.size() << '\n'
              << "      truth-only occurrences:    " << comparison.right_only_occurrences.size() << '\n'
              << "      unique candidate-only:     " << unique_candidate.size() << '\n'
              << "      unique truth-only:         " << unique_truth.size() << '\n'
              << "      filtered candidate PBC:    "
              << candidate.filtered_periodic_boundary_occurrences << '\n'
              << "      filtered truth PBC:        "
              << truth.filtered_periodic_boundary_occurrences << '\n';

    if (!unique_candidate.empty()) {
        std::cerr << "      CANDIDATE ONLY:\n";
        for (const auto& vertex : unique_candidate) {
            std::cerr << "        sigma=" << sigma_string(vertex.sigma)
                      << " position=(" << std::setprecision(15)
                      << vertex.position[0] << ','
                      << vertex.position[1] << ','
                      << vertex.position[2] << ")\n";
        }
    }
    if (!unique_truth.empty()) {
        std::cerr << "      TRUTH ONLY:\n";
        for (const auto& vertex : unique_truth) {
            std::cerr << "        sigma=" << sigma_string(vertex.sigma)
                      << " position=(" << std::setprecision(15)
                      << vertex.position[0] << ','
                      << vertex.position[1] << ','
                      << vertex.position[2] << ")\n";
        }
    }
}

MeshSnapshot capture_high_snapshot(
    HighMesh& mesh,
    bool filter_periodic_boundary) {
    ComputeRoundState state(
        mesh.internal_node_count(),
        mesh.internal_boundary().size(),
        mesh.internal_node_count());
    ComputeMesh view(mesh, state);
    return capture_snapshot(
        view,
        mesh.internal_boundary(),
        filter_periodic_boundary);
}

bool same_boundary_geometry(const Boundary& left, const Boundary& right) {
    if (left.size() != right.size()) return false;
    for (Index i = Index{0}; i < left.size(); ++i) {
        const auto& a = left.at(i);
        const auto& b = right.at(i);
        if ((a.base() - b.base()).norm() > Scalar{1e-14}) return false;
        if ((a.normal() - b.normal()).norm() > Scalar{1e-14}) return false;
        if (a.is_periodic() != b.is_periodic()) return false;
        if (a.periodic_partner() != b.periodic_partner()) return false;
    }
    return true;
}

bool snapshot_contains_sigma(
    const MeshSnapshot& snapshot,
    const Sigma& target) {
    for (const auto& cell : snapshot.cells) {
        for (const auto& vertex : cell) {
            if (vertex.sigma == target) return true;
        }
    }
    return false;
}

std::vector<Point> concatenate_points(
    const std::vector<Point>& first,
    const std::vector<Point>& second) {
    std::vector<Point> result = first;
    result.insert(result.end(), second.begin(), second.end());
    return result;
}


// -----------------------------------------------------------------------------
// Focused A/B diagnostics.
// -----------------------------------------------------------------------------

struct FullStateReport {
    bool consistent = false;
    bool complete = false;
    std::size_t consistency_errors = 0;
    std::size_t finite_edge_endpoints = 0;
    std::size_t infinite_edges = 0;
};

template <class MeshLike>
FullStateReport verify_full_state(
    const MeshLike& mesh,
    std::string_view label) {
    const auto report = highvoronoi::verify_mesh_complete(
        mesh,
        VerificationTolerance,
        false);

    FullStateReport result;
    result.consistent = report.consistency.valid();
    result.complete = report.complete();
    result.consistency_errors = report.consistency.error_count();
    result.finite_edge_endpoints = report.unique_finite_edge_endpoints;
    result.infinite_edges = report.infinite_edges;

    std::cout << "    " << label << ":\n"
              << "      consistency:          "
              << (result.consistent ? "YES" : "NO") << '\n'
              << "      complete:             "
              << (result.complete ? "YES" : "NO") << '\n'
              << "      consistency errors:   "
              << result.consistency_errors << '\n'
              << "      finite edge endpoints:"
              << result.finite_edge_endpoints << '\n'
              << "      infinite edges:       "
              << result.infinite_edges << '\n';

    return result;
}

bool same_ordinary_nodes(
    const OrdinaryMesh& left,
    const OrdinaryMesh& right,
    Scalar tolerance = Scalar{1e-14}) {
    if (left.size() != right.size()) return false;

    Point a;
    Point b;
    for (Index i = Index{0}; i < left.size(); ++i) {
        left.nodes().copy_node(i, a.data());
        right.nodes().copy_node(i, b.data());
        if ((a - b).norm() > tolerance) return false;
    }
    return true;
}

std::size_t shared_generator_count(
    const Sigma& left,
    const Sigma& right) {
    std::size_t result = 0;
    for (const Index generator : left) {
        if (std::find(right.begin(), right.end(), generator) != right.end()) {
            ++result;
        }
    }
    return result;
}

Sigma shared_generators(
    const Sigma& left,
    const Sigma& right) {
    Sigma result;
    for (const Index generator : left) {
        if (std::find(right.begin(), right.end(), generator) != right.end()) {
            result.push_back(generator);
        }
    }
    return result;
}

bool snapshot_has_sigma(
    const MeshSnapshot& snapshot,
    const Sigma& sigma) {
    for (const auto& cell : snapshot.cells) {
        for (const auto& vertex : cell) {
            if (vertex.sigma == sigma) return true;
        }
    }
    return false;
}

std::vector<VertexRecord> unique_vertices_by_sigma(
    const MeshSnapshot& snapshot) {
    std::vector<VertexRecord> result;
    for (const auto& cell : snapshot.cells) {
        for (const auto& vertex : cell) {
            const auto found = std::find_if(
                result.begin(),
                result.end(),
                [&](const VertexRecord& existing) {
                    return existing.sigma == vertex.sigma;
                });
            if (found == result.end()) result.push_back(vertex);
        }
    }
    return result;
}

std::vector<VertexRecord> missing_topological_vertices(
    const MeshSnapshot& candidate,
    const MeshSnapshot& truth) {
    std::vector<VertexRecord> result;
    for (const auto& vertex : unique_vertices_by_sigma(truth)) {
        if (!snapshot_has_sigma(candidate, vertex.sigma)) {
            result.push_back(vertex);
        }
    }
    return result;
}

std::vector<VertexRecord> edge_candidates_in_cell(
    const MeshSnapshot& snapshot,
    Index cell,
    const Sigma& target) {
    std::vector<VertexRecord> result;
    if (cell >= static_cast<Index>(snapshot.cells.size())) return result;

    for (const auto& vertex :
         snapshot.cells[static_cast<std::size_t>(cell)]) {
        if (vertex.sigma == target) continue;
        if (shared_generator_count(vertex.sigma, target) !=
            static_cast<std::size_t>(Dimension)) {
            continue;
        }

        const auto duplicate = std::find_if(
            result.begin(),
            result.end(),
            [&](const VertexRecord& existing) {
                return existing.sigma == vertex.sigma &&
                       (existing.position - vertex.position).norm() <=
                           VertexMatchTolerance;
            });
        if (duplicate == result.end()) result.push_back(vertex);
    }
    return result;
}

std::string generator_string(
    Index generator,
    Index node_count,
    const Boundary& boundary) {
    std::ostringstream out;
    if (generator < node_count) {
        out << 'N' << generator;
        return out.str();
    }

    const Index plane = static_cast<Index>(generator - node_count);
    out << 'B' << plane;
    if (plane < boundary.size()) {
        out << (boundary.at(plane).is_periodic()
                    ? "[periodic]"
                    : "[nonperiodic]");
    }
    return out.str();
}

void print_edge_candidate(
    const VertexRecord& candidate,
    const Sigma& target,
    Index node_count,
    const Boundary& boundary) {
    const Sigma shared = shared_generators(candidate.sigma, target);

    std::cout << "          sigma=" << sigma_string(candidate.sigma)
              << " shared=" << sigma_string(shared)
              << " position=(" << std::setprecision(15)
              << candidate.position[0] << ','
              << candidate.position[1] << ','
              << candidate.position[2] << ')';

    if (touches_periodic_boundary(
            candidate.sigma,
            node_count,
            boundary)) {
        std::cout << "  [PERIODIC-BOUNDARY VERTEX]";
    }

    std::cout << '\n';
}

void diagnose_missing_vertex_neighbours(
    const MeshSnapshot& refined,
    const VertexRecord& target,
    Index node_count,
    const Boundary& boundary) {
    std::cout << "\n    MISSING TARGET\n"
              << "      sigma=" << sigma_string(target.sigma)
              << " position=(" << std::setprecision(15)
              << target.position[0] << ','
              << target.position[1] << ','
              << target.position[2] << ")\n";

    for (const Index generator : target.sigma) {
        if (generator >= node_count) {
            std::cout << "      "
                      << generator_string(generator, node_count, boundary)
                      << " is a boundary constraint and has no ordinary cell.\n";
            continue;
        }

        const auto candidates =
            edge_candidates_in_cell(refined, generator, target.sigma);

        std::cout << "      cell " << generator
                  << ": vertices sharing exactly 3 target generators = "
                  << candidates.size() << '\n';

        for (const auto& candidate : candidates) {
            print_edge_candidate(
                candidate,
                target.sigma,
                node_count,
                boundary);
        }
    }
}


} // namespace

int main() {
    section("Ordinary periodic-boundary refine: focused A/B edge-neighbour diagnosis");
    std::cout
        << "A = all block-2 nodes at once, full ordinary ComputeVoronoi.\n"
        << "B0 = first 27 nodes, full ordinary ComputeVoronoi.\n"
        << "B1 = B0 plus the exact first HighVoronoi reference block through\n"
        << "     the ordinary RefineVoronoi core; only the production guard\n"
        << "     against periodic boundaries is omitted in this test.\n"
        << "A, B0 and B1 use one fixed, identical final internal boundary.\n";

    const std::vector<Point> initial_points = make_initial_visible_points();

    // -------------------------------------------------------------------------
    // 1. Derive the exact first reference block and the boundary that is already
    //    final before B0 is constructed.
    // -------------------------------------------------------------------------
    section("1. Derive exact block 2 and fixed boundary");

    auto high_control = make_empty_high_mesh();
    append_visible_points(*high_control, initial_points);

    const std::vector<Index> block1 =
        high_control->pending_new_visible_internal_nodes();
    check(block1.size() == static_cast<std::size_t>(InitialVisibleNodeCount),
          "control HighVoronoi starts from exactly 27 visible NEW nodes");

    HighRefiner first_high_refine(
        *high_control,
        highvoronoi::existing_incremental_nodes,
        block1,
        Index{0},
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters(),
        edge_parameters());
    (void)first_high_refine.compute();

    const auto planned = materialize_periodic_requests(
        *high_control,
        first_high_refine.periodic_requests());
    const std::vector<Point> block2_points = planned_points(planned);

    check(!block2_points.empty(),
          "first HighVoronoi step produces a non-empty reference block");

    (void)high_control->expand_internal_boundary_to_include(
        block2_points,
        BoundaryTolerance);
    const Boundary fixed_boundary = high_control->internal_boundary();

    const std::vector<Point> all_points =
        concatenate_points(initial_points, block2_points);

    std::cout << "    initial nodes: " << initial_points.size() << '\n'
              << "    block-2 nodes:" << block2_points.size() << '\n'
              << "    final nodes:  " << all_points.size() << '\n';

    // -------------------------------------------------------------------------
    // 2. A -- instant full reference.
    // -------------------------------------------------------------------------
    section("2. A: instant full ordinary VoronoiMesh");

    auto instant = make_ordinary_mesh(all_points, fixed_boundary);
    compute_full(*instant);

    const FullStateReport a_state =
        verify_full_state(*instant, "A full state");
    check(a_state.consistent, "A is geometrically consistent");
    check(a_state.complete, "A is complete");

    // -------------------------------------------------------------------------
    // 3. B0 -- the exact old state used by the refine.
    // -------------------------------------------------------------------------
    section("3. B0: 27-node full ordinary state on the SAME fixed boundary");

    auto refined = make_ordinary_mesh(initial_points, fixed_boundary);
    compute_full(*refined);

    const FullStateReport b0_state =
        verify_full_state(*refined, "B0 before refine");
    check(b0_state.consistent, "B0 is geometrically consistent");
    check(b0_state.complete, "B0 is complete");

    check(same_boundary_geometry(refined->boundary(), instant->boundary()),
          "A and B0 use identical boundary geometry and conditions");

    // -------------------------------------------------------------------------
    // 4. B1 -- same ordinary incremental core as production RefineVoronoi,
    //    except that this diagnostic omits only Backend::validate(mesh), because
    //    production deliberately rejects periodic ordinary boundaries.
    // -------------------------------------------------------------------------
    section("4. B1: append exact block 2 with ordinary refine core");

    PeriodicOrdinaryRefiner refiner(*refined, block2_points);
    refiner.compute();

    std::cout << "    refine report: appended="
              << refiner.report().appended_nodes
              << " NEW vertices=" << refiner.report().new_cell_vertices
              << " affected=" << refiner.report().affected_nodes
              << " invalidated=" << refiner.report().invalidated_old_vertices
              << '\n';

    check(refined->size() == instant->size(),
          "A and B1 have identical node counts");
    check(same_ordinary_nodes(*refined, *instant),
          "A and B1 have identical node coordinates in identical numbering");
    check(same_boundary_geometry(refined->boundary(), instant->boundary()),
          "A and B1 still use the identical fixed boundary");

    const FullStateReport b1_state =
        verify_full_state(*refined, "B1 after refine");
    check(b1_state.consistent, "B1 is geometrically consistent");
    check(b1_state.complete, "B1 is complete");

    // -------------------------------------------------------------------------
    // 5. Topological difference only: sigma presence, not position tolerance.
    //    For every sigma that exists in A but nowhere in B1, inspect the B1
    //    cells belonging to each target generator and print every vertex that
    //    shares exactly three generators with the missing target.
    // -------------------------------------------------------------------------
    section("5. Missing A vertices and their 3-generator neighbours in B1");

    const MeshSnapshot a_raw =
        capture_snapshot(*instant, fixed_boundary, false);
    const MeshSnapshot b_raw =
        capture_snapshot(*refined, fixed_boundary, false);

    const auto missing =
        missing_topological_vertices(b_raw, a_raw);

    std::cout << "    unique topological vertices present in A but absent in B1: "
              << missing.size() << '\n';

    for (const auto& target : missing) {
        diagnose_missing_vertex_neighbours(
            b_raw,
            target,
            refined->size(),
            fixed_boundary);
    }

    section("Result");
    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n'
              << "diagnosed missing topological vertices: "
              << missing.size() << '\n';

    // Missing vertices are the object of this diagnostic, not a test harness
    // failure. Setup/full-state invariants determine the exit code.
    return failed_checks == 0 ? 0 : 1;
}




