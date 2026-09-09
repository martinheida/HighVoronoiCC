#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/spherical_voronoi_mesh.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database3 = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters,
    3>;
using Database4 = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters,
    4>;

using SphereMesh = highvoronoi::SphereVoronoiMesh<Scalar, 3, Database3>;
using SpherePoint = SphereMesh::NodePoint;
using RotationMesh = highvoronoi::AntipodalSphericalVoronoiMesh<
    Scalar,
    4,
    Database4>;
using RotationPoint = RotationMesh::NodePoint;

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

template <class Database>
std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        16384,
        DatabaseParameters{highvoronoi::DirectHash{16384}});
}

highvoronoi::RaycastParameters<
    highvoronoi::ClassicRaycast,
    Scalar> ray_parameters() {
    highvoronoi::RaycastParameters<
        highvoronoi::ClassicRaycast,
        Scalar> parameters;
    parameters.variance_tolerance = Scalar{1e-12};
    return parameters;
}

template <class PointLike>
bool canonical_hemisphere(const PointLike& point) {
    for (Eigen::Index coordinate = 0;
         coordinate < point.size();
         ++coordinate) {
        if (point[coordinate] > Scalar{0}) {
            return true;
        }
        if (point[coordinate] < Scalar{0}) {
            return false;
        }
    }
    return true;
}

template <class A, class B>
Scalar squared_distance(const A& a, const B& b) {
    return (a.template cast<Scalar>() - b.template cast<Scalar>()).squaredNorm();
}

template <class A, class B>
Scalar projective_squared_distance(const A& a, const B& b) {
    const auto aa = a.template cast<Scalar>();
    const auto bb = b.template cast<Scalar>();
    return std::min(
        (aa - bb).squaredNorm(),
        (aa + bb).squaredNorm());
}

std::vector<SpherePoint> make_fibonacci_sphere(Index count) {
    constexpr Scalar pi = Scalar{3.141592653589793238462643383279502884};
    const Scalar golden_angle = pi * (Scalar{3} - std::sqrt(Scalar{5}));

    std::vector<SpherePoint> points;
    points.reserve(static_cast<std::size_t>(count));

    for (Index i = Index{0}; i < count; ++i) {
        const Scalar z = Scalar{1} -
            Scalar{2} * (static_cast<Scalar>(i) + Scalar{0.5}) /
            static_cast<Scalar>(count);
        const Scalar radius = std::sqrt(std::max(Scalar{0}, Scalar{1} - z * z));
        const Scalar phi = golden_angle * static_cast<Scalar>(i);

        SpherePoint point;
        point << radius * std::cos(phi), radius * std::sin(phi), z;
        points.push_back(point);
    }
    return points;
}

std::vector<RotationPoint> make_rotation_points(Index count) {
    std::mt19937_64 random(0x5350484552494341ULL);
    std::normal_distribution<Scalar> normal(Scalar{0}, Scalar{1});

    std::vector<RotationPoint> points;
    points.reserve(static_cast<std::size_t>(count));

    for (Index i = Index{0}; i < count; ++i) {
        RotationPoint point;
        do {
            for (int coordinate = 0; coordinate < 4; ++coordinate) {
                point[coordinate] = normal(random);
            }
        } while (point.norm() < Scalar{0.2});

        point.normalize();
        // Exercise the public canonicalization deliberately: roughly half of
        // the user inputs arrive in the "wrong" hemisphere.
        if ((i % Index{2}) == Index{0}) {
            point = -point;
        }
        points.push_back(point);
    }
    return points;
}

void validate_sphere_mesh(const SphereMesh& mesh, std::string_view stage) {
    const auto vertices = mesh.vertices();
    check(!vertices.empty(), "S2 has finite spherical vertices");

    const Scalar norm_tolerance = Scalar{2e-9};
    const Scalar voronoi_tolerance = Scalar{3e-7};

    std::unordered_set<std::size_t> addresses;
    bool unit_vertices = true;
    bool support_equal = true;
    bool globally_nearest = true;
    bool signatures_valid = true;
    bool unique_addresses = true;

    for (const auto& vertex : vertices) {
        unit_vertices = unit_vertices &&
            std::abs(vertex.position.norm() - Scalar{1}) <= norm_tolerance;
        signatures_valid = signatures_valid && vertex.sigma.size() >= 3;
        unique_addresses = unique_addresses && addresses.insert(vertex.address).second;

        if (vertex.sigma.empty()) {
            continue;
        }

        const Scalar reference_distance = squared_distance(
            vertex.position,
            mesh.node(vertex.sigma.front()));

        for (const Index generator : vertex.sigma) {
            const Scalar distance = squared_distance(
                vertex.position,
                mesh.node(generator));
            support_equal = support_equal &&
                std::abs(distance - reference_distance) <= voronoi_tolerance;
        }

        for (Index generator = Index{0}; generator < mesh.size(); ++generator) {
            const Scalar distance = squared_distance(
                vertex.position,
                mesh.node(generator));
            globally_nearest = globally_nearest &&
                distance + voronoi_tolerance >= reference_distance;
        }
    }

    check(unit_vertices, "S2 vertices are radially projected to the unit sphere");
    check(signatures_valid, "S2 public signatures contain at least three generators");
    check(support_equal, "S2 vertex support generators have equal spherical chord distance");
    check(globally_nearest, "S2 no visible generator is closer than the support generators");
    check(unique_addresses, "S2 public vertex view emits each center-cell record once");

    std::cout << "        " << stage << ": nodes=" << mesh.size()
              << " vertices=" << vertices.size() << '\n';
}

void validate_rotation_mesh(const RotationMesh& mesh, std::string_view stage) {
    const auto vertices = mesh.vertices();
    check(!vertices.empty(), "S3/+ has finite public quotient vertices");

    const Scalar norm_tolerance = Scalar{2e-9};
    const Scalar voronoi_tolerance = Scalar{5e-7};

    bool nodes_canonical = true;
    bool nodes_unit = true;
    for (Index node = Index{0}; node < mesh.size(); ++node) {
        const RotationPoint point = mesh.node(node);
        nodes_canonical = nodes_canonical && canonical_hemisphere(point);
        nodes_unit = nodes_unit &&
            std::abs(point.norm() - Scalar{1}) <= norm_tolerance;
    }

    bool vertices_canonical = true;
    bool vertices_unit = true;
    bool support_equal = true;
    bool globally_nearest = true;
    bool signatures_valid = true;

    for (const auto& vertex : vertices) {
        vertices_canonical = vertices_canonical &&
            canonical_hemisphere(vertex.position);
        vertices_unit = vertices_unit &&
            std::abs(vertex.position.norm() - Scalar{1}) <= norm_tolerance;
        signatures_valid = signatures_valid && !vertex.sigma.empty();

        if (vertex.sigma.empty()) {
            continue;
        }

        const Scalar reference_distance = projective_squared_distance(
            vertex.position,
            mesh.node(vertex.sigma.front()));

        for (const Index generator : vertex.sigma) {
            const Scalar distance = projective_squared_distance(
                vertex.position,
                mesh.node(generator));
            support_equal = support_equal &&
                std::abs(distance - reference_distance) <= voronoi_tolerance;
        }

        for (Index generator = Index{0}; generator < mesh.size(); ++generator) {
            const Scalar distance = projective_squared_distance(
                vertex.position,
                mesh.node(generator));
            globally_nearest = globally_nearest &&
                distance + voronoi_tolerance >= reference_distance;
        }
    }

    bool internal_pairs_consistent = true;
    for (Index public_node = Index{0}; public_node < mesh.size(); ++public_node) {
        const Index visible = mesh.public_node_to_internal(public_node);
        const Index antipode = static_cast<Index>(visible + Index{1});
        internal_pairs_consistent = internal_pairs_consistent &&
            mesh.reference_internal_node(visible) == visible &&
            mesh.reference_internal_node(antipode) == visible;
    }

    check(nodes_unit, "S3/+ user nodes are normalized");
    check(nodes_canonical, "S3/+ user nodes are canonicalized to the visible hemisphere");
    check(vertices_unit, "S3/+ public vertices lie on the unit S3");
    check(vertices_canonical, "S3/+ public vertices lie in the canonical visible hemisphere");
    check(signatures_valid, "S3/+ public vertex signatures are non-empty");
    check(support_equal, "S3/+ support generators have equal projective distance");
    check(globally_nearest, "S3/+ no visible rotation generator is projectively closer");
    check(internal_pairs_consistent, "S3/+ every invisible antipode references its visible master");

    std::size_t infinite_edges = 0;
    for ([[maybe_unused]] const auto& edge :
         mesh.construction_mesh().infinite_edges()) {
        ++infinite_edges;
    }
    check(infinite_edges == 0, "S3/+ internal origin cell stores no infinite edges");

    std::cout << "        " << stage << ": public nodes=" << mesh.size()
              << " public vertices=" << vertices.size() << '\n';
}

void test_s2_neighbours_keep_internal_origin() {
    std::cout << "\n============================================================\n"
              << "SphericalVoronoiMesh: S2 neighbour projection\n"
              << "============================================================\n";

    std::vector<SpherePoint> nodes;
    nodes.reserve(6);
    for (const auto& values : std::vector<std::array<Scalar, 3>>{
             {Scalar{1}, Scalar{0}, Scalar{0}},
             {Scalar{-1}, Scalar{0}, Scalar{0}},
             {Scalar{0}, Scalar{1}, Scalar{0}},
             {Scalar{0}, Scalar{-1}, Scalar{0}},
             {Scalar{0}, Scalar{0}, Scalar{1}},
             {Scalar{0}, Scalar{0}, Scalar{-1}}}) {
        SpherePoint point;
        point << values[0], values[1], values[2];
        nodes.push_back(point);
    }

    SphereMesh mesh(std::move(nodes), make_database<Database3>());
    mesh.compute(
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    std::vector<Index> public_neighbours;
    std::vector<Index> internal_neighbours;

    check(mesh.dirty(Index{0}),
          "S2 neighbour state starts dirty after geometry computation");
    check(!mesh.neighbours(Index{0}, public_neighbours),
          "S2 stale public neighbour read reports dirty state");
    check(public_neighbours.empty(),
          "S2 has no published public neighbour record before explicit compute");

    const std::array<Index, 6> opposite{
        Index{1}, Index{0}, Index{3}, Index{2}, Index{5}, Index{4}};

    bool all_public_lists_exact = true;
    bool all_internal_lists_keep_origin = true;
    bool all_internal_counts_exact = true;
    bool all_opposites_rejected = true;
    bool all_cells_clean_after_compute = true;

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        mesh.compute_neighbors(cell);
        all_cells_clean_after_compute =
            all_cells_clean_after_compute && !mesh.dirty(cell);

        public_neighbours.clear();
        all_public_lists_exact =
            all_public_lists_exact && mesh.neighbours(cell, public_neighbours);

        std::vector<Index> expected_public;
        for (Index candidate = Index{0}; candidate < mesh.size(); ++candidate) {
            if (candidate != cell && candidate != opposite[cell]) {
                expected_public.push_back(candidate);
            }
        }
        all_public_lists_exact =
            all_public_lists_exact && public_neighbours == expected_public;

        const Index internal_cell = mesh.public_node_to_internal(cell);
        internal_neighbours.clear();
        const bool has_internal_record =
            mesh.construction_mesh().internal_neighbours(
                internal_cell,
                internal_neighbours);
        all_internal_lists_keep_origin =
            all_internal_lists_keep_origin && has_internal_record &&
            std::binary_search(
                internal_neighbours.begin(),
                internal_neighbours.end(),
                Index{0});
        all_internal_counts_exact =
            all_internal_counts_exact && internal_neighbours.size() == 5;
        all_opposites_rejected =
            all_opposites_rejected &&
            std::find(
                internal_neighbours.begin(),
                internal_neighbours.end(),
                mesh.public_node_to_internal(opposite[cell])) ==
                internal_neighbours.end();
    }

    check(all_cells_clean_after_compute,
          "S2 explicit compute_neighbors clears every tested public dirty state");
    check(all_internal_lists_keep_origin,
          "S2 every internal neighbour record deliberately retains origin node 0");
    check(all_internal_counts_exact,
          "S2 every axis cell has origin plus four non-origin internal neighbours");
    check(all_opposites_rejected,
          "S2 every axis cell rejects its opposite generator as an internal neighbour");
    check(all_public_lists_exact,
          "S2 every axis cell exposes exactly its four spherical neighbours");
}

void test_antipodal_neighbour_projection_preserves_multiplicity() {
    std::cout << "\n============================================================\n"
              << "SphericalVoronoiMesh: antipodal neighbour projection\n"
              << "============================================================\n";

    std::vector<RotationPoint> nodes;
    nodes.reserve(4);
    for (int axis = 0; axis < 4; ++axis) {
        RotationPoint point = RotationPoint::Zero();
        point[axis] = Scalar{1};
        nodes.push_back(point);
    }

    RotationMesh mesh(std::move(nodes), make_database<Database4>());
    mesh.compute(
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    std::vector<Index> public_neighbours;
    std::vector<Index> visible_internal_neighbours;
    std::vector<Index> antipode_internal_neighbours;

    bool all_public_lists_exact = true;
    bool all_visible_internal_records = true;
    bool all_antipode_internal_records = true;
    bool all_internal_lists_keep_origin = true;
    bool all_internal_counts_exact = true;

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        mesh.compute_neighbors(cell);

        public_neighbours.clear();
        all_public_lists_exact =
            all_public_lists_exact && mesh.neighbours(cell, public_neighbours);

        std::vector<Index> expected_public;
        for (Index candidate = Index{0}; candidate < mesh.size(); ++candidate) {
            if (candidate != cell) {
                expected_public.push_back(candidate);
                expected_public.push_back(candidate);
            }
        }
        all_public_lists_exact =
            all_public_lists_exact && public_neighbours == expected_public;

        const Index visible_internal = mesh.public_node_to_internal(cell);
        const Index antipode_internal =
            static_cast<Index>(visible_internal + Index{1});

        visible_internal_neighbours.clear();
        antipode_internal_neighbours.clear();
        const bool visible_record =
            mesh.construction_mesh().internal_neighbours(
                visible_internal,
                visible_internal_neighbours);
        const bool antipode_record =
            mesh.construction_mesh().internal_neighbours(
                antipode_internal,
                antipode_internal_neighbours);

        all_visible_internal_records =
            all_visible_internal_records && visible_record;
        all_antipode_internal_records =
            all_antipode_internal_records && antipode_record;
        all_internal_lists_keep_origin =
            all_internal_lists_keep_origin &&
            std::binary_search(
                visible_internal_neighbours.begin(),
                visible_internal_neighbours.end(),
                Index{0}) &&
            std::binary_search(
                antipode_internal_neighbours.begin(),
                antipode_internal_neighbours.end(),
                Index{0});
        all_internal_counts_exact =
            all_internal_counts_exact &&
            visible_internal_neighbours.size() == 7 &&
            antipode_internal_neighbours.size() == 7;
    }

    check(all_public_lists_exact,
          "S3/+ projects +/- neighbours to the same public cells without deduplication");
    check(all_visible_internal_records,
          "S3/+ every visible internal representative stores its neighbour record");
    check(all_antipode_internal_records,
          "S3/+ compute_neighbors also computes every antipodal internal representative");
    check(all_internal_lists_keep_origin,
          "S3/+ all internal representative lists retain origin node 0");
    check(all_internal_counts_exact,
          "S3/+ each cross-polytope representative has origin plus six internal neighbours");
}

void test_s2_initial_refine_remove() {
    std::cout << "\n============================================================\n"
              << "SphericalVoronoiMesh: S2 initial/refine/remove\n"
              << "============================================================\n";

    std::vector<SpherePoint> initial = make_fibonacci_sphere(Index{20});
    std::vector<SpherePoint> added;
    added.reserve(4);
    for (const auto& values : std::vector<std::array<Scalar, 3>>{
             {Scalar{1}, Scalar{1}, Scalar{1}},
             {Scalar{-1}, Scalar{1}, Scalar{-1}},
             {Scalar{1}, Scalar{-1}, Scalar{-1}},
             {Scalar{-1}, Scalar{-1}, Scalar{1}}}) {
        SpherePoint point;
        point << values[0], values[1], values[2];
        point.normalize();
        added.push_back(point);
    }

    SphereMesh mesh(std::move(initial), make_database<Database3>());
    mesh.compute(
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());
    check(mesh.size() == Index{20}, "S2 initial mesh contains exactly 20 visible nodes");
    validate_sphere_mesh(mesh, "initial 20");

    const auto refine_report = mesh.refine(
        std::move(added),
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    check(mesh.size() == Index{24}, "S2 refine appends four visible nodes");
    check(refine_report.appended_public_nodes == 4,
          "S2 refine reports four appended public nodes");
    validate_sphere_mesh(mesh, "after refine to 24");

    const auto remove_report = mesh.remove(
        std::vector<Index>{Index{20}, Index{21}, Index{22}, Index{23}},
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    check(mesh.size() == Index{20}, "S2 remove returns to the original 20 visible nodes");
    check(remove_report.removed_internal_nodes == 4,
          "S2 remove deletes exactly four internal visible generators");
    validate_sphere_mesh(mesh, "after remove back to 20");
}

void test_s3_antipodal_initial_refine_remove() {
    std::cout << "\n============================================================\n"
              << "SphericalVoronoiMesh: S3 antipodal / SO(3) quotient\n"
              << "============================================================\n";

    const std::vector<RotationPoint> all_points = make_rotation_points(Index{14});
    std::vector<RotationPoint> initial(all_points.begin(), all_points.begin() + 10);
    std::vector<RotationPoint> added(all_points.begin() + 10, all_points.end());

    RotationMesh mesh(std::move(initial), make_database<Database4>());
    check(
        mesh.construction_mesh().internal_nodes().size() == Index{21},
        "S3/+ initial construction contains origin + 10 visible + 10 antipodes");

    mesh.compute(
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());
    validate_rotation_mesh(mesh, "initial 10");

    const auto refine_report = mesh.refine(
        std::move(added),
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    check(mesh.size() == Index{14}, "S3/+ refine reaches 14 visible rotation nodes");
    check(refine_report.appended_internal_nodes == 8,
          "S3/+ four visible refine nodes append four invisible antipodes");
    validate_rotation_mesh(mesh, "after refine to 14");

    const auto remove_report = mesh.remove(
        std::vector<Index>{Index{2}},
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    check(mesh.size() == Index{13}, "S3/+ remove deletes one public rotation node");
    check(remove_report.removed_internal_nodes == 2,
          "S3/+ remove deletes visible generator and invisible antipode together");
    validate_rotation_mesh(mesh, "after remove");
}

} // namespace

int main() {
    test_s2_neighbours_keep_internal_origin();
    test_antipodal_neighbour_projection_preserves_multiplicity();
    test_s2_initial_refine_remove();
    test_s3_antipodal_initial_refine_remove();

    std::cout << "\n============================================================\n"
              << "SphericalVoronoiMesh summary\n"
              << "============================================================\n"
              << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}
