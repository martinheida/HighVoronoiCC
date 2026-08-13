#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <highvoronoi/geometry/boundary.hpp>
#include <highvoronoi/geometry/voronoi_nodes.hpp>

namespace {

using highvoronoi::Boundary;
using highvoronoi::Dirichlet;
using highvoronoi::Dynamic;
using highvoronoi::DynamicPoint;
using highvoronoi::Neumann;
using highvoronoi::Periodic;
using highvoronoi::StaticPoint;
using highvoronoi::VoronoiNodes;

constexpr double tolerance = 1e-12;

bool close(double left, double right) {
    return std::abs(left - right) <= tolerance;
}

template <typename LeftDerived, typename RightDerived>
bool close_point(const Eigen::MatrixBase<LeftDerived>& left,
                 const Eigen::MatrixBase<RightDerived>& right) {
    return left.size() == right.size() &&
           (left - right).norm() <= tolerance;
}

void test_reflection_and_projection() {
    StaticPoint<double, 2> base;
    base << 1.0, 0.0;

    StaticPoint<double, 2> normal;
    normal << 2.0, 0.0; // deliberately not normalized

    StaticPoint<double, 2> point;
    point << 3.0, 4.0;

    Dirichlet<2> plane(base, normal);

    const auto reflected = plane.reflect(point);
    const auto projected = plane.project(point);

    static_assert(std::is_same_v<
                  std::remove_cv_t<decltype(reflected)>,
                  StaticPoint<double, 2>>);
    static_assert(std::is_same_v<
                  std::remove_cv_t<decltype(projected)>,
                  StaticPoint<double, 2>>);

    StaticPoint<double, 2> expected_reflected;
    expected_reflected << -1.0, 4.0;

    StaticPoint<double, 2> expected_projected;
    expected_projected << 1.0, 4.0;

    assert(close_point(reflected, expected_reflected));
    assert(close_point(projected, expected_projected));
    assert(close(plane.halfspace_value(point), 2.0));
    assert(close(plane.signed_distance(point), 2.0));
}

void test_boundary_and_intersection() {
    StaticPoint<double, 2> dimensions;
    dimensions << 2.0, 2.0;

    const Boundary<2> boundary = Boundary<2>::cuboid(
        dimensions,
        std::vector<std::size_t>{}); // no periodic axes

    StaticPoint<double, 2> inside;
    inside << 0.5, 0.5;

    StaticPoint<double, 2> outside;
    outside << 3.0, 0.25;

    assert(boundary.contains(inside));
    assert(!boundary.contains(outside));

    const auto projected = boundary.project_inside(outside);
    assert(boundary.contains(projected, tolerance));
    assert(close(projected[0], 2.0));

    StaticPoint<double, 2> direction;
    direction << 1.0, 0.0;

    const auto hit = boundary.first_intersection(inside, direction);
    assert(hit);
    assert(close(hit->parameter, 1.5));

    const auto hit_point = boundary.intersection_point(inside, direction);
    assert(hit_point);

    StaticPoint<double, 2> expected_hit;
    expected_hit << 2.0, 0.5;
    assert(close_point(*hit_point, expected_hit));
    assert(boundary.intersection_exists(inside, direction));
}

void test_periodic_indices_and_conversion() {
    StaticPoint<double, 2> upper;
    upper << 1.0, 0.0;

    StaticPoint<double, 2> lower;
    lower << 0.0, 0.0;

    StaticPoint<double, 2> normal;
    normal << 1.0, 0.0;

    Boundary<2> boundary;
    boundary.add(Periodic<2>(upper, lower, normal));

    assert(boundary.size() == 2);
    assert(boundary[0].periodic_partner() == 1);
    assert(boundary[1].periodic_partner() == 0);
    assert(close_point(boundary[1].normal(), -normal));

    const auto dynamic_boundary = boundary.to_dynamic();
    static_assert(std::is_same_v<
                  std::remove_cv_t<decltype(dynamic_boundary)>,
                  Boundary<Dynamic, double, std::size_t>>);

    assert(dynamic_boundary.dimension() == 2);
    assert(dynamic_boundary[0].periodic_partner() == 1);
    assert(dynamic_boundary[1].periodic_partner() == 0);

    const auto restored = dynamic_boundary.to_static<2>();
    static_assert(std::is_same_v<
                  std::remove_cv_t<decltype(restored)>,
                  Boundary<2, double, std::size_t>>);
    assert(restored.size() == 2);

    bool wrong_dimension_threw = false;
    try {
        (void)dynamic_boundary.to_static<3>();
    } catch (const std::invalid_argument&) {
        wrong_dimension_threw = true;
    }
    assert(wrong_dimension_threw);
}

void test_dynamic_default_and_deduction() {
    Boundary<> empty_dynamic_boundary;

    DynamicPoint<double> arbitrary_point(4);
    arbitrary_point.setZero();
    assert(empty_dynamic_boundary.contains(arbitrary_point));

    DynamicPoint<double> base(2);
    base << 0.0, 0.0;

    DynamicPoint<double> normal(2);
    normal << 0.0, 1.0;

    Dirichlet<> dynamic_plane(base, normal);
    const auto dynamic_result = dynamic_plane.reflect(base);
    static_assert(std::is_same_v<
                  std::remove_cv_t<decltype(dynamic_result)>,
                  DynamicPoint<double>>);

    StaticPoint<double, 2> static_base;
    static_base << 0.0, 0.0;

    StaticPoint<double, 2> static_normal;
    static_normal << 0.0, 1.0;

    Dirichlet deduced_plane(static_base, static_normal);
    static_assert(decltype(deduced_plane)::DimensionAtCompileTime == 2);

    const auto static_result = deduced_plane.reflect(static_base);
    static_assert(std::is_same_v<
                  std::remove_cv_t<decltype(static_result)>,
                  StaticPoint<double, 2>>);
}

void test_voronoi_nodes() {
    Boundary<2> boundary = Boundary<2>::centered_cube(2.0);

    VoronoiNodes<double, 2> nodes(2);

    StaticPoint<double, 2> first;
    first << 0.0, 0.0;

    StaticPoint<double, 2> second;
    second << 0.5, -0.5;

    nodes.set(0, first);
    nodes.set(1, second);
    boundary.check_nodes(nodes);

    second << 2.0, 0.0;
    nodes.set(1, second);

    bool outside_node_threw = false;
    try {
        boundary.check_nodes(nodes);
    } catch (const std::domain_error&) {
        outside_node_threw = true;
    }
    assert(outside_node_threw);
}

} // namespace

int main() {
    test_reflection_and_projection();
    test_boundary_and_intersection();
    test_periodic_indices_and_conversion();
    test_dynamic_default_and_deduction();
    test_voronoi_nodes();
}
