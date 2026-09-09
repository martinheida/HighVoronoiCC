#include <highvoronoi/algorithm/detail/float.hpp>
#include <highvoronoi/algorithm/normal_solver.hpp>
#include <highvoronoi/core/voronoi_nodes.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

// ============================================================================
// Global test types
// ============================================================================

using Scalar = double;
using Index = std::uint32_t;

inline constexpr int Dimension = 4;
inline constexpr Index NodeCount = Index{Dimension + 1};
inline constexpr Scalar Tolerance = 1.0e-12;

using Point = highvoronoi::StaticPoint<Scalar, Dimension>;
using Nodes = highvoronoi::VoronoiNodes<Scalar, Dimension, Index>;
using Sigma = std::vector<Index>;

using NormalSolver =
    highvoronoi::QRNormalSolver<Scalar, Dimension>;

using Float128NormalSolver =
    highvoronoi::ExtendedQRNormalSolver<Scalar, Dimension>;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

// ============================================================================
// Small test helpers
// ============================================================================

void check(bool condition, std::string_view description) {
    ++performed_checks;

    if (condition) {
        std::cout << "        [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "        [FAIL] " << description << '\n';
}

void print_point(const Point& point) {
    std::cout << '(';
    for (int k = 0; k < Dimension; ++k) {
        if (k != 0) {
            std::cout << ", ";
        }
        std::cout << point[k];
    }
    std::cout << ')';
}

Point load_point(const Nodes& nodes, Index index) {
    Point point;
    nodes.copy_node(index, point.data());
    return point;
}

Sigma make_sigma() {
    Sigma sigma;
    sigma.reserve(NodeCount);

    for (Index i = Index{0}; i < NodeCount; ++i) {
        sigma.push_back(i);
    }

    return sigma;
}

Nodes make_random_nodes() {
    Nodes nodes(NodeCount);

    // Fixed seed: random geometry, reproducible shell output.
    std::mt19937_64 generator(0x485651524E4F524DULL);
    std::normal_distribution<Scalar> distribution(Scalar{0}, Scalar{1});

    for (Index i = Index{0}; i < NodeCount; ++i) {
        Point point;

        do {
            for (int k = 0; k < Dimension; ++k) {
                point[k] = distribution(generator);
            }
        } while (point.norm() < Scalar{1.0e-8});

        // All nodes lie at distance 1 from r = 0. This is exactly the
        // Voronoi-vertex geometry used by the edge-direction calculation.
        point.normalize();
        nodes.set(i, point);
    }

    return nodes;
}

void print_nodes(const Nodes& nodes) {
    std::cout << "\nRandom Voronoi nodes (all |x_i| = 1):\n";

    for (Index i = Index{0}; i < NodeCount; ++i) {
        const Point point = load_point(nodes, i);
        std::cout << "    x[" << i << "] = ";
        print_point(point);
        std::cout << "    |x| = " << point.norm() << '\n';
    }
}

void check_orthogonality(
    const Nodes& nodes,
    Index omitted,
    const Point& u) {

    std::cout << "\n    omitted index = " << omitted << '\n';
    std::cout << "    u = ";
    print_point(u);
    std::cout << "    |u| = " << u.norm() << '\n';

    check(
        std::abs(u.norm() - Scalar{1}) <= Tolerance,
        "u is normalized");

    Scalar maximum_error = Scalar{0};

    for (Index i = Index{0}; i < NodeCount; ++i) {
        if (i == omitted) {
            continue;
        }

        for (Index j = i + Index{1}; j < NodeCount; ++j) {
            if (j == omitted) {
                continue;
            }

            const Point x = load_point(nodes, i);
            const Point y = load_point(nodes, j);
            const Point difference = x - y;
            const Scalar product = u.dot(difference);

            maximum_error = std::max(maximum_error, std::abs(product));

            std::cout << "        u * (x[" << i << "] - x[" << j << "])\n";
            std::cout << "            u       = ";
            print_point(u);
            std::cout << '\n';
            std::cout << "            x - y   = ";
            print_point(difference);
            std::cout << '\n';
            std::cout << "            product = " << product << '\n';

            check(
                std::abs(product) <= Tolerance,
                "u * (x_i - x_j) is zero within tolerance");
        }
    }

    std::cout << "        maximum |u * (x_i - x_j)| = "
              << maximum_error << '\n';
}

// ============================================================================
// Actual tests: deliberately no template notation here
// ============================================================================

void test_normal_double_qr(const Nodes& nodes, const Sigma& sigma) {
    std::cout << "\n============================================================\n";
    std::cout << "QR NORMAL SOLVER: double precision\n";
    std::cout << "============================================================\n";

    NormalSolver solver(Dimension);

    for (Index omitted = Index{0}; omitted < NodeCount; ++omitted) {
        Point u;

        const bool success = solver.solve(
            nodes,
            sigma,
            static_cast<std::size_t>(omitted),
            u);

        check(success, "QR solver returned a normal");
        if (!success) {
            continue;
        }

        check_orthogonality(nodes, omitted, u);
    }
}

void test_float128_qr(const Nodes& nodes, const Sigma& sigma) {
    std::cout << "\n============================================================\n";
    std::cout << "QR NORMAL SOLVER: extended / Float128 computation\n";
    std::cout << "============================================================\n";

    Float128NormalSolver solver(Dimension);

    for (Index omitted = Index{0}; omitted < NodeCount; ++omitted) {
        Point u;

        const bool success = solver.solve(
            nodes,
            sigma,
            static_cast<std::size_t>(omitted),
            u);

        check(success, "Float128 QR solver returned a normal");
        if (!success) {
            continue;
        }

        check_orthogonality(nodes, omitted, u);
    }
}

void print_precision_information() {
    std::cout << "============================================================\n";
    std::cout << "PRECISION INFORMATION\n";
    std::cout << "============================================================\n";

    std::cout << "    Scalar = double\n";
    std::cout << "    double digits10            = "
              << std::numeric_limits<Scalar>::digits10 << '\n';
    std::cout << "    ExtendedFloat digits10     = "
              << std::numeric_limits<highvoronoi::ExtendedFloat>::digits10
              << '\n';
    std::cout << "    cpp_bin_float_quad digits10 = "
              << std::numeric_limits<highvoronoi::MultiprecisionFloat>::digits10
              << '\n';

    std::cout << "    ExtendedFloat backend      = "
              << (std::is_same_v<
                      highvoronoi::ExtendedFloat,
                      highvoronoi::MultiprecisionFloat>
                      ? "cpp_bin_float_quad (Float128)"
                      : "long double")
              << '\n';

    std::cout << "\nNote: ExtendedQRNormalSolver computes QR internally in\n"
              << "ExtendedFloat, then converts the final normalized u back\n"
              << "to Scalar = double, exactly as the production solver does.\n";
}

} // namespace

int main() {
    std::cout << std::scientific << std::setprecision(17);

    print_precision_information();

    const Nodes nodes = make_random_nodes();
    const Sigma sigma = make_sigma();

    print_nodes(nodes);

    test_normal_double_qr(nodes, sigma);
    test_float128_qr(nodes, sigma);

    std::cout << "\n============================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "    checks : " << performed_checks << '\n';
    std::cout << "    failed : " << failed_checks << '\n';

    if (failed_checks == 0) {
        std::cout << "\nALL QR NORMAL TESTS PASSED\n";
        return 0;
    }

    std::cout << "\nQR NORMAL TEST FAILED\n";
    return 1;
}
