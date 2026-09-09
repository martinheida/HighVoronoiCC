#include <highvoronoi/integration/detail/incremental_minors.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool ok, std::string_view text) {
    ++checks;
    if (ok) {
        std::cout << "[OK]   " << text << '\n';
    } else {
        ++failures;
        std::cerr << "[FAIL] " << text << '\n';
    }
}

double determinant(std::vector<std::vector<double>> matrix) {
    const std::size_t n = matrix.size();
    double sign = 1.0;
    double result = 1.0;

    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row) {
            if (std::abs(matrix[row][column]) >
                std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) < 1e-14) {
            return 0.0;
        }
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
            sign = -sign;
        }
        const double p = matrix[column][column];
        result *= p;
        for (std::size_t row = column + 1; row < n; ++row) {
            const double factor = matrix[row][column] / p;
            for (std::size_t c = column + 1; c < n; ++c) {
                matrix[row][c] -= factor * matrix[column][c];
            }
        }
    }
    return sign * result;
}

void test_dimensions() {
    std::mt19937_64 rng(0x4d494e4f5253ULL);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    for (std::size_t dimension = 1; dimension <= 7; ++dimension) {
        for (std::size_t sample = 0; sample < 64; ++sample) {
            std::vector<std::vector<double>> rows(
                dimension,
                std::vector<double>(dimension));
            for (auto& row : rows) {
                for (double& value : row) {
                    value = dist(rng);
                }
            }

            highvoronoi::detail::IncrementalMinors<double> minors(dimension);
            for (std::size_t k = 1; k <= dimension; ++k) {
                minors.update(k, rows[k - 1]);
            }

            const double reference = determinant(rows);
            const double got = minors.determinant();
            const double tolerance = 2e-10 * (1.0 + std::abs(reference));
            if (std::abs(std::abs(got) - std::abs(reference)) > tolerance) {
                std::cerr << "dimension=" << dimension
                          << " sample=" << sample
                          << " got=" << got
                          << " ref=" << reference << '\n';
                check(false, "incremental top minor matches determinant magnitude");
                return;
            }
        }
        check(true, "incremental top minor matches determinant magnitude");
    }
}

void test_layer_reuse() {
    highvoronoi::detail::IncrementalMinors<double> minors(3);
    const std::vector<double> a{1.0, 2.0, 3.0};
    const std::vector<double> b{4.0, 5.0, 6.0};
    const std::vector<double> c{7.0, 8.0, 10.0};

    minors.update(1, a);
    const auto first_before = minors.data(1);
    minors.update(2, b);
    const auto second_before = minors.data(2);
    minors.update(3, c);

    check(minors.data(1) == first_before,
          "updating higher orders leaves first-order recycled layer unchanged");
    check(minors.data(2) == second_before,
          "updating top order leaves second-order recycled layer unchanged");
    check(std::abs(std::abs(minors.determinant()) - 3.0) < 1e-12,
          "known 3x3 determinant magnitude is correct");
}

} // namespace

int main() {
    test_dimensions();
    test_layer_reuse();
    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
