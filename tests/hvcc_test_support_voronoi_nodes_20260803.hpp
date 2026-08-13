#pragma once

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace hvtest {

using Scalar = double;
using Index = std::size_t;
using IndexVector = std::vector<Index>;

inline constexpr Scalar tolerance = 1e-12;

inline std::size_t performed_checks = 0;
inline std::size_t failed_checks = 0;

/**
 * Record one runtime condition and print an immediately readable result.
 *
 * This is intentionally simpler than a unit-test framework: every important
 * property appears as one named line in the shell output.
 */
inline void check(bool condition, std::string_view description) {
    ++performed_checks;

    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

/** Compare two scalar values with the common test tolerance. */
inline bool close(Scalar left, Scalar right) {
    return std::abs(left - right) <= tolerance;
}

/**
 * Check both coordinates of a two-dimensional point-like object.
 *
 * The helper works for owning Eigen points, Eigen views and NodeHandle.
 */
template <typename PointLike>
void check_point_2d(
    const PointLike& point,
    Scalar expected_x,
    Scalar expected_y,
    std::string_view description) {

    check(
        close(point[Index{0}], expected_x) &&
            close(point[Index{1}], expected_y),
        description);
}

/** Check that a routine throws exactly the requested exception type. */
template <typename Exception, typename Function>
void check_throws(Function&& function, std::string_view description) {
    bool correct_exception_was_thrown = false;

    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        correct_exception_was_thrown = true;
    } catch (...) {
        // A different exception is a failed check.
    }

    check(correct_exception_was_thrown, description);
}

/**
 * Run one named test routine and isolate unexpected exceptions.
 *
 * The routine prints a concise test header and an explicit PASS/FAIL footer,
 * so failures can be located without reading the source first.
 */
template <typename Function>
void run_test(std::string_view name, Function&& function) {
    std::cout << "\n[TEST] " << name << '\n';
    const std::size_t failures_before = failed_checks;

    try {
        std::forward<Function>(function)();
    } catch (const std::exception& exception) {
        ++failed_checks;
        std::cerr << "    [FAIL] unexpected exception: "
                  << exception.what() << '\n';
    } catch (...) {
        ++failed_checks;
        std::cerr << "    [FAIL] unexpected non-standard exception\n";
    }

    if (failed_checks == failures_before) {
        std::cout << "[PASS] " << name << '\n';
    } else {
        std::cout << "[FAIL] " << name << '\n';
    }
}

} // namespace hvtest
