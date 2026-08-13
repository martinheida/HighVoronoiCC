#pragma once

#include <limits>
#include <type_traits>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/eigen.hpp>

#include <Eigen/Dense>

namespace highvoronoi {

// -----------------------------------------------------------------------------
// Normale, layout-kompatible Maschinentypen
// -----------------------------------------------------------------------------

using Float32 = float;
using Float64 = double;

static_assert(
    std::numeric_limits<Float32>::is_iec559 &&
    std::numeric_limits<Float32>::radix == 2 &&
    std::numeric_limits<Float32>::digits == 24 &&
    sizeof(Float32) == 4,
    "HighVoronoi requires IEEE-754 binary32."
);

static_assert(
    std::numeric_limits<Float64>::is_iec559 &&
    std::numeric_limits<Float64>::radix == 2 &&
    std::numeric_limits<Float64>::digits == 53 &&
    sizeof(Float64) == 8,
    "HighVoronoi requires IEEE-754 binary64."
);

// -----------------------------------------------------------------------------
// Erweiterte Präzision
// -----------------------------------------------------------------------------

inline constexpr int required_extra_decimal_digits = 5;

using MultiprecisionFloat =
    boost::multiprecision::cpp_bin_float_quad;

inline constexpr bool long_double_is_sufficient =
    std::numeric_limits<long double>::is_specialized &&
    std::numeric_limits<long double>::digits10 >=
        std::numeric_limits<Float64>::digits10
        + required_extra_decimal_digits;

using ExtendedFloat = std::conditional_t<
    long_double_is_sufficient,
    long double,
    MultiprecisionFloat
>;

static_assert(
    std::numeric_limits<ExtendedFloat>::digits10 >=
        std::numeric_limits<Float64>::digits10
        + required_extra_decimal_digits,
    "ExtendedFloat does not provide enough additional precision."
);

// -----------------------------------------------------------------------------
// Eigen-Typen
// -----------------------------------------------------------------------------

template<int Dim = Eigen::Dynamic>
using ExtendedVector =
    Eigen::Matrix<ExtendedFloat, Dim, 1>;

template<
    int Rows = Eigen::Dynamic,
    int Cols = Eigen::Dynamic
>
using ExtendedMatrix =
    Eigen::Matrix<ExtendedFloat, Rows, Cols>;

} // namespace highvoronoi