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
//
// Numerical fallback code deliberately uses exactly two precision levels:
// Float64 for the normal path and software binary128 for the rare robust path.
// In particular, `long double` is not used as an intermediate level because
// its precision is platform dependent (and may equal double).

using Float128 =
    boost::multiprecision::cpp_bin_float_quad;

using MultiprecisionFloat = Float128;
using ExtendedFloat = Float128;

static_assert(
    std::numeric_limits<Float128>::digits >= 113,
    "HighVoronoi Float128 must provide at least IEEE-754 binary128 precision."
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
