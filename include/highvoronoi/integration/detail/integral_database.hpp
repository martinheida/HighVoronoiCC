#pragma once

#include <highvoronoi/integration/detail/numeric_database.hpp>

namespace highvoronoi::detail {

/** Semantic database for bulk and interface integrals.
 * Scalar integrals are records of length 1; vector integrals use length > 1. */
template <class Lock, class FloatT>
class IntegralDatabase final : public NumericDatabase<Lock, FloatT> {
public:
    using Base = NumericDatabase<Lock, FloatT>;
    using Base::Base;
};

} // namespace highvoronoi::detail
