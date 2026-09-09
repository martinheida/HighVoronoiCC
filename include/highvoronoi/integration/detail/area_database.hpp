#pragma once

#include <highvoronoi/integration/detail/numeric_database.hpp>

namespace highvoronoi::detail {

/** Semantic database for interface measures. One record is normally one cell's
 * area list, in exactly the same order and multiplicity as its neighbour list. */
template <class Lock, class FloatT>
class AreaDatabase final : public NumericDatabase<Lock, FloatT> {
public:
    using Base = NumericDatabase<Lock, FloatT>;
    using Base::Base;
};

} // namespace highvoronoi::detail
