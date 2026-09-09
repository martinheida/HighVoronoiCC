#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @file hvview.hpp
 * @brief Zero-overhead index views for translating HighVoronoi indices.
 *
 * An index view represents a reversible mapping from an external index to an
 * internal index. All indices are zero-based, as usual in C++. The index type
 * is supplied as an unsigned integer template parameter and is available in
 * every view as the nested type `Index`.
 *
 * A single index is translated with `operator*` and translated back with
 * `operator/`:
 *
 * @code{.cpp}
 * using Index = std::uint32_t;
 * highvoronoi::SwitchView<Index> view(2, 4);
 *
 * Index internal = view * external;
 * Index external_again = view / internal;
 * @endcode
 *
 * Entire index containers can be translated with `forward` and `backward`.
 * The input and output types only need to provide `size()` and `operator[]`.
 * The overload returning a value creates a `std::vector<Index>`. The overload
 * receiving an output container writes into existing storage and assumes that
 * the output container is large enough.
 *
 * @code{.cpp}
 * std::vector<Index> indices{0, 1, 2, 3};
 *
 * auto translated = view.forward(indices);
 *
 * std::vector<Index> output(indices.size());
 * view.backward(translated, output);
 * @endcode
 *
 * `CombinedView` composes two mappings. `ShuffleView` uses two externally
 * owned lookup containers, one for each direction. Those containers must
 * remain valid for the complete lifetime of the `ShuffleView`.
 *
 * The implementation uses the Curiously Recurring Template Pattern (CRTP).
 * It therefore provides shared functionality without virtual functions or
 * runtime dispatch. Concrete views only implement `map_index(Index)` and
 * `unmap_index(Index)`.
 *
 * @note Julia's one-based index ranges must be shifted when porting code.
 *       For example, Julia's `SwitchView(3, 5)` is `SwitchView(2, 4)` here.
 */

/**
 * @brief CRTP base class for reversible index views.
 *
 * @tparam Derived Concrete view type.
 * @tparam Index_ Unsigned integer type used for all indices.
 */
template<class Derived, typename Index_>
class HVView
{
    static_assert(std::is_integral_v<Index_>,
                  "HVView Index must be an integer type");
    static_assert(std::is_unsigned_v<Index_>,
                  "HVView Index must be unsigned");

public:
    using Index = Index_;

    /** @brief Translate one external index to its internal index. */
    [[nodiscard]] inline Index operator*(Index index) const
        noexcept(noexcept(derived().map_index(index)))
    {
        return derived().map_index(index);
    }

    /** @brief Translate one internal index back to its external index. */
    [[nodiscard]] inline Index operator/(Index index) const
        noexcept(noexcept(derived().unmap_index(index)))
    {
        return derived().unmap_index(index);
    }

    /** @brief Translate all indices and return newly allocated storage. */
    template<class InputVector>
    [[nodiscard]] std::vector<Index> forward(const InputVector& indices) const
    {
        std::vector<Index> output(indices.size());
        forward(indices, output);
        return output;
    }

    /**
     * @brief Translate all indices into caller-provided storage.
     *
     * The output container must contain at least `indices.size()` elements.
     */
    template<class InputVector, class OutputVector>
    void forward(const InputVector& indices, OutputVector& output) const
    {
        for(Index i = 0; i < indices.size(); ++i)
            output[i] = (*this) * indices[i];
    }

    /** @brief Translate all indices back and return newly allocated storage. */
    template<class InputVector>
    [[nodiscard]] std::vector<Index> backward(const InputVector& indices) const
    {
        std::vector<Index> output(indices.size());
        backward(indices, output);
        return output;
    }

    /**
     * @brief Translate all indices back into caller-provided storage.
     *
     * The output container must contain at least `indices.size()` elements.
     */
    template<class InputVector, class OutputVector>
    void backward(const InputVector& indices, OutputVector& output) const
    {
        for(Index i = 0; i < indices.size(); ++i)
            output[i] = (*this) / indices[i];
    }

private:
    [[nodiscard]] inline const Derived& derived() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }
};


/**
 * @brief Move an inclusive index interval to the beginning of its prefix.
 *
 * For `SwitchView(2, 4)`, the prefix `[0, 4]` is mapped as follows:
 *
 * @code{.txt}
 * external: 0 1 2 3 4
 * internal: 3 4 0 1 2
 * @endcode
 *
 * Indices greater than `last` remain unchanged.
 */
template<typename Index_>
class SwitchView : public HVView<SwitchView<Index_>, Index_>
{
public:
    using Index = Index_;

    SwitchView(Index first, Index last)
        : first_(first),
          last_(last),
          length_(0)
    {
        if(last < first)
            throw std::invalid_argument(
                "SwitchView: last must not be smaller than first");

        length_ = last - first + 1;
    }

    [[nodiscard]] inline Index map_index(Index index) const noexcept
    {
        if(index > last_)
            return index;

        return index >= first_
             ? index - first_
             : index + length_;
    }

    [[nodiscard]] inline Index unmap_index(Index index) const noexcept
    {
        if(index > last_)
            return index;

        return index < length_
             ? index + first_
             : index - length_;
    }

    [[nodiscard]] inline Index first() const noexcept { return first_; }
    [[nodiscard]] inline Index last() const noexcept { return last_; }

private:
    Index first_;
    Index last_;
    Index length_;
};


template<typename Index>
SwitchView(Index, Index) -> SwitchView<Index>;


/**
 * @brief Composition of two reversible index views.
 *
 * Forward translation applies the inner view first and the outer view second.
 * Backward translation applies the corresponding inverse operations in the
 * opposite order.
 */
template<class Outer, class Inner>
class CombinedView
    : public HVView<CombinedView<Outer, Inner>, typename Outer::Index>
{
public:
    using Index = typename Outer::Index;

    static_assert(std::is_same_v<Index, typename Inner::Index>,
                  "CombinedView requires equal Index types");

    CombinedView(Outer outer, Inner inner)
        : outer_(std::move(outer)),
          inner_(std::move(inner))
    {}

    [[nodiscard]] inline Index map_index(Index index) const
        noexcept(noexcept(outer_ * (inner_ * index)))
    {
        return outer_ * (inner_ * index);
    }

    [[nodiscard]] inline Index unmap_index(Index index) const
        noexcept(noexcept(inner_ / (outer_ / index)))
    {
        return inner_ / (outer_ / index);
    }

    [[nodiscard]] inline const Outer& outer() const noexcept { return outer_; }
    [[nodiscard]] inline const Inner& inner() const noexcept { return inner_; }

private:
    Outer outer_;
    Inner inner_;
};


template<class Outer, class Inner>
CombinedView(Outer, Inner) -> CombinedView<Outer, Inner>;


/**
 * @brief Reversible lookup view backed by two external index containers.
 *
 * `external_indices[index]` performs the forward translation and
 * `internal_indices[index]` performs the backward translation. Indices at or
 * above `length` remain unchanged.
 *
 * The lookup containers are referenced, not copied. They must outlive this
 * view and must provide `operator[]` and `size()`.
 */
template<typename Index_, class ExternalIndices, class InternalIndices>
class ShuffleView
    : public HVView<ShuffleView<Index_, ExternalIndices, InternalIndices>,
                    Index_>
{
public:
    using Index = Index_;

    ShuffleView(const ExternalIndices& external_indices,
                const InternalIndices& internal_indices)
        : external_indices_(external_indices),
          internal_indices_(internal_indices),
          length_(external_indices.size())
    {}

    ShuffleView(const ExternalIndices& external_indices,
                const InternalIndices& internal_indices,
                Index length)
        : external_indices_(external_indices),
          internal_indices_(internal_indices),
          length_(length)
    {}

    [[nodiscard]] inline Index map_index(Index index) const
    {
        return index < length_ ? external_indices_[index] : index;
    }

    [[nodiscard]] inline Index unmap_index(Index index) const
    {
        return index < length_ ? internal_indices_[index] : index;
    }

    [[nodiscard]] inline Index length() const noexcept { return length_; }

private:
    const ExternalIndices& external_indices_;
    const InternalIndices& internal_indices_;
    Index length_;
};


template<class ExternalIndices, class InternalIndices>
ShuffleView(const ExternalIndices&, const InternalIndices&)
    -> ShuffleView<typename ExternalIndices::value_type,
                   ExternalIndices,
                   InternalIndices>;

template<class ExternalIndices, class InternalIndices, typename Index>
ShuffleView(const ExternalIndices&, const InternalIndices&, Index)
    -> ShuffleView<Index, ExternalIndices, InternalIndices>;

} // namespace highvoronoi
