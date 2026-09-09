
#pragma once

/**
 * @file mesh_index_mapping.hpp
 * @brief Compile-time public/internal index mapping policies.
 *
 * `DenseIndexMapping` is the ordinary bijective policy used by VoronoiMesh and
 * compute views. `ProjectedIndexMapping` is the deliberately non-injective
 * internal-to-public policy used by HighVoronoiMesh: one public visible node has
 * one canonical visible internal representative, while active periodic copies
 * may project onto the same public index.
 */

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Dense bijective public/internal mapping with stable internal slots.
 */
template <class IndexT>
class DenseIndexMapping {
    static_assert(std::is_integral_v<IndexT> && std::is_unsigned_v<IndexT>);

public:
    using Index = IndexT;
    DenseIndexMapping() = default;

    explicit DenseIndexMapping(Index count) {
        reset_identity(count);
    }

    [[nodiscard]] static DenseIndexMapping identity(Index count) {
        return DenseIndexMapping(count);
    }

    [[nodiscard]] Index size() const noexcept {
        return static_cast<Index>(public_to_internal_.size());
    }

    [[nodiscard]] Index internal_size() const noexcept {
        return static_cast<Index>(internal_to_public_.size());
    }

    [[nodiscard]] Index public_to_internal(Index public_index) const {
        return public_to_internal_.at(static_cast<std::size_t>(public_index));
    }

    [[nodiscard]] std::optional<Index>
    internal_to_public(Index internal_index) const {
        const Index value = internal_to_public_.at(
            static_cast<std::size_t>(internal_index));
        if (value == invalid_index()) {
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] const std::vector<Index>&
    public_to_internal_data() const noexcept {
        return public_to_internal_;
    }

    [[nodiscard]] const std::vector<Index>&
    internal_to_public_data() const noexcept {
        return internal_to_public_;
    }

    void reset_identity(Index count) {
        public_to_internal_.resize(static_cast<std::size_t>(count));
        internal_to_public_.resize(static_cast<std::size_t>(count));
        for (Index i = Index{0}; i < count; ++i) {
            public_to_internal_[static_cast<std::size_t>(i)] = i;
            internal_to_public_[static_cast<std::size_t>(i)] = i;
        }
    }

    void assign(
        std::vector<Index> public_to_internal,
        std::vector<Index> internal_to_public) {
        public_to_internal_ = std::move(public_to_internal);
        internal_to_public_ = std::move(internal_to_public);
    }

    void mark_deleted(Index internal_index) {
        Index& value = internal_to_public_.at(
            static_cast<std::size_t>(internal_index));
        value = invalid_index();
    }

    [[nodiscard]] bool is_deleted(Index internal_index) const {
        return internal_to_public_.at(static_cast<std::size_t>(internal_index)) ==
               invalid_index();
    }

    /** Compact public numbering while preserving stable internal indices. */
    void rebuild() {
        public_to_internal_.clear();
        public_to_internal_.reserve(internal_to_public_.size());

        for (std::size_t internal = 0;
             internal < internal_to_public_.size();
             ++internal) {
            if (internal_to_public_[internal] == invalid_index()) {
                continue;
            }
            public_to_internal_.push_back(static_cast<Index>(internal));
        }

        std::fill(
            internal_to_public_.begin(),
            internal_to_public_.end(),
            invalid_index());

        for (std::size_t public_index = 0;
             public_index < public_to_internal_.size();
             ++public_index) {
            const Index internal = public_to_internal_[public_index];
            internal_to_public_[static_cast<std::size_t>(internal)] =
                static_cast<Index>(public_index);
        }
    }

    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return std::numeric_limits<Index>::max();
    }

private:
    std::vector<Index> public_to_internal_;
    std::vector<Index> internal_to_public_;
};

/**
 * @brief HighVoronoi's projected public mapping.
 *
 * `public_to_internal` remains injective and selects the canonical active
 * visible internal node for a public cell. `internal_to_public` is intentionally
 * many-to-one: an active invisible periodic copy may map to the same public
 * index as its visible reference. Hidden/deleted internal slots use the invalid
 * marker.
 *
 * This is not a reversible mapping and therefore deliberately has no generic
 * `rebuild()` operation. HighVoronoiMesh owns the visibility/reference state
 * needed to rebuild it correctly and supplies both arrays through `assign()`.
 */
template <class IndexT>
class ProjectedIndexMapping {
    static_assert(std::is_integral_v<IndexT> && std::is_unsigned_v<IndexT>);

public:
    using Index = IndexT;
    ProjectedIndexMapping() = default;

    [[nodiscard]] Index size() const noexcept {
        return static_cast<Index>(public_to_internal_.size());
    }

    [[nodiscard]] Index internal_size() const noexcept {
        return static_cast<Index>(internal_to_public_.size());
    }

    /** Return the canonical visible internal representative of one public cell. */
    [[nodiscard]] Index public_to_internal(Index public_index) const {
        return public_to_internal_.at(static_cast<std::size_t>(public_index));
    }

    /** Project one active internal node onto visible public numbering. */
    [[nodiscard]] std::optional<Index>
    internal_to_public(Index internal_index) const {
        const Index value = internal_to_public_.at(
            static_cast<std::size_t>(internal_index));
        if (value == invalid_index()) {
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] const std::vector<Index>&
    public_to_internal_data() const noexcept {
        return public_to_internal_;
    }

    [[nodiscard]] const std::vector<Index>&
    internal_to_public_data() const noexcept {
        return internal_to_public_;
    }

    /** Replace the complete projected mapping after structural HighVoronoi changes. */
    void assign(
        std::vector<Index> public_to_internal,
        std::vector<Index> internal_to_public) {
        public_to_internal_ = std::move(public_to_internal);
        internal_to_public_ = std::move(internal_to_public);
    }

    void clear() noexcept {
        public_to_internal_.clear();
        internal_to_public_.clear();
    }

    /**
     * Append one new canonical visible stable slot.
     *
     * HighVoronoi appends stable internal slots monotonically; enforcing that
     * invariant here catches accidental remapping early.
     */
    [[nodiscard]] Index append_visible(Index internal_index) {
        require_next_internal(internal_index);
        const Index public_index = static_cast<Index>(public_to_internal_.size());
        public_to_internal_.push_back(internal_index);
        internal_to_public_.push_back(public_index);
        return public_index;
    }

    /** Append one invisible stable slot projected onto an existing public node. */
    void append_alias(Index internal_index, Index public_index) {
        require_next_internal(internal_index);
        if (public_index >= static_cast<Index>(public_to_internal_.size())) {
            throw std::out_of_range(
                "ProjectedIndexMapping alias refers to an unknown public node.");
        }
        internal_to_public_.push_back(public_index);
    }

    /** Append one stable slot that currently has no public representation. */
    void append_hidden(Index internal_index) {
        require_next_internal(internal_index);
        internal_to_public_.push_back(invalid_index());
    }

    /** Hide one stable slot. Canonical public compaction is rebuilt by the owner. */
    void mark_deleted(Index internal_index) {
        internal_to_public_.at(static_cast<std::size_t>(internal_index)) =
            invalid_index();
    }

    [[nodiscard]] bool is_hidden(Index internal_index) const {
        return internal_to_public_.at(static_cast<std::size_t>(internal_index)) ==
               invalid_index();
    }

    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return std::numeric_limits<Index>::max();
    }

private:
    void require_next_internal(Index internal_index) const {
        if (static_cast<std::size_t>(internal_index) !=
            internal_to_public_.size()) {
            throw std::logic_error(
                "ProjectedIndexMapping stable slots must be appended in internal order.");
        }
    }

    std::vector<Index> public_to_internal_;
    std::vector<Index> internal_to_public_;
};

} // namespace highvoronoi


