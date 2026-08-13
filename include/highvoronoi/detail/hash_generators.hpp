#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "hash_functions.hpp"
#include "fnv_hash.hpp"

namespace highvoronoi::detail {

/*
Required interface of every hash generator:

    Generator(key)
    generator.index(mask, i)
    generator == other_generator

mask is table_size - 1 and table_size must be a power of two.
*/
struct HashGeneratorBase {
protected:
    [[nodiscard]] static std::size_t double_hash_index(
        std::uint64_t first,
        std::uint64_t second,
        std::size_t mask,
        std::size_t i) noexcept
    {
        return static_cast<std::size_t>(
            (first + static_cast<std::uint64_t>(i) * (second | 1ULL))
            & static_cast<std::uint64_t>(mask));
    }
};


// ============================================================================
// 1) Two or more ordinary uint64_t-valued hash functions.
//
// The first hash gives the start position.
// The second hash gives the probe step.
// All hashes are stored and compared.
// ============================================================================

template<class FirstHash, class SecondHash, class... MoreHashes>
class UInt64HashGenerator : public HashGeneratorBase {
public:
    static constexpr std::size_t hash_count = 2 + sizeof...(MoreHashes);

    UInt64HashGenerator() = default;
    
    template<class Key>
    explicit UInt64HashGenerator(const Key& key)
    {
        UInt64View<Key> view(key);
        hashes_ = {
            FirstHash{}(view),
            SecondHash{}(view),
            MoreHashes{}(view)...
        };
    }

    [[nodiscard]] std::size_t index(
        std::size_t mask,
        std::size_t i) const noexcept
    {
        return double_hash_index(hashes_[0], hashes_[1], mask, i);
    }

    [[nodiscard]] bool operator==(
        const UInt64HashGenerator& other) const noexcept
    {
        return hashes_ == other.hashes_;
    }

    [[nodiscard]] bool operator!=(
        const UInt64HashGenerator& other) const noexcept
    {
        return !(*this == other);
    }

private:
    std::array<std::uint64_t, hash_count> hashes_{};
};


// ============================================================================
// 2) Extend any existing hash generator with additional uint64_t hash functions.
//
// index() is delegated completely to the hosted generator.
// The additional hashes are used only for equality comparison.
// ============================================================================

template<class Generator, class... AppendixHashes>
class ExtendedHashGenerator : public HashGeneratorBase {
public:
    ExtendedHashGenerator() = default;

    template<class Key>
    explicit ExtendedHashGenerator(const Key& key)
        : generator_(key)
    {
        UInt64View<Key> view(key);
        appendix_ = {AppendixHashes{}(view)...};
    }

    [[nodiscard]] std::size_t index(
        std::size_t mask,
        std::size_t i) const noexcept
    {
        return generator_.index(mask, i);
    }

    [[nodiscard]] bool operator==(
        const ExtendedHashGenerator& other) const noexcept
    {
        return generator_ == other.generator_
            && appendix_ == other.appendix_;
    }

    [[nodiscard]] bool operator!=(
        const ExtendedHashGenerator& other) const noexcept
    {
        return !(*this == other);
    }

private:
    Generator generator_;
    std::array<std::uint64_t, sizeof...(AppendixHashes)> appendix_{};
};


// ============================================================================
// 3) MurmurHash3 x64 128-bit generator.
//
// MurmurHash_x64_128 already returns two uint64_t values.
// first  -> start position
// second -> probe step
// ============================================================================

template<std::uint32_t Seed = 0>
class Murmur128HashGenerator : public HashGeneratorBase {
public:
    Murmur128HashGenerator() = default;

    template<class Key>
    explicit Murmur128HashGenerator(const Key& key)
    {
        UInt64View<Key> view(key);
        const auto hash = MurmurHash_x64_128<Seed>{}(view);
        first_ = hash.first;
        second_ = hash.second;
    }

    [[nodiscard]] std::size_t index(
        std::size_t mask,
        std::size_t i) const noexcept
    {
        return double_hash_index(first_, second_, mask, i);
    }

    [[nodiscard]] bool operator==(
        const Murmur128HashGenerator& other) const noexcept
    {
        return first_ == other.first_
            && second_ == other.second_;
    }

    [[nodiscard]] bool operator!=(
        const Murmur128HashGenerator& other) const noexcept
    {
        return !(*this == other);
    }

private:
    std::uint64_t first_{0};
    std::uint64_t second_{0};
};


// ============================================================================
// 4) Original HighVoronoi FNV generator.
//
// Stores:
//   - original 128-bit FNV-style hash
//   - original  64-bit FNV-style hash
//
// This reproduces the old Julia probing:
//   start = low 63 bits of the 128-bit hash
//   step  = 64-bit hash
// ============================================================================

class FNV64_128HashGenerator : public HashGeneratorBase {
public:
    FNV64_128HashGenerator() = default;

    template<class Key>
    explicit FNV64_128HashGenerator(const Key& key)
    {
        UInt64View<Key> view(key);
        hash128_ = fnv1a_hash<UInt128>(view);
        hash64_ = fnv1a_hash<std::uint64_t>(view);
    }

    [[nodiscard]] std::size_t index(
        std::size_t mask,
        std::size_t i) const noexcept
    {
        const std::uint64_t first =
            hash128_.low & 0x7fffffffffffffffULL;

        return static_cast<std::size_t>(
            (first + static_cast<std::uint64_t>(i) * hash64_)
            & static_cast<std::uint64_t>(mask));
    }

    [[nodiscard]] bool operator==(
        const FNV64_128HashGenerator& other) const noexcept
    {
        return hash128_ == other.hash128_
            && hash64_ == other.hash64_;
    }

    [[nodiscard]] bool operator!=(
        const FNV64_128HashGenerator& other) const noexcept
    {
        return !(*this == other);
    }

private:
    UInt128 hash128_{};
    std::uint64_t hash64_{0};
};

} // namespace highvoronoi::detail

