#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

namespace highvoronoi::detail {

template<class Key>
class UInt64View {
public:
    explicit UInt64View(const Key& key) noexcept
        : key_(key) {}

    [[nodiscard]] std::size_t size() const {
        return std::size(key_);
    }

    [[nodiscard]] std::uint64_t operator[](std::size_t i) const {
        return key_[i];
    }

private:
    const Key& key_;
};


namespace hash_detail {

[[nodiscard]] constexpr std::uint64_t rotl64(
    std::uint64_t x,
    unsigned r
) noexcept
{
    return (x << r) | (x >> (64U - r));
}

[[nodiscard]] constexpr std::uint64_t fmix64(
    std::uint64_t x
) noexcept
{
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33U;
    return x;
}

} // namespace hash_detail


// ============================================================================
// FNV-1a 64
// ============================================================================

template<
    std::uint64_t Prime = 1099511628211ULL,
    std::uint64_t OffsetBasis = 14695981039346656037ULL
>
struct FNV1a64 {
    template<class UInt64Array>
    [[nodiscard]] std::uint64_t operator()(
        const UInt64Array& key
    ) const noexcept
    {
        std::uint64_t hash = OffsetBasis;

        for (std::size_t i = 0; i < key.size(); ++i) {
            std::uint64_t value = key[i];

            for (unsigned b = 0; b < 8U; ++b) {
                hash ^= value & 0xffULL;
                hash *= Prime;
                value >>= 8U;
            }
        }

        return hash;
    }
};


// ============================================================================
// XXHash64
// ============================================================================

template<std::uint64_t Seed = 0>
struct XXHash64 {
    template<class UInt64Array>
    [[nodiscard]] std::uint64_t operator()(
        const UInt64Array& key
    ) const noexcept
    {
        constexpr std::uint64_t P1 = 0x9E3779B185EBCA87ULL;
        constexpr std::uint64_t P2 = 0xC2B2AE3D27D4EB4FULL;
        constexpr std::uint64_t P3 = 0x165667B19E3779F9ULL;
        constexpr std::uint64_t P4 = 0x85EBCA77C2B2AE63ULL;
        constexpr std::uint64_t P5 = 0x27D4EB2F165667C5ULL;

        auto round = [](std::uint64_t acc, std::uint64_t input) noexcept {
            acc += input * P2;
            acc = hash_detail::rotl64(acc, 31U);
            acc *= P1;
            return acc;
        };

        auto merge_round = [&](std::uint64_t acc, std::uint64_t value) noexcept {
            acc ^= round(0, value);
            acc = acc * P1 + P4;
            return acc;
        };

        const std::size_t n = key.size();
        const std::uint64_t len = n * 8ULL;

        std::size_t i = 0;
        std::uint64_t hash;

        if (n >= 4) {
            std::uint64_t v1 = Seed + P1 + P2;
            std::uint64_t v2 = Seed + P2;
            std::uint64_t v3 = Seed;
            std::uint64_t v4 = Seed - P1;

            while (i + 4 <= n) {
                v1 = round(v1, key[i++]);
                v2 = round(v2, key[i++]);
                v3 = round(v3, key[i++]);
                v4 = round(v4, key[i++]);
            }

            hash =
                hash_detail::rotl64(v1, 1U) +
                hash_detail::rotl64(v2, 7U) +
                hash_detail::rotl64(v3, 12U) +
                hash_detail::rotl64(v4, 18U);

            hash = merge_round(hash, v1);
            hash = merge_round(hash, v2);
            hash = merge_round(hash, v3);
            hash = merge_round(hash, v4);
        }
        else {
            hash = Seed + P5;
        }

        hash += len;

        while (i < n) {
            const std::uint64_t k = round(0, key[i++]);

            hash ^= k;
            hash = hash_detail::rotl64(hash, 27U) * P1 + P4;
        }

        hash ^= hash >> 33U;
        hash *= P2;
        hash ^= hash >> 29U;
        hash *= P3;
        hash ^= hash >> 32U;

        return hash;
    }
};


// ============================================================================
// MurmurHash64A
// ============================================================================

template<std::uint64_t Seed = 0>
struct MurmurHash64 {
    template<class UInt64Array>
    [[nodiscard]] std::uint64_t operator()(
        const UInt64Array& key
    ) const noexcept
    {
        constexpr std::uint64_t M = 0xc6a4a7935bd1e995ULL;
        constexpr unsigned R = 47U;

        const std::uint64_t len = key.size() * 8ULL;

        std::uint64_t hash = Seed ^ (len * M);

        for (std::size_t i = 0; i < key.size(); ++i) {
            std::uint64_t k = key[i];

            k *= M;
            k ^= k >> R;
            k *= M;

            hash ^= k;
            hash *= M;
        }

        hash ^= hash >> R;
        hash *= M;
        hash ^= hash >> R;

        return hash;
    }
};


// ============================================================================
// MurmurHash3 x64 128
// ============================================================================

struct MurmurHash128Value {
    std::uint64_t first;
    std::uint64_t second;

    [[nodiscard]] constexpr bool operator==(
        const MurmurHash128Value& other
    ) const noexcept
    {
        return first == other.first &&
               second == other.second;
    }

    [[nodiscard]] constexpr bool operator!=(
        const MurmurHash128Value& other
    ) const noexcept
    {
        return !(*this == other);
    }
};


template<std::uint32_t Seed = 0>
struct MurmurHash_x64_128 {
    template<class UInt64Array>
    [[nodiscard]] MurmurHash128Value operator()(
        const UInt64Array& key
    ) const noexcept
    {
        constexpr std::uint64_t C1 =
            0x87c37b91114253d5ULL;

        constexpr std::uint64_t C2 =
            0x4cf5ad432745937fULL;

        std::uint64_t h1 = Seed;
        std::uint64_t h2 = Seed;

        const std::size_t n = key.size();

        std::size_t i = 0;

        while (i + 1 < n) {
            std::uint64_t k1 = key[i++];
            std::uint64_t k2 = key[i++];

            k1 *= C1;
            k1 = hash_detail::rotl64(k1, 31U);
            k1 *= C2;
            h1 ^= k1;

            h1 = hash_detail::rotl64(h1, 27U);
            h1 += h2;
            h1 = h1 * 5ULL + 0x52dce729ULL;

            k2 *= C2;
            k2 = hash_detail::rotl64(k2, 33U);
            k2 *= C1;
            h2 ^= k2;

            h2 = hash_detail::rotl64(h2, 31U);
            h2 += h1;
            h2 = h2 * 5ULL + 0x38495ab5ULL;
        }

        if (i < n) {
            std::uint64_t k1 = key[i];

            k1 *= C1;
            k1 = hash_detail::rotl64(k1, 31U);
            k1 *= C2;

            h1 ^= k1;
        }

        const std::uint64_t len = n * 8ULL;

        h1 ^= len;
        h2 ^= len;

        h1 += h2;
        h2 += h1;

        h1 = hash_detail::fmix64(h1);
        h2 = hash_detail::fmix64(h2);

        h1 += h2;
        h2 += h1;

        return {h1, h2};
    }
};


// ============================================================================
// SipHash-2-4
// ============================================================================

template<
    std::uint64_t K0 = 0,
    std::uint64_t K1 = 0
>
struct SipHash64 {
    template<class UInt64Array>
    [[nodiscard]] std::uint64_t operator()(
        const UInt64Array& key
    ) const noexcept
    {
        std::uint64_t v0 =
            0x736f6d6570736575ULL ^ K0;

        std::uint64_t v1 =
            0x646f72616e646f6dULL ^ K1;

        std::uint64_t v2 =
            0x6c7967656e657261ULL ^ K0;

        std::uint64_t v3 =
            0x7465646279746573ULL ^ K1;

        auto sip_round = [&]() noexcept {
            v0 += v1;
            v1 = hash_detail::rotl64(v1, 13U);
            v1 ^= v0;
            v0 = hash_detail::rotl64(v0, 32U);

            v2 += v3;
            v3 = hash_detail::rotl64(v3, 16U);
            v3 ^= v2;

            v0 += v3;
            v3 = hash_detail::rotl64(v3, 21U);
            v3 ^= v0;

            v2 += v1;
            v1 = hash_detail::rotl64(v1, 17U);
            v1 ^= v2;
            v2 = hash_detail::rotl64(v2, 32U);
        };

        for (std::size_t i = 0; i < key.size(); ++i) {
            const std::uint64_t value = key[i];

            v3 ^= value;

            sip_round();
            sip_round();

            v0 ^= value;
        }

        const std::uint64_t len =
            key.size() * 8ULL;

        const std::uint64_t end =
            (len & 0xffULL) << 56U;

        v3 ^= end;

        sip_round();
        sip_round();

        v0 ^= end;

        v2 ^= 0xffULL;

        sip_round();
        sip_round();
        sip_round();
        sip_round();

        return v0 ^ v1 ^ v2 ^ v3;
    }
};

} // namespace highvoronoi::detail
