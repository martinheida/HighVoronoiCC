#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace highvoronoi::detail {

/**
 * @brief Minimal portable unsigned 128-bit value used by the hash tables.
 *
 * Only the operations required by the HighVoronoi FNV-style hash are exposed.
 * This deliberately avoids compiler-specific __int128 so the header remains
 * valid C++17 on GCC, Clang, and MSVC.
 */
struct UInt128 final {
    std::uint64_t high{0};
    std::uint64_t low{0};

    constexpr UInt128() noexcept = default;
    constexpr UInt128(std::uint64_t high_value, std::uint64_t low_value) noexcept
        : high(high_value), low(low_value)
    {}

    [[nodiscard]] constexpr bool is_zero() const noexcept
    {
        return high == 0 && low == 0;
    }

    [[nodiscard]] constexpr bool is_odd() const noexcept
    {
        return (low & std::uint64_t{1}) != 0;
    }

    constexpr void shift_right_one() noexcept
    {
        low = (low >> 1U) | (high << 63U);
        high >>= 1U;
    }
};

[[nodiscard]] constexpr bool operator==(UInt128 lhs, UInt128 rhs) noexcept
{
    return lhs.high == rhs.high && lhs.low == rhs.low;
}

[[nodiscard]] constexpr bool operator!=(UInt128 lhs, UInt128 rhs) noexcept
{
    return !(lhs == rhs);
}

namespace fnv_detail {

inline constexpr std::uint64_t offset_basis_64 = 14695981039346656037ULL;
inline constexpr std::uint64_t prime_64 = 1099511628211ULL;

// 0x6c62272e07bb014262b821756295c58d
inline constexpr UInt128 offset_basis_128{
    0x6c62272e07bb0142ULL,
    0x62b821756295c58dULL
};

// FNV-128 prime = 2^88 + 0x13b.
inline constexpr std::uint64_t prime_128_high = 0x0000000001000000ULL;
inline constexpr std::uint64_t prime_128_low = 0x13bULL;

/** Return the high 64 bits of value * 0x13b and write the low 64 bits. */
[[nodiscard]] inline std::uint64_t multiply_low_by_315(
    std::uint64_t value,
    std::uint64_t& low_result) noexcept
{
    constexpr std::uint64_t mask32 = 0xffffffffULL;
    constexpr std::uint64_t multiplier = prime_128_low; // 315

    const std::uint64_t p0 = (value & mask32) * multiplier;
    const std::uint64_t p1 = (value >> 32U) * multiplier;

    const std::uint64_t shifted = p1 << 32U;
    low_result = p0 + shifted;

    const std::uint64_t carry_from_add = low_result < p0 ? 1ULL : 0ULL;
    return (p1 >> 32U) + carry_from_add;
}

/** Multiply by the FNV-128 prime modulo 2^128. */
[[nodiscard]] inline UInt128 multiply_prime_128(UInt128 value) noexcept
{
    std::uint64_t new_low = 0;
    const std::uint64_t carry = multiply_low_by_315(value.low, new_low);

    // (high:low) * (2^88 + 315) modulo 2^128:
    // high limb = high*315 + high64(low*315) + low*2^24.
    const std::uint64_t new_high =
        value.high * prime_128_low + carry + (value.low << 24U);

    return UInt128{new_high, new_low};
}

template <class>
struct always_false : std::false_type {};

template <class Value>
[[nodiscard]] inline std::uint64_t integer_word(const Value& value) noexcept
{
    using Raw = std::decay_t<Value>;
    static_assert(std::is_integral_v<Raw>,
                  "fnv1a_hash expects an integer-valued key");

    return value;
}


} // namespace fnv_detail

/**
 * @brief Return the next power of two, with next_power_of_two(0) == 1.
 */
[[nodiscard]] inline std::size_t next_power_of_two(std::size_t n)
{
    if (n <= 1) {
        return 1;
    }

    --n;
    for (std::size_t shift = 1;
         shift < std::numeric_limits<std::size_t>::digits;
         shift <<= 1U) {
        n |= n >> shift;
    }
    ++n;

    if (n == 0) {
        throw std::length_error("hash table size exceeds the largest power of two");
    }
    return n;
}

/**
 * @brief HighVoronoi's FNV-1a-style integer-vector hash.
 *
 * This intentionally matches the Julia implementation: every integer is XORed
 * as one 64-bit word and followed by one FNV multiplication. It is therefore
 * not the conventional byte-by-byte FNV-1a hash.
 *
 * The result is normalized to a non-zero odd value, exactly as in Julia.
 */
template <class UInt, class Key>
[[nodiscard]] inline UInt fnv1a_hash(const Key& key, std::int64_t bias = 0)
{
    if constexpr (std::is_same_v<UInt, std::uint64_t>) {
        std::uint64_t hash = fnv_detail::offset_basis_64;

        const auto count = static_cast<std::size_t>(key.size());
        for (std::size_t i = 0; i < count; ++i) {
            hash ^= fnv_detail::integer_word(key[i]);
            hash *= fnv_detail::prime_64; // unsigned overflow = modulo 2^64
        }

        if (bias != 0) {
            hash ^= static_cast<std::uint64_t>(bias);
            hash *= fnv_detail::prime_64;
        }

        if (hash == 0) {
            return std::uint64_t{1};
        }
        while ((hash & std::uint64_t{1}) == 0) {
            hash >>= 1U;
        }
        return hash;
    } else if constexpr (std::is_same_v<UInt, UInt128>) {
        UInt128 hash = fnv_detail::offset_basis_128;

        const auto count = static_cast<std::size_t>(key.size());
        for (std::size_t i = 0; i < count; ++i) {
            hash.low ^= fnv_detail::integer_word(key[i]);
            hash = fnv_detail::multiply_prime_128(hash);
        }

        if (bias != 0) {
            hash.low ^= static_cast<std::uint64_t>(bias);
            hash = fnv_detail::multiply_prime_128(hash);
        }

        if (hash.is_zero()) {
            return UInt128{0, 1};
        }
        while (!hash.is_odd()) {
            hash.shift_right_one();
        }
        return hash;
    } else {
        static_assert(fnv_detail::always_false<UInt>::value,
                      "fnv1a_hash supports only uint64_t and highvoronoi::detail::UInt128");
    }
}

} // namespace highvoronoi::detail
