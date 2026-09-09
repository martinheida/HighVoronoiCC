#pragma once

/**
 * @file incremental_minors.hpp
 * @brief Recycled determinant-minor hierarchy used by PolygonAlgorithm.
 *
 * Direct C++17 port of HighVoronoi.jl's Minors/k_minor implementation.
 * For every order k the object stores all k-row minors of the first k supplied
 * vectors. Updating order k reuses the complete order-(k-1) layer instead of
 * recomputing determinants independently.
 *
 * The internal row-combination buffers deliberately keep Julia's one-based
 * indices. That makes get_minor()/next_minor() mechanically comparable with
 * the reference implementation; only vector storage itself is zero-based.
 */

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace highvoronoi::detail {

template <class ScalarT = double>
class IncrementalMinors final {
public:
    using Scalar = ScalarT;

    explicit IncrementalMinors(std::size_t dimension)
        : dimension_(checked_dimension(dimension)),
          hash_(dimension_ * dimension_, std::size_t{0}),
          data_(dimension_),
          buffers_(dimension_) {
        static_assert(
            std::is_floating_point_v<Scalar>,
            "IncrementalMinors requires a floating-point scalar");

        build_hash_matrix();
        for (std::size_t column = 1; column <= dimension_; ++column) {
            data_[column - 1].assign(hash_column(column), Scalar{0});
            buffers_[column - 1].assign(column, std::size_t{0});
        }
    }

    [[nodiscard]] std::size_t dimension() const noexcept {
        return dimension_;
    }

    [[nodiscard]] const std::vector<Scalar>& data(std::size_t order) const {
        require_order(order);
        return data_[order - 1];
    }

    [[nodiscard]] Scalar determinant() const {
        return data_.back().front();
    }

    /** Equivalent of Julia k_minor(minors, k, V), with k one-based. */
    template <class VectorLike>
    void update(std::size_t k, const VectorLike& vector) {
        require_order(k);
        if (static_cast<std::size_t>(vector.size()) != dimension_) {
            throw std::invalid_argument(
                "IncrementalMinors vector dimension mismatch");
        }

        if (k == 1) {
            auto& target = data_.front();
            for (std::size_t i = 0; i < dimension_; ++i) {
                target[i] = static_cast<Scalar>(vector[i]);
            }
            return;
        }

        auto& minor = buffers_[k - 1];
        for (std::size_t j = 0; j < k; ++j) {
            minor[j] = j + 1; // Julia's 1,2,...,k row indices.
        }

        auto& target = data_[k - 1];
        const auto& previous = data_[k - 2];
        std::size_t output = 0;

        for (;;) {
            Scalar result = Scalar{0};
            int sign = -1;
            for (std::size_t j = 1; j <= k; ++j) {
                sign *= -1; // Julia: (-1)^(j+1)
                const std::size_t row = minor[j - 1];
                const std::size_t previous_index =
                    get_minor_index(minor, j, k - 1);
                result +=
                    static_cast<Scalar>(vector[row - 1]) *
                    previous[previous_index] *
                    static_cast<Scalar>(sign);
            }
            target[output++] = result;

            if (!next_minor(minor)) {
                break;
            }
        }

        if (output != target.size()) {
            throw std::logic_error(
                "IncrementalMinors combination count disagrees with hash layout");
        }
    }

private:
    [[nodiscard]] static std::size_t checked_dimension(std::size_t dimension) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "IncrementalMinors dimension must be positive");
        }
        return dimension;
    }

    void require_order(std::size_t order) const {
        if (order == 0 || order > dimension_) {
            throw std::out_of_range(
                "IncrementalMinors order is outside [1, dimension]");
        }
    }

    [[nodiscard]] std::size_t& hash(std::size_t row, std::size_t column) {
        return hash_.at((row - 1) * dimension_ + (column - 1));
    }

    [[nodiscard]] std::size_t hash(
        std::size_t row,
        std::size_t column) const {
        return hash_.at((row - 1) * dimension_ + (column - 1));
    }

    void build_hash_matrix() {
        // Julia: for j in 1:dim HASH[1,j]=1 end
        for (std::size_t j = 1; j <= dimension_; ++j) {
            hash(1, j) = 1;
        }

        // Mechanical translation of hash_matrix(dim).
        for (std::size_t i = 2; i <= dimension_; ++i) {
            const std::size_t last_j = dimension_ - (i - 1);
            for (std::size_t j = 1; j <= last_j; ++j) {
                const std::size_t last_l = last_j + 1;
                for (std::size_t l = j + 1; l <= last_l; ++l) {
                    hash(i, j) += hash(i - 1, l);
                }
            }
        }
    }

    [[nodiscard]] std::size_t hash_column(std::size_t column) const {
        std::size_t sum = 0;
        for (std::size_t row = 1; row <= dimension_; ++row) {
            sum += hash(column, row);
        }
        return sum;
    }

    /** Julia next_minor(minor, dim); minor entries remain one-based. */
    [[nodiscard]] bool next_minor(std::vector<std::size_t>& minor) const {
        const std::size_t k = minor.size();
        std::size_t column = k;

        for (;;) {
            if (column == 0) {
                return false;
            }

            ++minor[column - 1];
            if (minor[column - 1] <= dimension_ - k + column) {
                for (std::size_t c = column + 1; c <= k; ++c) {
                    minor[c - 1] = minor[c - 2] + 1;
                }
                return true;
            }
            --column;
        }
    }

    /**
     * Mechanical zero-based-storage version of Julia get_minor().
     * `indices` and `skip` use Julia's one-based convention. The returned
     * value is converted from Julia's one-based storage index to zero-based.
     */
    [[nodiscard]] std::size_t get_minor_index(
        const std::vector<std::size_t>& indices,
        std::size_t skip,
        std::size_t top_column) const {
        std::size_t index = 1;
        std::size_t storage = 0;
        std::size_t column = top_column;
        std::size_t old_row = 0;

        while (column > 0) {
            if (index == skip) {
                ++index;
            }

            const std::size_t row = indices.at(index - 1);
            for (std::size_t i = old_row + 1; i < row; ++i) {
                storage += hash(column, i);
            }
            old_row = row;
            if (column == 1) {
                ++storage;
            }
            --column;
            ++index;
        }

        if (storage == 0) {
            throw std::logic_error(
                "IncrementalMinors produced Julia storage index zero");
        }
        return storage - 1;
    }

    std::size_t dimension_;
    std::vector<std::size_t> hash_;
    std::vector<std::vector<Scalar>> data_;
    std::vector<std::vector<std::size_t>> buffers_;
};

} // namespace highvoronoi::detail
