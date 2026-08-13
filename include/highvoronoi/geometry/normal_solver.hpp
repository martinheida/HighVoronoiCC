#pragma once

/**
 * @file normal_solver.hpp
 * @brief Small reusable Householder-QR normal solvers.
 *
 * These classes provide reusable QR workspaces for the Julia-compatible
 * normal calculations and the direct Voronoi-vertex correction.
 *
 * `QRNormalSolver` performs the factorization in `Scalar`.
 * `ExtendedQRNormalSolver` performs the same factorization in `ExtendedFloat`
 * and converts only the final result back to `Scalar`.
 *
 * No Gram-Schmidt iteration belongs here.  The iterative orthogonalization of
 * the degenerate edge algorithm is implemented by `EdgeIterator` itself.
 */

#include <highvoronoi/detail/float.hpp>
#include <highvoronoi/geometry/point.hpp>

#include <Eigen/Core>
#include <Eigen/QR>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace highvoronoi {

namespace detail {

template <typename Scalar, int Dim>
using NormalPoint = std::conditional_t<
    Dim == Dynamic,
    DynamicPoint<Scalar>,
    StaticPoint<Scalar, Dim>>;

template <typename Scalar, int Dim>
using NormalMatrix = Eigen::Matrix<Scalar, Dim, Dim>;

template <typename Scalar, int Dim>
[[nodiscard]] NormalPoint<Scalar, Dim>
make_normal_point(std::size_t dimension) {
    if constexpr (Dim == Dynamic) {
        return NormalPoint<Scalar, Dim>(
            static_cast<Eigen::Index>(dimension));
    } else {
        if (dimension != static_cast<std::size_t>(Dim)) {
            throw std::invalid_argument(
                "Normal solver dimension does not match fixed Dim.");
        }
        return NormalPoint<Scalar, Dim>{};
    }
}

template <typename Scalar, int Dim>
[[nodiscard]] NormalMatrix<Scalar, Dim>
make_normal_matrix(std::size_t dimension) {
    if constexpr (Dim == Dynamic) {
        return NormalMatrix<Scalar, Dim>(
            static_cast<Eigen::Index>(dimension),
            static_cast<Eigen::Index>(dimension));
    } else {
        if (dimension != static_cast<std::size_t>(Dim)) {
            throw std::invalid_argument(
                "Normal solver dimension does not match fixed Dim.");
        }
        return NormalMatrix<Scalar, Dim>{};
    }
}

template <typename OutputScalar, typename WorkScalar, int Dim>
class QRNormalSolverCore {
public:
    using Scalar = OutputScalar;
    using Point = NormalPoint<Scalar, Dim>;
    using WorkPoint = NormalPoint<WorkScalar, Dim>;
    using WorkMatrix = NormalMatrix<WorkScalar, Dim>;

    explicit QRNormalSolverCore(std::size_t dimension)
        : dimension_(checked_dimension(dimension)),
          matrix_(make_normal_matrix<WorkScalar, Dim>(dimension_)),
          origin_(make_normal_point<WorkScalar, Dim>(dimension_)),
          node_(make_normal_point<WorkScalar, Dim>(dimension_)),
          q_last_(make_normal_point<WorkScalar, Dim>(dimension_)),
          scalar_buffer_(make_normal_point<Scalar, Dim>(dimension_)) {}

    /**
     * Reproduce Julia `u_qr(sig, xs, omitted_position)`.
     *
     * `sigma` contains exactly `dimension + 1` generators.  The selected
     * generator is omitted from the affine span and used only to orient the
     * resulting normal away from it.
     */
    template <class Nodes, class Sigma>
    [[nodiscard]] bool solve(
        const Nodes& nodes,
        const Sigma& sigma,
        std::size_t omitted_position,
        Point& direction) {

        if (static_cast<std::size_t>(nodes.dimension()) != dimension_) {
            throw std::invalid_argument(
                "Node dimension does not match QRNormalSolver.");
        }
        if (static_cast<std::size_t>(sigma.size()) != dimension_ + 1) {
            throw std::invalid_argument(
                "u_qr requires exactly dimension + 1 generators.");
        }
        if (omitted_position >= static_cast<std::size_t>(sigma.size())) {
            throw std::out_of_range(
                "Omitted generator position is outside sigma.");
        }

        resize_output(direction);

        // Julia first writes every non-omitted node into X, takes the last
        // such node as origin, replaces that final column by the omitted node,
        // and subtracts origin from every column.
        std::size_t column = 0;
        for (std::size_t position = 0;
             position < static_cast<std::size_t>(sigma.size());
             ++position) {
            if (position == omitted_position) {
                continue;
            }

            copy_node(nodes, sigma[position], node_);
            matrix_.col(static_cast<Eigen::Index>(column)) = node_;
            ++column;
        }

        origin_ = matrix_.col(static_cast<Eigen::Index>(dimension_ - 1));
        copy_node(nodes, sigma[omitted_position], node_);
        matrix_.col(static_cast<Eigen::Index>(dimension_ - 1)) = node_;

        for (std::size_t c = 0; c < dimension_; ++c) {
            matrix_.col(static_cast<Eigen::Index>(c)) -= origin_;
        }

        return factor_and_extract(direction);
    }

    /** Reproduce Julia `u_qr_onb(onb, x0)`. `x0` is unused in Julia. */
    template <class Basis>
    [[nodiscard]] bool solve_basis(
        const Basis& basis,
        Point& direction) {

        if (static_cast<std::size_t>(basis.size()) != dimension_) {
            throw std::invalid_argument(
                "u_qr_onb requires exactly dimension basis vectors.");
        }

        resize_output(direction);
        for (std::size_t column = 0; column < dimension_; ++column) {
            const auto& source = basis[column];
            if (static_cast<std::size_t>(source.size()) != dimension_) {
                throw std::invalid_argument(
                    "Basis-vector dimension does not match QRNormalSolver.");
            }
            for (std::size_t coordinate = 0;
                 coordinate < dimension_;
                 ++coordinate) {
                matrix_(static_cast<Eigen::Index>(coordinate),
                        static_cast<Eigen::Index>(column)) =
                    static_cast<WorkScalar>(
                        source[static_cast<Eigen::Index>(coordinate)]);
            }
        }

        return factor_and_extract(direction);
    }

    /** Solve the d x d equal-distance system of a Voronoi vertex. */
    template <class Nodes, class Support>
    [[nodiscard]] bool solve_vertex(
        const Nodes& nodes,
        const Support& support,
        Point& vertex) {

        if (static_cast<std::size_t>(nodes.dimension()) != dimension_) {
            throw std::invalid_argument(
                "Node dimension does not match QRNormalSolver.");
        }
        if (static_cast<std::size_t>(support.size()) != dimension_ + 1) {
            throw std::invalid_argument(
                "Vertex solve requires exactly dimension + 1 generators.");
        }

        resize_output(vertex);
        copy_node(nodes, support[dimension_], origin_);
        const WorkScalar reference_norm = origin_.squaredNorm();

        for (std::size_t row = 0; row < dimension_; ++row) {
            copy_node(nodes, support[row], node_);
            matrix_.row(static_cast<Eigen::Index>(row)) =
                (node_ - origin_).transpose();
            q_last_[static_cast<Eigen::Index>(row)] =
                WorkScalar{0.5} * (node_.squaredNorm() - reference_norm);
        }

        qr_.compute(matrix_);
        for (std::size_t i = 0; i < dimension_; ++i) {
            const auto pivot = qr_.matrixQR()(
                static_cast<Eigen::Index>(i),
                static_cast<Eigen::Index>(i));
            if (!finite(pivot) || pivot == WorkScalar{0}) {
                return false;
            }
        }

        node_ = qr_.solve(q_last_);
        for (std::size_t c = 0; c < dimension_; ++c) {
            const WorkScalar value = node_[static_cast<Eigen::Index>(c)];
            if (!finite(value)) {
                return false;
            }
            vertex[static_cast<Eigen::Index>(c)] = static_cast<Scalar>(value);
        }
        return true;
    }

    [[nodiscard]] std::size_t dimension() const noexcept {
        return dimension_;
    }

private:
    [[nodiscard]] static std::size_t checked_dimension(
        std::size_t dimension) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "Normal solver dimension must be positive.");
        }
        if constexpr (Dim != Dynamic) {
            if (dimension != static_cast<std::size_t>(Dim)) {
                throw std::invalid_argument(
                    "Normal solver runtime dimension does not match Dim.");
            }
        }
        return dimension;
    }

    void resize_output(Point& direction) const {
        if constexpr (Dim == Dynamic) {
            direction.resize(static_cast<Eigen::Index>(dimension_));
        }
    }

    template <class Nodes, class Index>
    void copy_node(
        const Nodes& nodes,
        Index index,
        WorkPoint& target) {

        if constexpr (std::is_same_v<WorkScalar, Scalar>) {
            nodes.copy_node(index, target.data());
        } else {
            nodes.copy_node(index, scalar_buffer_.data());
            for (std::size_t coordinate = 0;
                 coordinate < dimension_;
                 ++coordinate) {
                target[static_cast<Eigen::Index>(coordinate)] =
                    static_cast<WorkScalar>(
                        scalar_buffer_[static_cast<Eigen::Index>(coordinate)]);
            }
        }
    }

    [[nodiscard]] bool factor_and_extract(Point& direction) {
        qr_.compute(matrix_);

        const Eigen::Index last =
            static_cast<Eigen::Index>(dimension_ - 1);
        const WorkScalar pivot = qr_.matrixQR()(last, last);

        if (!finite(pivot) || pivot == WorkScalar{0}) {
            direction.setZero();
            return false;
        }

        q_last_.setZero();
        q_last_[last] = WorkScalar{1};
        q_last_ = qr_.householderQ() * q_last_;

        // Julia: u = -Q[:,end] * sign(R[end,end])
        if (pivot > WorkScalar{0}) {
            q_last_ = -q_last_;
        }

        for (std::size_t coordinate = 0;
             coordinate < dimension_;
             ++coordinate) {
            direction[static_cast<Eigen::Index>(coordinate)] =
                static_cast<Scalar>(
                    q_last_[static_cast<Eigen::Index>(coordinate)]);
        }

        const Scalar norm = direction.norm();
        if (!(norm > Scalar{0}) || !finite(norm)) {
            direction.setZero();
            return false;
        }
        direction /= norm;
        return true;
    }

    template <class T>
    [[nodiscard]] static bool finite(const T& value) {
        const T maximum = (std::numeric_limits<T>::max)();
        return value == value && value <= maximum && value >= -maximum;
    }

    std::size_t dimension_;
    WorkMatrix matrix_;
    WorkPoint origin_;
    WorkPoint node_;
    WorkPoint q_last_;
    Point scalar_buffer_;
    Eigen::HouseholderQR<WorkMatrix> qr_;
};

} // namespace detail

/** Ordinary-precision Householder QR matching Julia `u_qr`/`u_qr_onb`. */
template <typename Scalar, int Dim>
class QRNormalSolver final {
public:
    using Point = detail::NormalPoint<Scalar, Dim>;

    explicit QRNormalSolver(std::size_t dimension)
        : core_(dimension) {}

    template <class Nodes, class Sigma>
    [[nodiscard]] bool solve(
        const Nodes& nodes,
        const Sigma& sigma,
        std::size_t omitted_position,
        Point& direction) {
        return core_.solve(nodes, sigma, omitted_position, direction);
    }

    template <class Basis>
    [[nodiscard]] bool solve_basis(
        const Basis& basis,
        Point& direction) {
        return core_.solve_basis(basis, direction);
    }

    template <class Nodes, class Support>
    [[nodiscard]] bool solve_vertex(
        const Nodes& nodes,
        const Support& support,
        Point& vertex) {
        return core_.solve_vertex(nodes, support, vertex);
    }

private:
    detail::QRNormalSolverCore<Scalar, Scalar, Dim> core_;
};

/** Same QR algorithm, evaluated in HighVoronoi's `ExtendedFloat`. */
template <typename Scalar, int Dim>
class ExtendedQRNormalSolver final {
public:
    using Point = detail::NormalPoint<Scalar, Dim>;

    explicit ExtendedQRNormalSolver(std::size_t dimension)
        : core_(dimension) {}

    template <class Nodes, class Sigma>
    [[nodiscard]] bool solve(
        const Nodes& nodes,
        const Sigma& sigma,
        std::size_t omitted_position,
        Point& direction) {
        return core_.solve(nodes, sigma, omitted_position, direction);
    }

    template <class Basis>
    [[nodiscard]] bool solve_basis(
        const Basis& basis,
        Point& direction) {
        return core_.solve_basis(basis, direction);
    }

    template <class Nodes, class Support>
    [[nodiscard]] bool solve_vertex(
        const Nodes& nodes,
        const Support& support,
        Point& vertex) {
        return core_.solve_vertex(nodes, support, vertex);
    }

private:
    detail::QRNormalSolverCore<Scalar, ExtendedFloat, Dim> core_;
};

} // namespace highvoronoi
