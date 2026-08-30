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

/**
 * @brief Diagnostic result of one iterative Voronoi-vertex correction.
 *
 * `pivot_ratio` is the smallest divided by the largest absolute diagonal
 * entry of the column-pivoted QR factor.  It is a cheap conditioning proxy,
 * not an exact reciprocal condition number.
 */
template <typename Scalar>
struct VertexCorrectionResult {
    bool converged = false;
    bool factorization_ok = false;
    bool poor_conditioning = false;
    std::size_t iterations = 0;
    Scalar pivot_ratio = Scalar{0};
    Scalar relative_correction =
        (std::numeric_limits<Scalar>::max)();
};

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
          vertex_rhs_(make_normal_point<WorkScalar, Dim>(dimension_)),
          vertex_solution_(make_normal_point<WorkScalar, Dim>(dimension_)),
          vertex_candidate_(make_normal_point<WorkScalar, Dim>(dimension_)),
          vertex_delta_(make_normal_point<WorkScalar, Dim>(dimension_)),
          vertex_row_scale_(make_normal_point<WorkScalar, Dim>(dimension_)),
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

    /**
     * @brief Solve a d+1-generator Voronoi vertex in local coordinates.
     *
     * This compatibility entry point still computes the complete vertex, but
     * unlike the old implementation it uses the translated equations
     *
     *   d_i^T y = 0.5 |d_i|^2,  d_i = p_i - p_0,  x = p_0 + y,
     *
     * row-normalizes them, and factorizes the coordinate columns with
     * ColPivHouseholderQR.  Column pivoting is safe for the vertex system:
     * columns are coordinate unknowns and carry no special edge semantics.
     */
    template <class Nodes, class Support>
    [[nodiscard]] bool solve_vertex(
        const Nodes& nodes,
        const Support& support,
        Point& vertex) {

        resize_output(vertex);
        if (!prepare_vertex_system(nodes, support)) {
            return false;
        }

        for (std::size_t row = 0; row < dimension_; ++row) {
            vertex_rhs_[static_cast<Eigen::Index>(row)] =
                WorkScalar{0.5} *
                vertex_row_scale_[static_cast<Eigen::Index>(row)];
        }

        vertex_solution_ = vertex_qr_.solve(vertex_rhs_);
        for (std::size_t c = 0; c < dimension_; ++c) {
            const WorkScalar value =
                origin_[static_cast<Eigen::Index>(c)] +
                vertex_solution_[static_cast<Eigen::Index>(c)];
            if (!finite(value)) {
                return false;
            }
            vertex[static_cast<Eigen::Index>(c)] =
                static_cast<Scalar>(value);
        }
        return true;
    }

    /**
     * @brief Improve an existing RayCaster vertex candidate by iterative
     *        correction with one reused column-pivoted QR factorization.
     *
     * The current point r is treated as data. For every support generator p_i
     * relative to p_0, the correction equation is
     *
     *   (p_i-p_0)^T delta
     *       = 0.5 ( |r-p_i|^2 - |r-p_0|^2 ).
     *
     * Rows are normalized by |p_i-p_0| before factorization.  The caller may
     * reject a poor pivot ratio immediately in ordinary precision; the
     * ExtendedFloat path can then solve the same correction without inserting
     * a platform-dependent intermediate precision.
     *
     * `relative_tolerance` is tested against |delta| / |r_initial-p_0|.
     * At most `max_iterations` corrections are attempted. A correction that
     * stops decreasing is reported as non-converged.
     */
    template <class Nodes, class Support>
    [[nodiscard]] VertexCorrectionResult<Scalar> correct_vertex(
        const Nodes& nodes,
        const Support& support,
        const Point& initial_vertex,
        Point& corrected_vertex,
        Scalar condition_tolerance,
        Scalar relative_tolerance,
        std::size_t max_iterations,
        bool reject_poor_conditioning) {

        VertexCorrectionResult<Scalar> result;

        if (!prepare_vertex_system(nodes, support)) {
            return result;
        }

        result.factorization_ok = true;
        result.pivot_ratio =
            static_cast<Scalar>(last_vertex_pivot_ratio_);
        result.poor_conditioning =
            last_vertex_pivot_ratio_ <
                static_cast<WorkScalar>(condition_tolerance);

        if (reject_poor_conditioning && result.poor_conditioning) {
            return result;
        }
        if (max_iterations == 0) {
            return result;
        }

        copy_point_to_work(initial_vertex, vertex_candidate_);
        vertex_delta_ = vertex_candidate_ - origin_;
        const WorkScalar reference_scale = vertex_delta_.norm();
        if (!(reference_scale > WorkScalar{0}) ||
            !finite(reference_scale)) {
            return result;
        }

        WorkScalar previous_relative =
            (std::numeric_limits<WorkScalar>::max)();

        for (std::size_t iteration = 0;
             iteration < max_iterations;
             ++iteration) {

            const WorkScalar reference_squared_distance =
                (vertex_candidate_ - origin_).squaredNorm();

            for (std::size_t row = 0; row < dimension_; ++row) {
                copy_node(nodes, support[row], node_);
                const WorkScalar candidate_squared_distance =
                    (vertex_candidate_ - node_).squaredNorm();

                vertex_rhs_[static_cast<Eigen::Index>(row)] =
                    WorkScalar{0.5} *
                    (candidate_squared_distance -
                     reference_squared_distance) /
                    vertex_row_scale_[static_cast<Eigen::Index>(row)];
            }

            vertex_solution_ = vertex_qr_.solve(vertex_rhs_);
            for (std::size_t c = 0; c < dimension_; ++c) {
                if (!finite(
                        vertex_solution_[static_cast<Eigen::Index>(c)])) {
                    return result;
                }
            }

            const WorkScalar correction_norm = vertex_solution_.norm();
            const WorkScalar relative =
                correction_norm / reference_scale;
            if (!finite(relative)) {
                return result;
            }

            vertex_candidate_ += vertex_solution_;
            result.iterations = iteration + 1;
            result.relative_correction =
                static_cast<Scalar>(relative);

            if (relative <= static_cast<WorkScalar>(relative_tolerance)) {
                copy_work_to_point(vertex_candidate_, corrected_vertex);
                result.converged = true;
                return result;
            }

            if (iteration > 0 && relative >= previous_relative) {
                return result;
            }
            previous_relative = relative;
        }

        return result;
    }

    [[nodiscard]] Scalar last_vertex_pivot_ratio() const {
        return static_cast<Scalar>(last_vertex_pivot_ratio_);
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

    template <class Nodes, class Support>
    [[nodiscard]] bool prepare_vertex_system(
        const Nodes& nodes,
        const Support& support) {

        if (static_cast<std::size_t>(nodes.dimension()) != dimension_) {
            throw std::invalid_argument(
                "Node dimension does not match QRNormalSolver.");
        }
        if (static_cast<std::size_t>(support.size()) != dimension_ + 1) {
            throw std::invalid_argument(
                "Vertex solve requires exactly dimension + 1 generators.");
        }

        copy_node(nodes, support[dimension_], origin_);

        for (std::size_t row = 0; row < dimension_; ++row) {
            copy_node(nodes, support[row], node_);
            vertex_delta_ = node_ - origin_;
            const WorkScalar row_scale = vertex_delta_.norm();

            if (!(row_scale > WorkScalar{0}) || !finite(row_scale)) {
                last_vertex_pivot_ratio_ = WorkScalar{0};
                return false;
            }

            vertex_row_scale_[static_cast<Eigen::Index>(row)] = row_scale;
            matrix_.row(static_cast<Eigen::Index>(row)) =
                (vertex_delta_ / row_scale).transpose();
        }

        vertex_qr_.compute(matrix_);

        WorkScalar minimum_pivot =
            (std::numeric_limits<WorkScalar>::max)();
        WorkScalar maximum_pivot = WorkScalar{0};

        for (std::size_t i = 0; i < dimension_; ++i) {
            const WorkScalar pivot = abs_value(
                vertex_qr_.matrixR()(
                    static_cast<Eigen::Index>(i),
                    static_cast<Eigen::Index>(i)));

            if (!finite(pivot)) {
                last_vertex_pivot_ratio_ = WorkScalar{0};
                return false;
            }

            minimum_pivot = std::min(minimum_pivot, pivot);
            maximum_pivot = std::max(maximum_pivot, pivot);
        }

        if (!(maximum_pivot > WorkScalar{0}) ||
            !(minimum_pivot > WorkScalar{0})) {
            last_vertex_pivot_ratio_ = WorkScalar{0};
            return false;
        }

        last_vertex_pivot_ratio_ = minimum_pivot / maximum_pivot;
        return finite(last_vertex_pivot_ratio_);
    }

    void copy_point_to_work(
        const Point& source,
        WorkPoint& target) const {
        for (std::size_t coordinate = 0;
             coordinate < dimension_;
             ++coordinate) {
            target[static_cast<Eigen::Index>(coordinate)] =
                static_cast<WorkScalar>(
                    source[static_cast<Eigen::Index>(coordinate)]);
        }
    }

    void copy_work_to_point(
        const WorkPoint& source,
        Point& target) const {
        resize_output(target);
        for (std::size_t coordinate = 0;
             coordinate < dimension_;
             ++coordinate) {
            target[static_cast<Eigen::Index>(coordinate)] =
                static_cast<Scalar>(
                    source[static_cast<Eigen::Index>(coordinate)]);
        }
    }

    template <class T>
    [[nodiscard]] static T abs_value(const T& value) {
        using std::abs;
        return abs(value);
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

    WorkPoint vertex_rhs_;
    WorkPoint vertex_solution_;
    WorkPoint vertex_candidate_;
    WorkPoint vertex_delta_;
    WorkPoint vertex_row_scale_;
    WorkScalar last_vertex_pivot_ratio_{WorkScalar{0}};

    Point scalar_buffer_;
    Eigen::HouseholderQR<WorkMatrix> qr_;
    Eigen::ColPivHouseholderQR<WorkMatrix> vertex_qr_;
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

    template <class Nodes, class Support>
    [[nodiscard]] VertexCorrectionResult<Scalar> correct_vertex(
        const Nodes& nodes,
        const Support& support,
        const Point& initial_vertex,
        Point& corrected_vertex,
        Scalar condition_tolerance,
        Scalar relative_tolerance,
        std::size_t max_iterations,
        bool reject_poor_conditioning = true) {
        return core_.correct_vertex(
            nodes,
            support,
            initial_vertex,
            corrected_vertex,
            condition_tolerance,
            relative_tolerance,
            max_iterations,
            reject_poor_conditioning);
    }

    [[nodiscard]] Scalar last_vertex_pivot_ratio() const {
        return core_.last_vertex_pivot_ratio();
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

    template <class Nodes, class Support>
    [[nodiscard]] VertexCorrectionResult<Scalar> correct_vertex(
        const Nodes& nodes,
        const Support& support,
        const Point& initial_vertex,
        Point& corrected_vertex,
        Scalar condition_tolerance,
        Scalar relative_tolerance,
        std::size_t max_iterations,
        bool reject_poor_conditioning = true) {
        return core_.correct_vertex(
            nodes,
            support,
            initial_vertex,
            corrected_vertex,
            condition_tolerance,
            relative_tolerance,
            max_iterations,
            reject_poor_conditioning);
    }

    [[nodiscard]] Scalar last_vertex_pivot_ratio() const {
        return core_.last_vertex_pivot_ratio();
    }

private:
    detail::QRNormalSolverCore<Scalar, ExtendedFloat, Dim> core_;
};

} // namespace highvoronoi
