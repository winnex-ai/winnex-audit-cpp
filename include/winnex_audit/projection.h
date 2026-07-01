#ifndef WINNEX_AUDIT_PROJECTION_H
#define WINNEX_AUDIT_PROJECTION_H

#include <vector>
#include <Eigen/Dense>

namespace winnex {

/// QR-orthogonalized random projection matrix
/// P has shape (d_out, d_in) with P * P^T = I (orthonormal rows)
class OrthogonalProjection {
public:
    OrthogonalProjection() = default;

    /// Build projection: d_in -> d_out with QR orthogonality
    /// @param d_in   Input dimension
    /// @param d_out  Output dimension
    /// @param seed   Random seed for reproducibility
    void build(int64_t d_in, int64_t d_out, int64_t seed = 42);

    /// Project vectors: data @ P^T  (batched)
    /// @param data  Matrix (N, d_in)
    /// @return      Matrix (N, d_out)
    Eigen::MatrixXf project(const Eigen::MatrixXf& data) const;

    /// Project single vector
    Eigen::VectorXf project_vector(const Eigen::VectorXf& v) const;

    /// Verify orthogonality: must satisfy ||P*P^T - I|| < tolerance
    double orthogonality_error() const;

    const Eigen::MatrixXf& matrix() const { return P_; }
    int64_t d_in() const { return P_.cols(); }
    int64_t d_out() const { return P_.rows(); }

private:
    Eigen::MatrixXf P_;  // (d_out, d_in), rows are orthonormal
};

/// Compute Pythagorean residual: sqrt(||v||^2 - ||Pv||^2)
/// This is the "error" term in the Cauchy-Schwarz bound
class ResidualComputer {
public:
    /// Precompute norms and projected norms for all vectors
    /// @param vectors     (N, dim) matrix
    /// @param projection  Projection matrix
    /// @return            Residual vector of length N
    static Eigen::VectorXd compute_residuals(
        const Eigen::MatrixXf& vectors,
        const OrthogonalProjection& projection);

    /// Compute query residual: sqrt(||q||^2 - ||Pq||^2)
    static double query_residual(
        const Eigen::VectorXf& query,
        const OrthogonalProjection& projection);
};

} // namespace winnex
#endif // WINNEX_AUDIT_PROJECTION_H
