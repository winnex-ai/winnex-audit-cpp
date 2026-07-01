#include "winnex_audit/projection.h"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <random>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace winnex {

void OrthogonalProjection::build(int64_t d_in, int64_t d_out, int64_t seed) {
    if (d_out > d_in) {
        throw std::invalid_argument(
            "Output dimension cannot exceed input dimension");
    }

    // Generate random matrix R of shape (d_in, d_out)
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXd R(d_in, d_out);
    for (int64_t i = 0; i < d_in * d_out; ++i) {
        R.data()[i] = dist(rng);
    }

    // QR decomposition: R = Q * R_qr
    // Q has shape (d_in, d_in), R_qr has shape (d_in, d_out)
    auto qr = R.householderQr();
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(d_in, d_in);

    // P = Q.topRows(d_out)^T = Q^T.topRows(d_out) but we want (d_out, d_in)
    // Actually: P = Q[:, :d_out]^T has shape (d_out, d_in)
    P_ = Q.leftCols(d_out).transpose().cast<float>();

    // Verify orthogonality
    double err = orthogonality_error();
    if (err > 1e-5) {
        throw std::runtime_error(
            "QR orthogonality failed: " + std::to_string(err));
    }
}

Eigen::MatrixXf OrthogonalProjection::project(const Eigen::MatrixXf& data) const {
    return data * P_.transpose();  // (N, d_in) @ (d_in, d_out) = (N, d_out)
}

Eigen::VectorXf OrthogonalProjection::project_vector(const Eigen::VectorXf& v) const {
    return v.transpose() * P_.transpose();  // (d_in,) -> (d_out,)
}

double OrthogonalProjection::orthogonality_error() const {
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(P_.rows(), P_.rows());
    return (P_ * P_.transpose() - I).cwiseAbs().maxCoeff();
}

// ----- ResidualComputer -----

Eigen::VectorXd ResidualComputer::compute_residuals(
    const Eigen::MatrixXf& vectors,
    const OrthogonalProjection& proj)
{
    int64_t N = vectors.rows();
    Eigen::VectorXd residuals(N);

    auto projected = proj.project(vectors);

    for (int64_t i = 0; i < N; ++i) {
        double v_norm_sq = vectors.row(i).squaredNorm();
        double pv_norm_sq = projected.row(i).squaredNorm();
        double residual_sq = std::max(0.0, v_norm_sq - pv_norm_sq);
        residuals(i) = std::sqrt(residual_sq);
    }

    return residuals;
}

double ResidualComputer::query_residual(
    const Eigen::VectorXf& query,
    const OrthogonalProjection& proj)
{
    auto pq = proj.project_vector(query);
    double q_norm_sq = query.squaredNorm();
    double pq_norm_sq = pq.squaredNorm();
    double residual_sq = std::max(0.0, q_norm_sq - pq_norm_sq);
    return std::sqrt(residual_sq);
}

} // namespace winnex
