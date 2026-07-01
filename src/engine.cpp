#include "winnex_audit/engine.h"
#include "winnex_audit/projection.h"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>

namespace winnex {

struct AuditEngine::Impl {
    Config cfg;
    Eigen::MatrixXf vectors;                    // (N, dim)
    OrthogonalProjection proj_stage1;           // dim -> stage1_dim
    OrthogonalProjection proj_stage2;           // dim -> stage2_dim
    Eigen::MatrixXf proj1_cache;                // projected vectors stage1
    Eigen::MatrixXf proj2_cache;                // projected vectors stage2
    Eigen::VectorXd error1;                     // residuals stage1
    Eigen::VectorXd error2;                     // residuals stage2
    int64_t N = 0;

    void compute_upper_bounds(
        const Eigen::VectorXf& q,
        const Eigen::MatrixXf& proj_cache,
        const Eigen::VectorXd& errors,
        const OrthogonalProjection& proj,
        Eigen::VectorXd& bounds_out)
    {
        auto pq = proj.project_vector(q);
        double q_res = ResidualComputer::query_residual(q, proj);

        int64_t n = proj_cache.rows();
        bounds_out.resize(n);

#pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            double pv_dot_pq = proj_cache.row(i).dot(pq);
            bounds_out(i) = pv_dot_pq + errors(i) * q_res + 1e-5;
        }
    }

    void adaptive_partition(
        const Eigen::VectorXd& bounds,
        double& keep_ratio_out,
        int64_t& k1_out)
    {
        double b_min = bounds.minCoeff();
        double b_max = bounds.maxCoeff();
        double b_range = b_max - b_min;

        double raw_keep = cfg.keep_base * cfg.bounds_sensitivity
                        / std::max(b_range, 0.01);
        keep_ratio_out = std::min(cfg.keep_max,
                         std::max(cfg.keep_min, raw_keep));

        k1_out = std::min(
            std::max(static_cast<int64_t>(N * keep_ratio_out), int64_t(100)),
            N);
    }

    std::vector<int64_t> top_k_indices(
        const Eigen::VectorXd& scores,
        int64_t k)
    {
        int64_t n = scores.size();
        k = std::min(k, n);

        std::vector<int64_t> idx(n);
        std::iota(idx.begin(), idx.end(), int64_t(0));

        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int64_t a, int64_t b) { return scores(a) > scores(b); });

        idx.resize(k);
        return idx;
    }
};

// ----- AuditEngine -----

AuditEngine::AuditEngine(const Config& cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>())
{
    impl_->cfg = cfg;
}

AuditEngine::~AuditEngine() = default;

void AuditEngine::build(const float* data, int64_t N) {
    auto t0 = std::chrono::high_resolution_clock::now();

    N_ = N;
    impl_->N = N;

    // Copy data to Eigen matrix (N, input_dim)
    impl_->vectors = Eigen::Map<const Eigen::MatrixXf>(
        data, cfg_.input_dim, N).transpose();

    // Build orthogonal projections
    impl_->proj_stage1.build(cfg_.input_dim, cfg_.stage1_dim, cfg_.random_seed);
    impl_->proj_stage2.build(cfg_.input_dim, cfg_.stage2_dim, cfg_.random_seed + 1);

    // Precompute projections
    impl_->proj1_cache = impl_->proj_stage1.project(impl_->vectors);
    impl_->proj2_cache = impl_->proj_stage2.project(impl_->vectors);

    // Precompute residuals
    impl_->error1 = ResidualComputer::compute_residuals(
        impl_->vectors, impl_->proj_stage1);
    impl_->error2 = ResidualComputer::compute_residuals(
        impl_->vectors, impl_->proj_stage2);

    auto t1 = std::chrono::high_resolution_clock::now();
    build_time_ = std::chrono::duration<double>(t1 - t0).count();
}

AuditResult AuditEngine::search(const float* query, int64_t k) {
    AuditResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    Eigen::Map<const Eigen::VectorXf> q(query, cfg_.input_dim);
    int64_t final_k = std::min(k, cfg_.final_k);

    // ---- Stage 1: bound ALL vectors ----
    Eigen::VectorXd B1;
    impl_->compute_upper_bounds(q, impl_->proj1_cache,
                                impl_->error1, impl_->proj_stage1, B1);

    double keep_ratio;
    int64_t k1;
    impl_->adaptive_partition(B1, keep_ratio, k1);

    auto idx1 = impl_->top_k_indices(B1, k1);
    result.profile.stage1_survivors = idx1.size();

    // ---- Stage 2: refine survivors ----
    int64_t n2 = idx1.size();
    Eigen::VectorXd B2(n2);
    auto pq2 = impl_->proj_stage2.project_vector(q);
    double q_res2 = ResidualComputer::query_residual(q, impl_->proj_stage2);

#pragma omp parallel for
    for (int64_t i = 0; i < n2; ++i) {
        int64_t vi = idx1[i];
        double pv_dot_pq = impl_->proj2_cache.row(vi).dot(pq2);
        B2(i) = pv_dot_pq + impl_->error2(vi) * q_res2 + 1e-5;
    }

    // Error backpropagation modulation
    Eigen::VectorXd scores(n2);
    for (int64_t i = 0; i < n2; ++i) {
        int64_t vi = idx1[i];
        double e1 = impl_->error1(vi);
        double e2 = impl_->error2(vi);
        double alpha = 1.0 / (1.0 + std::exp(
            -(e1 - e2) / std::max(e1 / n2, 1e-9) * 0.5));
        alpha = std::clamp(alpha, 0.01, 0.99);
        scores(i) = B1(vi) + alpha * (B2(i) - B1(vi));
    }

    // Select top candidates
    auto idx2 = impl_->top_k_indices(scores, cfg_.stage2_topk);
    result.profile.stage2_survivors = idx2.size();

    // ---- Stage 3: exact cosine on survivors ----
    int64_t n3 = idx2.size();
    std::vector<std::pair<int64_t, double>> candidates;
    candidates.reserve(n3);

    for (int64_t i = 0; i < n3; ++i) {
        int64_t vi = idx1[idx2[i]];
        double cos = impl_->vectors.row(vi).dot(q);
        candidates.emplace_back(vi, cos);
    }

    std::partial_sort(candidates.begin(),
                      candidates.begin() + std::min(final_k, (int64_t)candidates.size()),
                      candidates.end(),
                      [](const auto& a, const auto& b) {
                          return a.second > b.second;
                      });

    // Build result
    result.indices.reserve(final_k);
    result.scores.reserve(final_k);
    for (int64_t i = 0; i < std::min(final_k, (int64_t)candidates.size()); ++i) {
        result.indices.push_back(candidates[i].first);
        result.scores.push_back(candidates[i].second);
    }

    // ---- Build audit trail for excluded docs (optimized) ----
    double threshold = result.scores.empty() ? 0.0 : result.scores.back();
    result.audit.clear();
    result.audit.reserve(std::min(int64_t(200), N_));

    // Precompute query projections once (prefix a_ to avoid shadowing outer variables)
    Eigen::VectorXf a_pq1 = impl_->proj_stage1.project_vector(q);
    Eigen::VectorXf a_pq2 = impl_->proj_stage2.project_vector(q);
    double a_res1 = ResidualComputer::query_residual(q, impl_->proj_stage1);
    double a_res2 = ResidualComputer::query_residual(q, impl_->proj_stage2);

    // Only audit docs near the boundary
    for (int64_t i = 0; i < std::min(int64_t(500), N_); ++i) {
        double true_cos = impl_->vectors.row(i).dot(q);
        if (true_cos < threshold - 0.15) continue;

        AuditRecord rec;
        rec.doc_id = i;
        rec.true_cosine = true_cos;
        rec.projected_cosine = impl_->proj1_cache.row(i).dot(a_pq1);
        rec.residual_norm = impl_->error1(i) * a_res1;
        rec.upper_bound = rec.projected_cosine + rec.residual_norm + 1e-5;
        rec.threshold = threshold;
        rec.excluded = rec.upper_bound < threshold;

        if (rec.excluded) {
            rec.stage = "stage1";
        } else {
            double pv2 = impl_->proj2_cache.row(i).dot(a_pq2);
            double ub2 = pv2 + impl_->error2(i) * a_res2 + 1e-5;
            rec.stage = (ub2 < threshold) ? "stage2" : "survived";
        }

        result.audit.push_back(rec);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
}

std::pair<int64_t, int64_t> AuditEngine::check_bounds(const float* query) {
    Eigen::Map<const Eigen::VectorXf> q(query, cfg_.input_dim);

    auto true_cos = impl_->vectors * q;
    int64_t viol_64d = 0, viol_128d = 0;

    // Check stage 1 (64D)
    auto pq1 = impl_->proj_stage1.project_vector(q);
    double q_res1 = ResidualComputer::query_residual(q, impl_->proj_stage1);

    for (int64_t i = 0; i < N_; ++i) {
        double pv_pq = impl_->proj1_cache.row(i).dot(pq1);
        double ub = pv_pq + impl_->error1(i) * q_res1 + 1e-5;
        if (true_cos(i) > ub + 1e-9) viol_64d++;
    }

    // Check stage 2 (128D)
    auto pq2 = impl_->proj_stage2.project_vector(q);
    double q_res2 = ResidualComputer::query_residual(q, impl_->proj_stage2);

    for (int64_t i = 0; i < N_; ++i) {
        double pv_pq = impl_->proj2_cache.row(i).dot(pq2);
        double ub = pv_pq + impl_->error2(i) * q_res2 + 1e-5;
        if (true_cos(i) > ub + 1e-9) viol_128d++;
    }

    return {viol_64d, viol_128d};
}

std::string AuditEngine::audit_json(const float* query, int64_t k) {
    auto result = search(query, k);
    std::string json = "{\n  \"query_id\": \"audit_001\",\n";
    json += "  \"latency_ms\": " + std::to_string(result.latency_ms) + ",\n";
    json += "  \"results\": [";
    for (size_t i = 0; i < result.indices.size(); ++i) {
        if (i > 0) json += ",";
        json += "\n    {\"rank\": " + std::to_string(i + 1) +
                ", \"doc_id\": " + std::to_string(result.indices[i]) +
                ", \"score\": " + std::to_string(result.scores[i]) + "}";
    }
    json += "\n  ],\n  \"audit_trail\": [\n";
    int count = 0;
    for (const auto& rec : result.audit) {
        if (count++ > 0) json += ",\n";
        json += "    {\"doc_id\": " + std::to_string(rec.doc_id) +
                ", \"true_cosine\": " + std::to_string(rec.true_cosine) +
                ", \"upper_bound\": " + std::to_string(rec.upper_bound) +
                ", \"threshold\": " + std::to_string(rec.threshold) +
                ", \"excluded\": " + (rec.excluded ? "true" : "false") +
                ", \"stage\": \"" + rec.stage + "\"}";
    }
    json += "\n  ]\n}\n";
    return json;
}

} // namespace winnex
