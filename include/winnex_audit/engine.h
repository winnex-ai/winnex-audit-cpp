#ifndef WINNEX_AUDIT_ENGINE_H
#define WINNEX_AUDIT_ENGINE_H

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <Eigen/Dense>

namespace winnex {

/// Per-document audit record
struct AuditRecord {
    int64_t doc_id;
    double true_cosine;         // true cosine similarity <v, q>
    double projected_cosine;    // projected <Pv, Pq>
    double residual_norm;       // ||v - P^T P v|| * ||q - P^T P q||
    double upper_bound;         // Cauchy-Schwarz upper bound
    double threshold;           // score of the K-th result
    bool excluded;              // true if upper_bound < threshold
    std::string stage;          // which stage pruned this doc
};

/// Search result with audit trail
struct AuditResult {
    std::vector<int64_t> indices;       // top-K result indices
    std::vector<double> scores;         // corresponding scores
    std::vector<AuditRecord> audit;     // per-document proofs (excluded docs)
    double latency_ms;                  // query latency
    struct {
        int64_t stage1_survivors;
        int64_t stage2_survivors;
        std::vector<int64_t> violations_64d;
        std::vector<int64_t> violations_128d;
    } profile;
};

/// Configuration for engine
struct Config {
    int64_t input_dim = 128;
    int64_t stage1_dim = 64;
    int64_t stage2_dim = 128;
    int64_t final_k = 10;
    int64_t stage2_topk = 500;
    double keep_base = 0.25;
    double keep_min = 0.05;
    double keep_max = 0.50;
    double bounds_sensitivity = 0.12;
    double ortho_tolerance = 1e-5;
    int64_t random_seed = 42;
    bool verbose = false;
};

/// Madhava Audit Engine: configuration, build, search, verify
class AuditEngine {
public:
    explicit AuditEngine(const Config& cfg = Config{});
    ~AuditEngine();

    // Non-copyable
    AuditEngine(const AuditEngine&) = delete;
    AuditEngine& operator=(const AuditEngine&) = delete;

    /// Build index from dense float vectors. Thread-safe after build.
    /// @param data  Row-major matrix of shape (N, input_dim)
    /// @param N     Number of vectors
    void build(const float* data, int64_t N);

    /// Search with full audit trail
    /// @param query       Query vector (input_dim elements)
    /// @param k           Number of results to return
    /// @return            AuditResult with indices, scores, audit records
    AuditResult search(const float* query, int64_t k = 10);

    /// Verify bound violations for a query
    /// @param query  Query vector
    /// @return       Pair of (violations_64d, violations_128d)
    std::pair<int64_t, int64_t> check_bounds(const float* query);

    /// Get total vectors in index
    int64_t size() const { return N_; }

    /// Get build time in seconds
    double build_time() const { return build_time_; }

    /// Generate JSON audit trail for a query
    std::string audit_json(const float* query, int64_t k = 10);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    Config cfg_;
    int64_t N_ = 0;
    double build_time_ = 0;
};

} // namespace winnex
#endif // WINNEX_AUDIT_ENGINE_H
