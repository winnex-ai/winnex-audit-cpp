#include "winnex_audit/engine.h"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    int64_t N = (argc > 1) ? std::stoll(argv[1]) : 50000;
    int64_t nq = (argc > 2) ? std::stoll(argv[2]) : 200;
    int64_t k = 10;
    int64_t dim = 128;

    std::cout << "Winnex Audit C++ Benchmark\n";
    std::cout << "N=" << N << " dim=" << dim << " queries=" << nq << " k=" << k << "\n";
    std::cout << std::string(60, '=') << "\n";

    // Generate random unit sphere data
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vectors(N * dim);
    for (int64_t i = 0; i < N * dim; ++i) vectors[i] = dist(rng);
    // Normalize rows
    for (int64_t i = 0; i < N; ++i) {
        float norm = 0;
        for (int64_t j = 0; j < dim; ++j)
            norm += vectors[i * dim + j] * vectors[i * dim + j];
        norm = std::sqrt(norm);
        for (int64_t j = 0; j < dim; ++j)
            vectors[i * dim + j] /= norm;
    }

    // Generate queries
    std::vector<float> queries(nq * dim);
    for (int64_t i = 0; i < nq * dim; ++i) queries[i] = dist(rng);
    for (int64_t i = 0; i < nq; ++i) {
        float norm = 0;
        for (int64_t j = 0; j < dim; ++j)
            norm += queries[i * dim + j] * queries[i * dim + j];
        norm = std::sqrt(norm);
        for (int64_t j = 0; j < dim; ++j)
            queries[i * dim + j] /= norm;
    }

    // Build engine
    winnex::Config cfg;
    cfg.input_dim = dim;
    cfg.stage1_dim = 64;
    cfg.stage2_dim = dim;
    cfg.final_k = k;
    cfg.random_seed = 42;

    winnex::AuditEngine engine(cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    engine.build(vectors.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "  Build: " << build_s << "s\n";

    // Run queries
    double total_latency = 0;
    int64_t total_viol_64d = 0;
    int64_t total_viol_128d = 0;

    for (int64_t qi = 0; qi < nq; ++qi) {
        auto result = engine.search(queries.data() + qi * dim, k);

        auto [v64, v128] = engine.check_bounds(queries.data() + qi * dim);
        total_viol_64d += v64;
        total_viol_128d += v128;

        total_latency += result.latency_ms;
    }

    double avg_latency = total_latency / nq;
    std::cout << "  Avg query: " << avg_latency << " ms\n";
    std::cout << "  Violations (64D): " << total_viol_64d << " / " << N * nq << " pairs\n";
    std::cout << "  Violations (128D): " << total_viol_128d << " / " << N * nq << " pairs\n";
    std::cout << std::string(60, '=') << "\n";

    // Audit JSON example
    std::cout << "\nAudit JSON sample:\n";
    std::cout << engine.audit_json(queries.data(), k) << "\n";

    // Show per-document proof for first excluded doc
    auto sample_result = engine.search(queries.data(), k);
    for (const auto& rec : sample_result.audit) {
        if (rec.excluded) {
            std::cout << "Sample proof for excluded doc #" << rec.doc_id << ":\n";
            std::cout << "  True cosine:         " << std::fixed << std::setprecision(4)
                      << rec.true_cosine << "\n";
            std::cout << "  Projected cosine:    " << rec.projected_cosine << "\n";
            std::cout << "  Pythagorean residual: " << rec.residual_norm << "\n";
            std::cout << "  Cauchy-Schwarz bound: " << rec.upper_bound << "\n";
            std::cout << "  Threshold:            " << rec.threshold << "\n";
            std::cout << "  Verdict:              "
                      << (rec.excluded ? "EXCLUDED (mathematically proven)" : "SURVIVED")
                      << "\n";
            break;
        }
    }

    return 0;
}
