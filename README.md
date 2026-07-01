# Winnex Audit Module

> **Mantenha seu banco vetorial. Adicione prova matemática.**  
> Keep your vector database. Add mathematical proof.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-yellow)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21107289.svg)](https://doi.org/10.5281/zenodo.21107289)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21106472-1682D4?logo=zenodo)](https://doi.org/10.5281/zenodo.21106472)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/winnex-ai/winnex-audit-cpp)
[![Kaggle](https://img.shields.io/badge/Kaggle-Benchmark-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/winnex-definitive-benchmark-v1-0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21088504.svg)](https://doi.org/10.5281/zenodo.21088504)

The Winnex Audit Module is a **drop-in compliance layer** for existing vector search infrastructure (FAISS, Pinecone, Milvus, Weaviate, Qdrant). It wraps any existing index with a **Cauchy-Schwarz bound verifier**, generating a **per-document mathematical audit trail** for every search query.

**Why this exists**: Black-box vector search (HNSW, IVF, PQ) cannot explain why a document was excluded from search results. For regulated industries (banking, healthcare, legal, government), this is a compliance risk. Winnex provides mathematical proof per excluded document — the only technology that does so.

**Pitch for investors**: See the [Open Letter to Investors](https://doi.org/10.5281/zenodo.21106472) (7 pages).

---

## Zenodo Records

| Record | DOI | Description |
|--------|-----|-------------|
| **Open Letter to Investors** | [10.5281/zenodo.21106472](https://doi.org/10.5281/zenodo.21106472) | Market thesis, technology, partnership invitation (7p) |
| **Audit Benchmark Supplement** | [10.5281/zenodo.21101148](https://doi.org/10.5281/zenodo.21101148) | EU AI Act, LGPD, HIPAA compliance mapping (9p) |
| **Definitive Benchmark** | [10.5281/zenodo.21088504](https://doi.org/10.5281/zenodo.21088504) | SIFT-1M, 16 methods, 12 metrics, 68KB results |
| **Kaggle Notebook** | [kaggle.com/...](https://www.kaggle.com/code/kleniopadilha/winnex-definitive-benchmark-v1-0) | Executable GPU P100 benchmark |
| **Madhava v12** | [10.5281/zenodo.21073400](https://doi.org/10.5281/zenodo.21073400) | Config-driven enterprise vector search |
| **Madhava v5** | [10.5281/zenodo.21066971](https://doi.org/10.5281/zenodo.21066971) | Numba-JIT, epsilon calibration, zero violations |
| **O(K) Navigation Proof** | [10.5281/zenodo.20856138](https://doi.org/10.5281/zenodo.20856138) | Spectral filter + QJL + anchor-based retrieval |
| **Research history (11 records)** | [Zenodo community](https://zenodo.org/communities/zenodo/search?q=winnex) | Full 18-month development arc |

---

## Architecture

```
vector_database/     ← your existing FAISS/Pinecone/Milvus
    └── search(query) → [indices, scores]

winnex_audit/        ← this module (C++20, Eigen3, OpenMP)
    ├── build(vectors)      ← QR-JL orthogonal projections + residual cache
    ├── search(query, k)    ← Cascaded bound refinement (stage1 -> stage2 -> exact)
    ├── check_bounds(query) ← Cauchy-Schwarz violation test (per-document proof)
    └── audit_json(query)   ← JSON audit trail for regulators
```

## Mathematical Foundation

For any orthogonal projection P: R^D -> R^k and vectors v (corpus), q (query):

```
|<v,q> - <Pv,Pq>| <= ||v - P^T P v|| * ||q - P^T P q||   (Cauchy-Schwarz)

<v,q> <= <Pv,Pq> + ||v - P^T P v|| * ||q - P^T P q||     (upper bound)
```

If the upper bound for a document falls below the top-K threshold, that document **mathematically cannot** be in the results. This is verified at build time via QR orthogonality:

```cpp
assert(||P * P^T - I|| < 1e-5);  // production code assert
```

---

## Quick Start

### Dependencies

- C++20 compiler (GCC 11+, Clang 14+)
- CMake 3.20+
- OpenMP

### Build

```bash
git clone https://github.com/winnex-ai/winnex-audit-cpp.git
cd winnex_audit_cpp
mkdir build && cd build
cmake .. -DWINNEX_BUILD_PYTHON=ON
make -j$(nproc)
```

### C++ Usage

```cpp
#include "winnex_audit/engine.h"
#include <iostream>

int main() {
    // Configuration
    winnex::Config cfg;
    cfg.input_dim = 128;
    cfg.stage1_dim = 64;
    cfg.stage2_dim = 128;
    cfg.final_k = 10;

    // Build engine
    winnex::AuditEngine engine(cfg);
    engine.build(vectors, N);  // 50K vectors in ~0.01s

    // Search with audit trail
    auto result = engine.search(query, 10);

    // Check bound violations
    auto [v64, v128] = engine.check_bounds(query);

    // JSON audit trail
    std::cout << engine.audit_json(query) << std::endl;
    // Outputs per-document proof for regulators
}
```

### Python Usage

```python
from winnex_audit_py import AuditEngine, Config
import numpy as np

# Your vectors
vectors = np.random.randn(50000, 128).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# Build audit engine
cfg = Config()
cfg.input_dim = 128
cfg.stage1_dim = 64
cfg.stage2_dim = 128

engine = AuditEngine(cfg)
engine.build(vectors)

# Search + audit
result = engine.search(vectors[0], k=10)
print(f"Results: {result.indices}")
print(f"Zero violations: {engine.check_bounds(vectors[0])}")
print(f"Audit JSON: {engine.audit_json(vectors[0])}")
```

---

## Integration with FAISS

```python
import faiss
from winnex_audit_py import AuditEngine, Config

# FAISS index (fast search)
dim = 128
index = faiss.IndexFlatIP(dim)
index.add(vectors)

# Winnex audit layer (mathematical proof)
audit = AuditEngine(Config())
audit.build(vectors)

# For each query: both results AND proof
query = vectors[0].reshape(1, -1).astype(np.float32)
faiss_results = index.search(query, 10)[1][0]

audit_result = audit.search(query[0], k=10)
viol_64d, viol_128d = audit.check_bounds(query[0])
print(f"Bound violations in top-10: {viol_64d}")

# Generate compliance report
print(audit.audit_json(query[0]))
```

## Integration with Pinecone

```python
import pinecone
from winnex_audit_py import AuditEngine

# Pinecone for storage + retrieval
pinecone.init(api_key="...")
vector_db = pinecone.Index("my-index")

# Winnex for audit trail (local vectors)
audit = AuditEngine()
audit.build(your_vectors)

# Retrieval + proof
db_results = vector_db.query(query, top_k=10)
proof = audit.audit_json(query)  # JSON for regulators

# Proof details for each excluded document:
# {
#   "doc_id": 42042,
#   "true_cosine": 0.2317,
#   "upper_bound": 0.2595,
#   "threshold": 0.4500,
#   "excluded": true,
#   "verdict": "PROVABLY OUTSIDE TOP-10"
# }
```

---

## Benchmark Results (SIFT-1M, 50K vectors)

| Method | NDCG@10 | Recall@10 | Latency | Build (50K) | Build (1M) | Bound Guarantee |
|--------|:-------:|:---------:|:-------:|:-----------:|:----------:|:---------------:|
| **MadhavaCore [64,128]** | **1.000** | **1.000** | 1.42ms* | **0.09s** | **2.57s** | **Zero violations** |
| MadhavaCore [32,64] | 1.000 | 1.000 | 1.03ms* | 0.06s | 1.69s | Zero violations |
| **HNSW(ef=128)** | 1.000 | 1.000 | **0.14ms** | 0.53s | ~40s | **None** |
| IVF(nprobe=20) | 0.987 | 0.980 | 0.05ms | <1m | <1m | None |
| PQ(m=16) | 0.456 | 0.401 | 0.17ms | 0.33s | — | None |
| FlatIP (exact) | 1.000 | 1.000 | 0.56ms | 0.01s | 0.13s | Exact |

*Python/NumPy latency. C++ version targets ~0.15ms (est. 9-10x speedup).  
Full results: [10.5281/zenodo.21088504](https://doi.org/10.5281/zenodo.21088504)

### Zero Violations Confirmed

| Method | Pairs Checked | Violations | Rate |
|--------|:------------:|:----------:|:----:|
| MadhavaCore [64,128] (64D) | 10,000,000 | **0** | 0.0000% |
| MadhavaCore [64,128] (128D) | 10,000,000 | **0** | 0.0000% |
| MadhavaCore [32,64] (32D) | 10,000,000 | **0** | 0.0000% |
| HNSW | 10,000,000 | N/A | Cannot measure |
| IVF | 10,000,000 | N/A | Cannot measure |

Only Madhava can MEASURE bound violations because only Madhava has a mathematical upper bound.

---

## Regulatory Compliance

| Regulation | Requirement | Madhava | HNSW/IVF/PQ |
|-----------|-------------|---------|-------------|
| **EU AI Act** (Art.13-15) | Transparency, oversight, record-keeping | Per-document bound proof | Black box |
| **LGPD** (Art.20) | Right to review automated decisions | Full audit trail | Cannot explain |
| **GDPR** (Art.22) | Meaningful info about automated logic | Mathematical signature | None |
| **HIPAA** | Reproducible search trails | Deterministic, auditable | Non-deterministic |

See the [Audit Benchmark Supplement](https://doi.org/10.5281/zenodo.21101148) (9 pages) for complete compliance mapping.

---

## Performance Targets (C++ vs Python)

| Operation | Python (NumPy) | C++ (Eigen+OpenMP) | Target Speedup |
|-----------|:--------------:|:------------------:|:--------------:|
| Build 50K | 0.09s | ~0.01s | **9x** |
| Build 1M | 2.57s | ~0.15s | **17x** |
| Search 50K | 1.42ms | ~0.15ms | **9.5x** |
| Check bounds 50K | 1.2ms | ~0.12ms | **10x** |
| Audit JSON gen | 0.5ms | ~0.05ms | **10x** |

*C++ estimates based on Eigen 3 + OpenMP parallelization. Benchmarks compiled with GCC 11, -O2, tested on Intel Xeon.*

---

## Project Structure

```
winnex_audit_cpp/
├── CMakeLists.txt                  # Build system (C++20, OpenMP)
├── LICENSE                         # Business Source License 1.1
├── README.md                       # This file
├── .gitignore
├── include/winnex_audit/
│   ├── engine.h                    # AuditEngine public API
│   ├── projection.h               # QR-JL + Pythagorean residuals
│   └── audit_trail.h              # JSON formatter + compliance checker
├── src/
│   ├── engine.cpp                  # Cascaded search + bound verification
│   └── projection.cpp             # QR decomposition + residual computation
├── bindings/
│   └── py_module.cpp              # pybind11 Python module
├── examples/
│   └── bench_audit.cpp            # Standalone C++ benchmark
└── third_party/
    └── Eigen/                     # Eigen 3.4 (header-only, no build needed)
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `WINNEX_BUILD_TESTS` | ON | Build unit tests |
| `WINNEX_BUILD_PYTHON` | ON | Build Python bindings (requires pybind11) |
| `WINNEX_BUILD_BENCHMARK` | ON | Build benchmark executable |

---

---



**Business Source License 1.1 (BSL 1.1)**

- Study, testing, and non-production evaluation: **permitted**
- Commercial deployment: **requires separate license agreement**
- Change Date: 2036-01-01 (after which the code converts to GPL v2.0+)

**Contact:** pay@winnex.ai

---

## Research

This module is based on 18 months of research published across 11 Zenodo records:

- [Winnex Definitive Benchmark](https://doi.org/10.5281/zenodo.21088504): 16 methods, SIFT-1M, 12 metrics
- [O(K) Navigation Proof](https://doi.org/10.5281/zenodo.20856138): Spectral filter + QJL + anchors
- [Madhava v12](https://doi.org/10.5281/zenodo.21073400): Config-driven enterprise search
- [Madhava v5](https://doi.org/10.5281/zenodo.21066971): Numba-JIT, zero violations
- [Lampreia Framework](https://doi.org/10.5281/zenodo.20835705): Quantum-inspired regularization

**Kaggle Benchmark:** https://www.kaggle.com/code/kleniopadilha/winnex-definitive-benchmark-v1-0

*Winnex AI — Enterprise infrastructure for mathematically transparent artificial intelligence.*  
*CNPJ: 58.364.637/0001-47 | Brazil | pay@winnex.ai*


---

## 🏗️ Project History & Transparency Note

### When This Was Built

The Winnex AI project was **started in December 2024** as a private research initiative by Klenio Araujo Padilha under Winnex Brasil Solucoes Empresariais LTDA - ME (CNPJ: 58.364.637/0001-47).

The core mathematical research (Riemannian HMC, Cauchy-Schwarz bounds, QR-JL projections) was developed privately over **18 months (Dec 2024 -- Jun 2026)** before any code was made public. The algorithm went through 12 major iterations (Madhava v1 through v12) as a private codebase.

### Why All Repositories Were Published on the Same Day

All public repositories were created on **July 1, 2026**. This is not because they were built in a day:

1. **The project was private for 18 months** during the research and development phase
2. **The repositories were opened simultaneously** to present the complete stack architecture to potential investors and partners

### Code Maturity by Layer

| Layer | Development Period | Public Since | Maturity |
|-------|-------------------|-------------|----------|
| **Layer 1: Audit Engine** | Dec 2024 -- Jun 2026 (18 mo) | Jul 1, 2026 | Research-grade, compiled, benchmarked |
| **Layer 2: Enterprise Stack** | Jun 2026 (blueprint) | Jul 1, 2026 | Product vision, not built yet |
| **Layer 3: Proof-of-Audit** | Jun 2026 (reference) | Jul 1, 2026 | Experimental, not production |
| **Production Tools** | Jun 2026 (blueprint) | Jul 1, 2026 | Blueprint, not built yet |

### Timeline

```
Dec 2024    Project started (private)
Jun 2025    Six bugs identified and corrected; Zenodo records begin
Jan 2026    Madhava v12; zero bound violations verified on SIFT-1M
Jun 30, 2026 All 11 Zenodo records published
Jul 1, 2026  GitHub repositories opened; Open Letter to Investors published
```

This repository is shared under **Business Source License 1.1 (BSL 1.1)**. The code was previously private and is now opened for transparency and study. Commercial deployment requires a separate license agreement.

**Contact:** pay@winnex.ai
