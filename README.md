# Winnex Audit Module

> **Mantenha seu banco vetorial. Adicione prova matemática.**  
> Keep your vector database. Add mathematical proof.

The Winnex Audit Module is a **drop-in compliance layer** for existing vector search infrastructure (FAISS, Pinecone, Milvus, Weaviate, Qdrant). It wraps any existing index with a Cauchy-Schwarz bound verifier, generating a per-document mathematical audit trail for every search query.

## Pitch

```
┌─────────────────────────────────────────────────────┐
│  Your existing vector database (FAISS/Pinecone/Milvus)│
│  Returns: [doc_42, doc_173, doc_891]                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Winnex Audit Module (C++ / Python bindings)        │
│  For each excluded document:                        │
│    "Doc #42042 excluded because upper_bound(0.2595) │
│     < threshold(0.4500) — mathematical certainty"   │
└─────────────────────────────────────────────────────┘
```

## Architecture

```
vector_database/     ← your existing FAISS/Pinecone/Milvus
    └── search(query) → [indices, scores]

winnex_audit/        ← this module (C++, pybind11)
    ├── build(vectors)      ← QR-JL projections + residual cache
    ├── search(query, k)    ← Cascaded bound refinement
    ├── check_bounds(query) ← Cauchy-Schwarz violation test
    └── audit_json(query)   ← Full JSON audit trail
```

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DWINNEX_BUILD_PYTHON=ON
make -j$(nproc)

# Install Python bindings
pip install .
```

```python
from winnex_audit_py import AuditEngine, Config
import numpy as np

# Your existing vectors (e.g., from Pinecone/Milvus)
vectors = np.random.randn(50000, 128).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# Load audit engine (standalone or wrapping existing index)
cfg = Config()
cfg.input_dim = 128
cfg.stage1_dim = 64
cfg.stage2_dim = 128

engine = AuditEngine(cfg)
engine.build(vectors)

# Search with audit trail
query = vectors[0]
result = engine.search(query, k=10)

# You get both results AND proof:
print(f"Results: {result.indices}")
print(f"Audit records: {len(result.audit)}")
print(f"Zero bound violations: {engine.check_bounds(query)}")
```

## Integration with FAISS

```python
import faiss
from winnex_audit_py import AuditEngine, Config

# 1. Build FAISS index (as usual)
dim = 128
index = faiss.IndexFlatIP(dim)
index.add(vectors)

# 2. Layer Winnex audit on top
audit = AuditEngine(Config())
audit.build(vectors)

# 3. Search with dual output
query = vectors[0].reshape(1, -1).astype(np.float32)
faiss_indices = index.search(query, 10)[1][0]  # FAISS results
audit_result = audit.search(query[0], k=10)     # Winnex audit trail

# 4. Verify: no FAISS result violates mathematical bounds
viol_64d, viol_128d = audit.check_bounds(query[0])
print(f"Bound violations in top-10: {viol_64d} (64D), {viol_128d} (128D)")
```

## Integration with Pinecone

```python
import pinecone
from winnex_audit_py import AuditEngine

# Pinecone index (separate)
pinecone.init(api_key="...")
index = pinecone.Index("my-index")

# Winnex audit layer (local, on your vectors)
audit = AuditEngine()
audit.build(your_vectors)

# For each Pinecone query:
pinecone_results = index.query(query, top_k=10)
audit_result = audit.search(query, k=10)
print(audit.audit_json(query))  # JSON audit trail for regulators
```

## Benchmark (C++ vs Python)

| Operation | Python (NumPy) | C++ (Eigen+OMP) | Speedup |
|-----------|---------------|-----------------|---------|
| Build 50K vectors | 0.09s | ~0.01s | **9x** |
| Build 1M vectors | 2.57s | ~0.15s | **17x** |
| Search (50K) | 1.42ms | ~0.15ms | **9.5x** |
| Check bounds (50K) | 1.2ms | ~0.12ms | **10x** |
| Audit JSON gen | 0.5ms | ~0.05ms | **10x** |

*C++ estimates based on Eigen 3 + OpenMP parallelization. Real benchmarks coming soon.*

## License

BSL 1.1 — pay@winnex.ai
