# Performance

`benchmarks/benchmark_local.py` measures the complete offline path with a deterministic generated Python repository. The default fixture contains 250 modules, each with one class and one method. It records cold indexing, unchanged incremental indexing, and 100 exact SQLite retrievals over the resulting chunks.

Run it from an installed development checkout:

```bash
python benchmarks/benchmark_local.py
python benchmarks/benchmark_local.py --files 500 --queries 200 --output /tmp/result.json
```

The committed `benchmarks/baseline.json` records one reference run. It is evidence of scale and a regression reference, not a performance guarantee: hardware, Python, filesystem caching, and background load materially affect timings. Remote provider latency and rate limits are excluded because the benchmark uses deterministic fake embeddings.

The reference run on 2026-10-18 used Python 3.14.8 under Linux/WSL2 on x86-64:

| Fixture or measurement | Result |
| --- | ---: |
| Python files | 250 |
| Stored chunks | 500 |
| Cold index | 0.108 s |
| Unchanged index | 0.060 s |
| Search mean, 100 queries | 0.026 s |
| Search p95, 100 queries | 0.032 s |

Exact values and environment metadata remain in the machine-readable baseline.

The benchmark also verifies that an unchanged reindex reports every fixture file as unchanged and makes no embedding batch calls. Search currently loads all vectors for one repository and performs exact cosine similarity in Python, so query cost grows linearly with stored chunks. Use this adapter for small local repositories and rerun the benchmark with a representative size before adopting it for a larger workload.
