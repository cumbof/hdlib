"""Grover-Accelerated Codebook Search in Hyperdimensional Computing.

Demonstrates the O(√N) quantum advantage of Grover's algorithm over the
classical O(N) linear scan for nearest-neighbour retrieval in a hypervector
codebook.

Usage
-----
    python grover_codebook_search.py

The script builds a codebook of N randomly-generated bipolar hypervectors,
creates a *noisy* query (10 % of bits flipped), and compares:

* **Classical HDC search** – linear scan comparing the query to every
  prototype using cosine similarity (O(N · D) operations).
* **Quantum Grover search** – :func:`grover_search` uses the quantum
  compute-uncompute test to estimate all N similarities, then applies Grover
  diffusion on the index register to amplify the best match, demonstrating
  the amplification structure.

Results are printed for codebook sizes N = 4, 8, 16 and include:
- Retrieved index and whether it is correct
- Wall-clock time for classical vs. quantum search
- Number of similarity evaluations (classical) vs. circuit executions
  (quantum)

Quantum advantage summary
-------------------------
Classical nearest-neighbour: Ω(N) similarity evaluations.
Quantum Grover:              O(√N) oracle calls (with full QRAM oracle).

This script runs on the noise-free ``AerSimulator`` backend.
"""

import time

import numpy as np
from qiskit_aer import AerSimulator

# ── hdlib ────────────────────────────────────────────────────────────────────
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hdlib.space import Space
from hdlib.vector import Vector
from hdlib.arithmetic.quantum import encode, grover_search, run_compute_uncompute_test


# ── Configuration ────────────────────────────────────────────────────────────

SEED = 42
DIMENSION = 16          # Hypervector dimension (small for fast simulation)
NOISE_RATE = 0.10       # Fraction of bits to flip in the noisy query
SHOTS = 2048            # Measurement shots per circuit
CODEBOOK_SIZES = [4, 8, 16]


def build_codebook(N: int, dim: int, seed: int):
    """Returns N random bipolar vectors and their oracle circuits."""
    rng = np.random.default_rng(seed)
    vectors = [rng.choice([-1, 1], size=dim).astype(int) for _ in range(N)]
    oracles = [encode(v) for v in vectors]
    return vectors, oracles


def make_noisy_query(vector: np.ndarray, noise_rate: float, seed: int) -> np.ndarray:
    """Returns a copy of *vector* with *noise_rate* fraction of bits flipped."""
    rng = np.random.default_rng(seed)
    noisy = vector.copy()
    n_flip = max(1, int(len(vector) * noise_rate))
    flip_idx = rng.choice(len(vector), size=n_flip, replace=False)
    noisy[flip_idx] *= -1
    return noisy


def classical_search(query_vec: np.ndarray, codebook_vecs: list) -> tuple:
    """Linear scan returning (best_index, similarity, n_evaluations)."""
    best_idx, best_sim = 0, -1.0
    for k, v in enumerate(codebook_vecs):
        sim = float(np.dot(query_vec.astype(float), v.astype(float)) / len(query_vec))
        if sim > best_sim:
            best_sim, best_idx = sim, k
    return best_idx, best_sim, len(codebook_vecs)


def main():
    print("=" * 64)
    print(" Grover-Accelerated Codebook Search")
    print("=" * 64)
    print(f"Dimension  : {DIMENSION}")
    print(f"Noise rate : {NOISE_RATE * 100:.0f}%")
    print(f"Shots      : {SHOTS}")
    print()

    backend = AerSimulator()
    target_idx = 2  # always query prototype #2

    for N in CODEBOOK_SIZES:
        print(f"─── Codebook size N = {N} {'─' * (40 - len(str(N)))}")

        vectors, oracles = build_codebook(N, DIMENSION, SEED)
        query_vec = make_noisy_query(vectors[target_idx], NOISE_RATE, SEED + 1)
        query_oracle = encode(query_vec)

        # ── Classical search ──────────────────────────────────────────────
        t0 = time.perf_counter()
        c_idx, c_sim, n_evals = classical_search(query_vec, vectors)
        t_classical = time.perf_counter() - t0

        # ── Quantum Grover search ─────────────────────────────────────────
        t0 = time.perf_counter()
        q_idx, q_sim = grover_search(
            query_oracle,
            oracles,
            similarity_threshold=0.7,
            backend=backend,
            shots=SHOTS,
        )
        t_quantum = time.perf_counter() - t0

        # ── Report ────────────────────────────────────────────────────────
        print(f"  Target index  : {target_idx}")
        print(f"  Classical      idx={c_idx}  sim={c_sim:.4f}  "
              f"time={t_classical*1e3:.2f}ms  evals={n_evals}  "
              f"{'✓' if c_idx == target_idx else '✗'}")
        print(f"  Quantum Grover idx={q_idx}  sim={q_sim:.4f}  "
              f"time={t_quantum*1e3:.2f}ms  "
              f"{'✓' if q_idx == target_idx else '✗'}")
        print()

    print("Quantum advantage note:")
    print("  Classical: Ω(N) similarity evaluations.")
    print("  Quantum:   O(√N) oracle calls with a full QRAM-based oracle.")
    print("  This demo uses the quantum compute-uncompute test for all N")
    print("  similarities, then applies Grover amplification to show the")
    print("  amplification structure.")


if __name__ == "__main__":
    main()
