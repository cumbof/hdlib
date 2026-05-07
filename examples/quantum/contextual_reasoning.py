"""Quantum Contextual Analogy Solving.

Demonstrates :func:`quantum_contextual_bind` for solving multiple analogical
reasoning queries simultaneously via quantum superposition.

Classical problem
-----------------
Solve C analogy problems of the form "What is the [Currency] of [Country]?"
independently.  Each query requires a separate bind → search chain: C chains
in total.

Quantum approach
----------------
:func:`quantum_contextual_bind` creates a *single* entangled quantum state
that simultaneously encodes all C contextual bindings:

    |ψ_ctx⟩ = (1/√C) Σ_k |k⟩ ⊗ |bind(DOL, USTATES_k, MEXICO_k)⟩

Measuring the index register in basis |k⟩ projects the system onto the
specific analogy result for query k—a quantum key-value lookup over multiple
queries at once.

Analogy problems
----------------
1. USA / Dollar  → Mexico / ?    (answer: Peso)
2. Germany / Euro → France / ?   (answer: FrancF — simplified)
3. Japan / Yen   → Korea / ?     (answer: Won)

Quantum advantage summary
-------------------------
Classical: C separate bind+search chains.
Quantum:   1 contextual circuit encodes all C simultaneously.
The contextual state uses n + ⌈log₂C⌉ qubits vs C × D classical bits.

Usage
-----
    python contextual_reasoning.py
"""

import numpy as np
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hdlib.arithmetic.quantum import (
    encode,
    quantum_contextual_bind,
    quantum_inner_product,
    statevector_to_bipolar,
    bind as quantum_bind,
)
from hdlib.space import Space
from hdlib.vector import Vector
from hdlib.arithmetic import bind, bundle


# ── Configuration ────────────────────────────────────────────────────────────

SEED = 42
DIMENSION = 8       # Must be a power of 2 ≥ 4 for oracle encoding
EPSILON = 0.05


# ── Analogy problems ─────────────────────────────────────────────────────────
# Each entry: (source_country, source_currency, target_country, target_currency)
ANALOGIES = [
    ("USA",     "DOL", "MEX", "PES"),   # Dollar of Mexico
    ("DEU",     "EUR", "FRA", "FRF"),   # Euro of France (simplified)
    ("JPN",     "YEN", "KOR", "WON"),   # Yen of Korea
]

ALL_CONCEPTS = list({c for quad in ANALOGIES for c in quad})


def make_concepts(concepts, dim, seed):
    """Return a dict of random bipolar vectors for each concept."""
    rng = np.random.default_rng(seed)
    return {c: rng.choice([-1, 1], size=dim).astype(int) for c in concepts}


def classical_analogy(src_country, src_currency, tgt_country, concepts, codebook, dim):
    """Classical HDC analogy: bind(src_currency, bind(src_country, tgt_country))."""
    src_c_vec = Vector(size=dim, vector=concepts[src_country].copy(), vtype="bipolar")
    src_m_vec = Vector(size=dim, vector=concepts[src_currency].copy(), vtype="bipolar")
    tgt_c_vec = Vector(size=dim, vector=concepts[tgt_country].copy(), vtype="bipolar")

    # F = bind(USTATES, MEXICO) ≈ bind(src_country, tgt_country)
    f = bind(src_c_vec, tgt_c_vec)
    # guess = bind(src_currency, F)
    guess = bind(src_m_vec, f)

    best_name, best_dist = None, float("inf")
    for name, vec in codebook.items():
        v_obj = Vector(size=dim, vector=vec.copy(), vtype="bipolar")
        d = guess.dist(v_obj, method="cosine")
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def quantum_analogy_single(src_country, src_currency, tgt_country,
                            concepts, codebook, backend, dim):
    """Quantum HDC analogy via bind + IQAE nearest-neighbour search."""
    query_oracle = quantum_bind([
        encode(concepts[src_currency]),
        encode(concepts[src_country]),
        encode(concepts[tgt_country]),
    ])
    best_name, best_ip = None, -np.inf
    for name, vec in codebook.items():
        ip = quantum_inner_product(
            query_oracle, encode(vec), backend=backend, epsilon=EPSILON
        )
        if ip > best_ip:
            best_ip, best_name = ip, name
    return best_name


def quantum_contextual_demo(analogies, concepts, codebook, backend, dim):
    """Build a single contextual binding circuit for all C analogies."""
    print("  Building quantum_contextual_bind circuit for all analogies...")

    # Context oracle: shared "Dollar equivalent" role (src_currency for analogy 0)
    # We use the DOL oracle as the context and the src_country × tgt_country
    # bindings as the K values.
    context_oracle = encode(concepts["DOL"])

    # Values: bind(src_country, tgt_country) for each analogy
    value_oracles = []
    for src_c, src_m, tgt_c, tgt_m in analogies:
        val_oracle = quantum_bind([encode(concepts[src_c]), encode(concepts[tgt_c])])
        value_oracles.append(val_oracle)

    qc = quantum_contextual_bind(context_oracle, value_oracles)

    from math import ceil, log2
    n_idx = max(1, ceil(log2(len(analogies)))) if len(analogies) > 1 else 1
    n_sys = qc.num_qubits - n_idx
    step = 2 ** n_idx

    print(f"  Circuit qubits: {qc.num_qubits} (idx={n_idx} + sys={n_sys})")

    # For each analogy, post-select the system in the corresponding index branch
    sv = Statevector.from_instruction(qc.decompose().decompose())
    sv_data = np.asarray(sv.data)

    results = {}
    for k, (src_c, src_m, tgt_c, tgt_m) in enumerate(analogies):
        # Extract system amplitudes for index = k
        # idx=k: statevector indices where lowest n_idx bits == k
        sys_amps = sv_data[k::step]
        norm = np.linalg.norm(sys_amps)
        if norm < 1e-10:
            results[k] = None
            continue
        sys_amps = sys_amps / norm

        best_name, best_ip = None, -np.inf
        for name, vec in codebook.items():
            cand_oracle = encode(vec)
            from qiskit import QuantumCircuit
            qc_ref = QuantumCircuit(n_sys)
            qc_ref.h(range(n_sys))
            qc_ref.compose(cand_oracle, inplace=True)
            sv_ref = Statevector.from_instruction(qc_ref.decompose().decompose())
            # |<ref|sys>|^2 = fidelity
            ip = float(np.abs(np.dot(np.conj(sv_ref.data), sys_amps)) ** 2)
            if ip > best_ip:
                best_ip, best_name = ip, name
        results[k] = best_name

    return results


def main():
    print("=" * 64)
    print(" Quantum Contextual Analogy Solving")
    print("=" * 64)
    print(f"HD dimension : {DIMENSION}")
    print(f"Analogies    : {len(ANALOGIES)}")
    print()

    concepts = make_concepts(ALL_CONCEPTS, DIMENSION, SEED)
    backend = AerSimulator()

    # Codebook: only the target currencies (what we're looking for)
    currency_names = list({quad[3] for quad in ANALOGIES})
    codebook = {name: concepts[name] for name in currency_names}

    print("─── Classical HDC (individual queries) ───")
    c_results = []
    for src_c, src_m, tgt_c, tgt_m in ANALOGIES:
        ans = classical_analogy(src_c, src_m, tgt_c, concepts, codebook, DIMENSION)
        correct = (ans == tgt_m)
        c_results.append(correct)
        print(f"  '{src_m} of {tgt_c}'  → {ans}  (expected {tgt_m})  "
              f"{'✓' if correct else '✗'}")
    print(f"  Classical accuracy: {sum(c_results)}/{len(c_results)}")

    print()
    print("─── Quantum Individual Queries ───────────")
    q_results = []
    for src_c, src_m, tgt_c, tgt_m in ANALOGIES:
        ans = quantum_analogy_single(src_c, src_m, tgt_c, concepts, codebook, backend, DIMENSION)
        correct = (ans == tgt_m)
        q_results.append(correct)
        print(f"  '{src_m} of {tgt_c}'  → {ans}  (expected {tgt_m})  "
              f"{'✓' if correct else '✗'}")
    print(f"  Quantum (individual) accuracy: {sum(q_results)}/{len(q_results)}")

    print()
    print("─── Quantum Contextual (all at once) ────")
    ctx_results = quantum_contextual_demo(ANALOGIES, concepts, codebook, backend, DIMENSION)
    ctx_correct = []
    for k, (src_c, src_m, tgt_c, tgt_m) in enumerate(ANALOGIES):
        ans = ctx_results.get(k, "None")
        correct = (ans == tgt_m)
        ctx_correct.append(correct)
        print(f"  Index k={k}: '{src_m} of {tgt_c}'  → {ans}  "
              f"(expected {tgt_m})  {'✓' if correct else '✗'}")
    print(f"  Contextual accuracy: {sum(ctx_correct)}/{len(ctx_correct)}")

    print()
    print("Quantum advantage summary:")
    print("  quantum_contextual_bind encodes all C analogy states in a single")
    print("  entangled circuit using n + ⌈log₂C⌉ qubits.")
    print("  Classical: C separate bind+search chains.")
    print(f"  Here C={len(ANALOGIES)}, circuit qubits = {DIMENSION.bit_length()-1} + "
          f"{max(1, int(np.ceil(np.log2(len(ANALOGIES)))))} = "
          f"{DIMENSION.bit_length()-1 + max(1, int(np.ceil(np.log2(len(ANALOGIES)))))}.")


if __name__ == "__main__":
    main()
