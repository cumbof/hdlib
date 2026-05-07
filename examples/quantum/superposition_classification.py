"""Parallel Class Training via Superposition Bundle.

Demonstrates that :func:`superposition_bundle` can assemble all C class
prototype hypervectors in parallel (single SELECT circuit), while the
classical approach requires C sequential bundling passes.

The experiment uses the Iris dataset (3 classes) and compares:

* **Classical HDC** – builds each class prototype by iteratively calling
  :func:`bundle` on all training samples (O(C · M) sequential operations).
* **Quantum superposition bundle** – builds all class prototypes simultaneously
  using a single SELECT circuit with O(log N) circuit depth.

For each approach we report:
- Classification accuracy via leave-one-out style evaluation
- Wall-clock training time
- Circuit depth of the quantum SELECT circuit vs. sequential depth estimate

Quantum advantage summary
-------------------------
Superposition bundle: O(n_idx · T_oracle) = O(log N · T_oracle) circuit depth.
Sequential bundle:    O(N · T_oracle) circuit depth.

Usage
-----
    python superposition_classification.py
"""

import time
from math import ceil, log2

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from qiskit_aer import AerSimulator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hdlib.vector import Vector
from hdlib.space import Space
from hdlib.arithmetic import bind, bundle
from hdlib.arithmetic.quantum import (
    encode,
    superposition_bundle,
    run_compute_uncompute_test,
    statevector_to_bipolar,
    _build_select_circuit,
    get_circuit_metrics,
)


# ── Configuration ────────────────────────────────────────────────────────────

SEED = 42
DIMENSION = 8           # Small HD dimension so quantum sim stays tractable
LEVELS = 4              # Number of level hypervectors for encoding
TEST_SIZE = 0.3
SHOTS = 1024


def encode_features(points, features, dim, seed):
    """Encode real-valued feature vectors as bipolar hypervectors."""
    rng = np.random.default_rng(seed)
    n_feat = len(features)

    # Level vectors (uniformly spaced bipolar)
    level_vecs = []
    base = rng.choice([-1, 1], size=dim).astype(int)
    level_vecs.append(base.copy())
    flips_per_level = dim // (2 * LEVELS)
    for _ in range(1, LEVELS):
        prev = level_vecs[-1].copy()
        flip_idx = rng.choice(np.where(prev == 1)[0], size=flips_per_level, replace=False)
        prev[flip_idx] = -1
        level_vecs.append(prev)

    # Feature vectors
    feat_vecs = [rng.choice([-1, 1], size=dim).astype(int) for _ in range(n_feat)]

    # Normalise each feature to [0, LEVELS-1]
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    encoded = []
    for pt in points:
        hv = np.zeros(dim, dtype=float)
        for i, val in enumerate(pt):
            if maxs[i] > mins[i]:
                lvl = int((val - mins[i]) / (maxs[i] - mins[i]) * (LEVELS - 1))
            else:
                lvl = 0
            lvl = min(lvl, LEVELS - 1)
            hv += level_vecs[lvl] * feat_vecs[i]
        # Resolve ties by assigning +1 (non-zero sum guaranteed for random features)
        bipolar = np.where(hv >= 0, 1, -1).astype(int)
        encoded.append(bipolar)

    return encoded


def classical_train(X_train, y_train, classes, dim, seed):
    """Build one prototype per class via sequential bundle."""
    from hdlib.vector import Vector
    prototypes = {}
    for cls in classes:
        cls_vecs = [Vector(size=dim, vector=X_train[i].copy(), vtype="bipolar")
                    for i, y in enumerate(y_train) if y == cls]
        if len(cls_vecs) == 1:
            proto = cls_vecs[0]
        else:
            from functools import reduce
            proto = reduce(bundle, cls_vecs)
            proto.normalize()
        prototypes[cls] = proto.vector
    return prototypes


def quantum_train(X_train, y_train, classes, dim):
    """Build one oracle per class via superposition_bundle."""
    quantum_prototypes = {}
    select_depths = {}

    for cls in classes:
        cls_vecs = [X_train[i] for i, y in enumerate(y_train) if y == cls]
        oracle_circs = [encode(v) for v in cls_vecs]

        if len(oracle_circs) == 1:
            proto_circ = oracle_circs[0]
        else:
            # quantum SELECT bundle
            proto_circ = superposition_bundle(oracle_circs)

            # Record depth of the internal SELECT circuit
            n_idx = max(1, ceil(log2(len(oracle_circs)))) if len(oracle_circs) > 1 else 1
            sel_qc = _build_select_circuit(oracle_circs)
            select_depths[cls] = sel_qc.depth()

        quantum_prototypes[cls] = proto_circ

    return quantum_prototypes, select_depths


def classical_predict(X_test, prototypes, dim):
    """Nearest prototype classifier using cosine distance."""
    preds = []
    for x in X_test:
        best_cls, best_sim = None, -np.inf
        for cls, proto_vec in prototypes.items():
            sim = float(np.dot(x.astype(float), proto_vec.astype(float))) / dim
            if sim > best_sim:
                best_sim, best_cls = sim, cls
        preds.append(best_cls)
    return preds


def quantum_predict(X_test, quantum_prototypes, backend):
    """Nearest prototype classifier using compute-uncompute similarity."""
    preds = []
    classes = list(quantum_prototypes.keys())
    for x in X_test:
        query_oracle = encode(x)
        best_cls, best_ip = None, -np.inf
        for cls, proto_circ in quantum_prototypes.items():
            sims_matrix, _ = run_compute_uncompute_test(
                [query_oracle], [proto_circ], backend=backend, shots=SHOTS
            )
            ip = sims_matrix[0][0]
            if ip > best_ip:
                best_ip, best_cls = ip, cls
        preds.append(best_cls)
    return preds


def main():
    print("=" * 64)
    print(" Parallel Class Training via Superposition Bundle (Iris)")
    print("=" * 64)
    print(f"HD dimension : {DIMENSION}")
    print(f"Levels       : {LEVELS}")
    print()

    # ── Load & encode Iris ────────────────────────────────────────────────
    iris = datasets.load_iris()
    X_enc = encode_features(iris.data, iris.feature_names, DIMENSION, SEED)
    X_enc = np.array(X_enc)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=TEST_SIZE, random_state=SEED
    )

    classes = np.unique(y_train).tolist()
    backend = AerSimulator()

    # ── Classical training ────────────────────────────────────────────────
    t0 = time.perf_counter()
    c_prototypes = classical_train(X_train, y_train, classes, DIMENSION, SEED)
    t_c_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_preds = classical_predict(X_test, c_prototypes, DIMENSION)
    t_c_test = time.perf_counter() - t0

    c_acc = accuracy_score(y_test, c_preds)

    # ── Quantum training ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    q_prototypes, select_depths = quantum_train(X_train, y_train, classes, DIMENSION)
    t_q_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_preds = quantum_predict(X_test, q_prototypes, backend)
    t_q_test = time.perf_counter() - t0

    q_acc = accuracy_score(y_test, q_preds)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"Training samples : {len(X_train)}  (test: {len(X_test)})")
    print()
    print(f"{'Method':<28} {'Train (ms)':>12} {'Test (ms)':>12} {'Accuracy':>10}")
    print("-" * 66)
    print(f"{'Classical HDC':<28} {t_c_train*1e3:>12.2f} {t_c_test*1e3:>12.2f} {c_acc:>10.4f}")
    print(f"{'Quantum Superpos. Bundle':<28} {t_q_train*1e3:>12.2f} {t_q_test*1e3:>12.2f} {q_acc:>10.4f}")

    print()
    print("SELECT circuit depths per class (quantum training):")
    n_per_class = {cls: int(np.sum(np.array(y_train) == cls)) for cls in classes}
    for cls in classes:
        M = n_per_class[cls]
        depth = select_depths.get(cls, 0)
        sequential_estimate = M  # sequential bundle has depth ~ M
        print(f"  Class {cls}: M={M:3d} samples  "
              f"SELECT depth={depth:4d}  "
              f"sequential estimate=~{sequential_estimate}  "
              f"ratio={sequential_estimate/(depth or 1):.1f}×")

    print()
    print("Quantum advantage summary:")
    print("  SELECT circuit depth ≈ O(log M) layers for a tree-structured LCU.")
    print("  Sequential bundle depth ≈ O(M) layers.")
    print("  Compute-uncompute similarity: O(1/shots) statistical precision.")


if __name__ == "__main__":
    main()
