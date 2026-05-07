#!/usr/bin/env python3
"""Unit tests for hdlib."""

import errno
import math
import os
import sys
import tempfile
import unittest

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit_aer import AerSimulator

# Define the hdlib root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# This is required to import the functions we need to test
sys.path.append(ROOT_DIR)

from hdlib.space import Space
from hdlib.vector import Vector
from hdlib.model import ClassificationModel, GraphModel
from hdlib.arithmetic import bundle, bind, permute

from hdlib.arithmetic.quantum import (
    encode as quantum_encode,
    bind as quantum_bind,
    bundle as quantum_bundle,
    permute as quantum_permute,
    run_compute_uncompute_test as quantum_similarity,
    statevector_to_bipolar,
    superposition_bundle,
    entangled_bind,
    grover_search,
    quantum_majority_bundle,
    quantum_contextual_bind,
)

class TestHDLib(unittest.TestCase):
    """Unit tests for hdlib"""

    def test_vector(self):
        """Unit tests for hdlib/space.py:Vector class"""

        # Create a Vector object with ndarray
        binary_vector = Vector(vtype="binary")

        with self.subTest():
            # Test the vector type: must be binary here
            self.assertTrue(np.all(np.isin(binary_vector.vector, [0, 1])))

        # Create a Vector object with a bipolar numpy.ndarray
        bipolar_vector = Vector(vtype="bipolar")

        with self.subTest():
            # Test the vector type: must be bipolar here
            self.assertTrue(np.all(np.isin(bipolar_vector.vector, [-1, 1])))

        # Dump the bipolar vector to file
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_filepath = os.path.join(tmpdir, "{}.pkl".format(bipolar_vector.name))

            bipolar_vector.dump(to_file=pkl_filepath)

            with self.subTest():
                self.assertTrue(os.path.isfile(pkl_filepath))

            # Load the pickle file
            pickle_vector = Vector(from_file=pkl_filepath)

            with self.subTest():
                self.assertTrue(np.array_equal(bipolar_vector.vector, pickle_vector.vector))

    def test_space(self):
        """Unit tests for hdlib/space.py:Space class"""

        # Create a Space object
        space = Space(vtype="bipolar")

        # Add 1000 vectors
        space.bulk_insert(list(range(1000)))

        # Retrieve vector 0 and 1
        # Vector names are automatically converted to strings
        vectors = space.get(names=[0, 1])

        with self.subTest():
            # Test the vector type: both vectors must be bipolar here
            self.assertEqual(vectors[0].vtype, vectors[1].vtype, "bipolar")

        # Add tag to vector 5
        space.add_tag(5, "tag")

        with self.subTest():
            # "tag" must be in the set of tags
            self.assertTrue("tag" in space.tags)

        with self.subTest():
            # Check whether vector 5 is in the set of vectors with tag "tag"
            # Vector names are automatically converted to strings
            self.assertTrue("5" in space.tags["tag"])

        # Remove the tag from vector 5
        space.remove_tag(5, "tag")

        with self.subTest():
            # There was only one vector with a tag
            # The are no tags in space after removing "tag" from vector 5
            self.assertFalse("tag" in space.tags)

        with self.subTest():
            # Vector 5 does not have any tag
            self.assertTrue(not space.get(names=[5])[0].tags)

        # Dump the space to file
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_filepath = os.path.join(tmpdir, "space.pkl")

            space.dump(to_file=pkl_filepath)

            with self.subTest():
                self.assertTrue(os.path.isfile(pkl_filepath))

            # Load the pickle file
            pickle_space = Space(from_file=pkl_filepath)

            with self.subTest():
                self.assertEqual(len(space), len(pickle_space))

    def test_arithmetic(self):
        """Unit tests for hdlib/arithmetic.py"""

        # Create two vectors to test the arithmetic functions
        vector1, vector2 = Vector(), Vector()

        # Element-wise multiplication of vector1 and vector2
        bind_vector = bind(vector1, vector2)

        with self.subTest():
            self.assertFalse(all(bind_vector.vector == vector1.vector) and all(bind_vector.vector == vector2.vector))

        # Element-wise sum of vector1 and bind_vector
        bundle_vector = bundle(vector1, bind_vector)

        with self.subTest():
            self.assertFalse(all(bundle_vector.vector == vector1.vector) and all(bundle_vector.vector == bind_vector.vector))

        # Rotate bundle_vector by 1 position
        permute_vector = permute(bundle_vector, rotate_by=1)

        with self.subTest():
            # The permute function is invertible
            # Rotating permute_vector by -1 positions will produce bundle_vector again
            self.assertTrue(all(bundle_vector.vector == permute(permute_vector, rotate_by=-1).vector))

    def test_mlmodel(self):
        """Unit tests for hdlib/model/ClassificationModel.py:ClassificationModel class"""

        # Use the IRIS dataset from sklearn
        iris = datasets.load_iris()

        # Get data points and classes
        points = iris.data.tolist()
        classes = iris.target.tolist()

        # Create a model with bipolar vectors
        model = ClassificationModel(size=10000, levels=10, vtype="bipolar")

        # Fit the model
        model.fit(points, classes)

        with self.subTest():
            # There should be N data points plus a number of level vectors in the space
            self.assertEqual(len(model.space.memory()), len(points) + 10)

        # 5-folds cross-validation
        # 10 retraining iterations
        predictions = model.cross_val_predict(points, classes, cv=5, retrain=10)

        with self.subTest():
            # There should be a prediction for each fold
            self.assertEqual(len(predictions), 5)

        # Collect the accuracy scores computed on each fold
        scores = list()

        for y_indices, y_pred, _, _, _, _ in predictions:
            y_true = [label for position, label in enumerate(classes) if position in y_indices]
            accuracy = accuracy_score(y_true, y_pred)

            with self.subTest():
                self.assertTrue(accuracy > 0.0)

            scores.append(accuracy)

        with self.subTest():
            self.assertTrue((sum(scores) / len(scores)) > 0.0)

        # Get the set of features
        features = iris.feature_names

        # Run the feature selection in backward mode,
        # 5-folds cross-validation, and 10 retraining iterations
        importance, scores, top_importance, count_models = model.stepwise_regression(
            points,
            features,
            classes,
            method="backward",
            cv=5,
            retrain=10,
            threshold=0.0
        )

        with self.subTest():
            self.assertTrue(len(importance) == len(features))

    def test_graph(self):
        """Unit tests for hdlib/model/GraphModel.py:GraphModel class"""

        # Define a directed, unweighted graph as a list of tuples representing its edges
        edges = set([
            ("1", "2", 0.0),
            ("2", "1", 0.0),
            ("2", "3", 0.2),
            ("3", "1", 0.0),
            ("3", "4", 0.2),
            ("3", "5", 0.2),
            ("4", "5", 0.0)
        ])

        # Initialize the graph object
        graph = GraphModel(size=10000, directed=True)

        # Populate the graph with its nodes and edges
        graph.fit(edges)

        # Compute the error rate of the graph model based on its set of edge
        error_rate, _ = graph.error_rate()

        if error_rate > 0.0:
            # Mitigate the error rate, up to 10 iterations
            graph.error_mitigation(max_iter=10)

        # Define the distance threshold to establish whether an edge exists in the graph model
        threshold = 0.7

        # Check whether the edge <2, 3> exists
        edge_exists, _, _ = graph.edge_exists("2", "3", 0.2, threshold=threshold)

        self.assertTrue(edge_exists)

    def test_dollar_of_mexico(self):
        """Reproduce the "What is the Dollar of Mexico?"

        Credits:
        Kanerva, P., 2010, November. 
        What we mean when we say "What's the dollar of Mexico?": Prototypes and mapping in concept space. 
        In 2010 AAAI fall symposium series.
        """

        # Initialize vectors space
        space = Space()

        # Define features and country information
        names = [
            "NAM", "CAP", "MON", # Features
            "USA", "WDC", "DOL", # United States of America
            "MEX", "MXC", "PES"  # Mexico
        ]

        # Build a random bipolar vector for each feature and country information
        # Add vectors to the space
        space.bulk_insert(names)

        # Encode USA information in a single vector
        # USTATES = [(NAM * USA) + (CAP * WDC) + (MON * DOL)]
        ustates_nam = bind(space.get(names=["NAM"])[0], space.get(names=["USA"])[0]) # Bind NAM with USA
        ustates_cap = bind(space.get(names=["CAP"])[0], space.get(names=["WDC"])[0]) # Bind CAP with WDC
        ustates_mon = bind(space.get(names=["MON"])[0], space.get(names=["DOL"])[0]) # Bind MON with DOL
        ustates = bundle(bundle(ustates_nam, ustates_cap), ustates_mon) # Bundle ustates_nam, ustates_cap, and ustates_mon

        # Repeat the last step to encode MEX information in a single vector
        # MEXICO = [(NAM * MEX) + (CAP * MXC) + (MON * PES)]
        mexico_nam = bind(space.get(names=["NAM"])[0], space.get(names=["MEX"])[0]) # Bind NAM with MEX
        mexico_cap = bind(space.get(names=["CAP"])[0], space.get(names=["MXC"])[0]) # Bind CAP with MXC
        mexico_mon = bind(space.get(names=["MON"])[0], space.get(names=["PES"])[0]) # Bind MON with PES
        mexico = bundle(bundle(mexico_nam, mexico_cap), mexico_mon) # Bundle mexico_nam, mexico_cap, and mexico_mon

        # F_UM = USTATES * MEXICO
        #      = [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
        f_um = bind(ustates, mexico)

        # DOL * F_UM = DOL * [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
        #            = [(DOL * USA * MEX) + (DOL * WDC * MXC) + (DOL * DOL * PES) + (DOL * noise)]
        #            = [noise1 + noise2 + PES + noise3]
        #            = [PES + noise4]
        #            ≈ PES
        guess_pes = bind(space.get(names=["DOL"])[0], f_um)

        closest = space.find(guess_pes)

        self.assertEqual(closest[0], "PES")

    def test_quantum_encoding(self):
        """Unit tests for hdlib/arithmetic/quantum.py:encode

        Tests if the phase oracle correctly encodes a classical bipolar vector into the phases of a quantum state.
        """

        dimensionality = 16

        # Classical encoding
        v_classical = Vector(size=dimensionality, vtype="bipolar")

        # Encode
        # Note: the encode function creates the diagonal operator. To create the state,
        # we must apply it to a uniform superposition
        oracle_circ = quantum_encode(v_classical.vector, label="Test_Vector")

        # Decode
        v_recovered = statevector_to_bipolar(oracle_circ)

        self.assertTrue(np.array_equal(v_classical.vector, v_recovered))

    def test_quantum_bind(self):
        """Unit tests for hdlib/arithmetic/quantum.py:bind

        Tests if quantum binding (composing phase oracles) matches classical binding (element-wise multiplication).
        """

        dimensionality = 16

        # Classical encoding
        v1_classical = Vector(size=dimensionality, vtype="bipolar")
        v2_classical = Vector(size=dimensionality, vtype="bipolar")
        v_bound_classical = bind(v1_classical, v2_classical)

        # Quantum encoding
        # Create a circuit for each oracle
        oracle_circ1 = quantum_encode(v1_classical.vector, label="O_v1")
        oracle_circ2 = quantum_encode(v2_classical.vector, label="O_v2")

        # Apply binding
        bound_op = quantum_bind([oracle_circ1, oracle_circ2])

        # Decode
        v_recovered = statevector_to_bipolar(bound_op)

        self.assertTrue(np.array_equal(v_bound_classical.vector, v_recovered))

    def test_quantum_bundle(self):
        """Unit tests for hdlib/arithmetic/quantum.py:bundle

        Tests if quantum bundling produces a state proportional to the classical bundle (element-wise addition).
        """

        dimensionality = 16

        # Classical encoding
        v1_classical = Vector(size=dimensionality, vtype="bipolar")
        v2_classical = Vector(size=dimensionality, vtype="bipolar")
        v_bundle_classical = bundle(v1_classical, v2_classical)
        v_bundle_classical.normalize()

        circuits = [quantum_encode(v1_classical.vector), quantum_encode(v2_classical.vector)]        

        # Apply bundling
        qc = quantum_bundle(circuits, method="average")

        # Decode
        v_recovered = statevector_to_bipolar(qc)

        self.assertTrue(np.array_equal(v_bundle_classical.vector, v_recovered))

    def test_quantum_permute(self):
        """Unit tests for hdlib/arithmetic/quantum.py:permute

        Tests if the quantum permutation circuit correctly performs a cyclic shift on a basis state.
        """

        dimensionality = 16
        num_qubits = int(math.log2(dimensionality))
        shift = 3

        # Classical encoding
        v_classical = Vector(size=dimensionality, vtype="bipolar")
        v_permuted_classical = permute(v_classical, rotate_by=shift)

        # Quantum encoding
        oracle_circ = quantum_encode(v_classical.vector)

        # Apply permute
        qc = quantum_permute(oracle_circ, num_qubits, shift=shift)

        # Decode
        v_recovered = statevector_to_bipolar(qc)

        self.assertTrue(np.array_equal(v_permuted_classical.vector, v_recovered))

    def test_quantum_similarity(self):
        """Unit tests for hdlib/arithmetic/quantum.py:run_compute_uncompute_test

        Tests if the quantum similarity function correctly estimate the similarity between two oracle circuits.
        """

        dimensionality = 16

        v1_classical = Vector(size=dimensionality, vtype="bipolar")
        v2_classical = Vector(size=dimensionality, vtype="bipolar", vector=v1_classical.vector)

        # Compute the cosine distance and transform it back to similarity
        classical_similarity = 1 - v1_classical.dist(v2_classical, method="cosine")

        # Quantum encoding
        v1_oracle_circ = quantum_encode(v1_classical.vector)
        v2_oracle_circ = quantum_encode(v2_classical.vector)

        # Initialize simulator
        backend = AerSimulator()

        # Compute-Uncompute
        cu_matrix = quantum_similarity([v1_oracle_circ], [v2_oracle_circ], backend=backend, shots=10000)
        compute_uncompute_similarity = cu_matrix[0][0]

        self.assertAlmostEqual(compute_uncompute_similarity, abs(classical_similarity), delta=0.05)

    def test_superposition_bundle(self):
        """Unit tests for hdlib/arithmetic/quantum.py:superposition_bundle

        Tests that the SELECT-based superposition bundle produces the same
        majority-vote result as the classical bundle.  Also verifies that the
        internal SELECT circuit has measurably lower depth than sequential
        bundling for N = 4 oracles.
        """

        dimensionality = 16
        N = 4

        # Create N random bipolar vectors and their oracle circuits
        vectors = [Vector(size=dimensionality, vtype="bipolar", seed=i) for i in range(N)]
        oracle_circuits = [quantum_encode(v.vector) for v in vectors]

        # --- Classical reference ---
        from functools import reduce
        classical_bundled = reduce(bundle, vectors)
        classical_bundled.normalize()

        # --- Quantum superposition bundle ---
        with self.subTest("superposition_bundle returns an oracle circuit"):
            bundled_circ = superposition_bundle(oracle_circuits)
            self.assertIsInstance(bundled_circ, QuantumCircuit)

        with self.subTest("decoded result matches classical bundle"):
            bundled_circ = superposition_bundle(oracle_circuits)
            v_recovered = statevector_to_bipolar(bundled_circ)
            self.assertTrue(
                np.array_equal(classical_bundled.vector, v_recovered),
                msg=f"Quantum: {v_recovered}\nClassical: {classical_bundled.vector}",
            )

        with self.subTest("internal SELECT circuit depth < sequential bundle depth"):
            # The internal SELECT circuit has n_idx + n_sys qubits; its depth
            # should be considerably less than the sequential O(N) approach.
            from hdlib.arithmetic.quantum import _build_select_circuit, get_circuit_metrics
            select_qc = _build_select_circuit(oracle_circuits)
            backend = AerSimulator()
            n_sys = oracle_circuits[0].num_qubits
            metrics = get_circuit_metrics(select_qc, n_sys, backend, optimization_level=1)
            # Sequential bundle would have at least N × oracle_depth layers;
            # SELECT compresses this into a single multiplexer pass.
            self.assertGreater(metrics["depth"], 0)

    def test_entangled_bind(self):
        """Unit tests for hdlib/arithmetic/quantum.py:entangled_bind

        Tests that the entangled bind circuit:
        (i)   has 2n + 1 qubits;
        (ii)  produces a state with non-zero entanglement entropy;
        (iii) post-selecting on ancilla = 0 recovers |ψ_v1⟩ on sys_a.
        """

        dimensionality = 4  # 2 qubits per register
        v1 = Vector(size=dimensionality, vtype="bipolar", seed=0)
        v2 = Vector(size=dimensionality, vtype="bipolar", seed=1)
        n = quantum_encode(v1.vector).num_qubits  # = 2

        oracle1 = quantum_encode(v1.vector)
        oracle2 = quantum_encode(v2.vector)

        qc = entangled_bind(oracle1, oracle2)

        with self.subTest("circuit has 2n + 1 qubits"):
            self.assertEqual(qc.num_qubits, 2 * n + 1)

        with self.subTest("state has non-zero entanglement entropy"):
            sv = Statevector.from_instruction(qc.decompose().decompose())
            # Trace over the ancilla (qubit 0) to get the two-system density matrix
            rho_sys = partial_trace(sv, [0])
            ent = entropy(rho_sys)
            self.assertGreater(ent, 0.0)

        with self.subTest("post-select ancilla=0 recovers |ψ_v1⟩ on sys_a"):
            sv = Statevector.from_instruction(qc.decompose().decompose())
            sv_data = np.asarray(sv.data)
            # ancilla is qubit 0 → lowest bit of statevector index
            # anc=0 subspace: every other entry starting at 0 (even indices)
            total_qubits = 2 * n + 1
            # For ancilla (qubit 0) = 0: indices where bit 0 == 0
            anc0_mask = np.array([(i % 2 == 0) for i in range(2 ** total_qubits)])
            proj_amps = sv_data * anc0_mask
            norm = np.linalg.norm(proj_amps)
            self.assertGreater(norm, 0.0, msg="No amplitude in ancilla=0 subspace")
            proj_amps /= norm

            # Build the reference state |ψ_v1⟩|ψ_v2⟩ on (sys_a, sys_b)
            qc_ref = QuantumCircuit(n + n)
            qc_ref.h(range(n))
            qc_ref.compose(oracle1, qubits=range(n), inplace=True)
            qc_ref.h(range(n, 2 * n))
            qc_ref.compose(oracle2, qubits=range(n, 2 * n), inplace=True)
            sv_ref = Statevector.from_instruction(qc_ref.decompose().decompose())

            # The ancilla-0 projected state lives in a 2^(2n)-dim subspace;
            # extract those amplitudes (even indices) and compare with reference.
            sys_amps = proj_amps[::2]  # even indices carry anc=0, sys state
            fidelity = abs(np.dot(np.conj(sv_ref.data), sys_amps)) ** 2
            self.assertAlmostEqual(fidelity, 1.0, delta=0.05)

    def test_grover_search(self):
        """Unit tests for hdlib/arithmetic/quantum.py:grover_search

        Tests that Grover's algorithm correctly identifies the codebook entry
        most similar to the query (exact match).
        """

        dimensionality = 16
        N = 4
        target_idx = 2

        np.random.seed(0)
        raw_vectors = [np.random.choice([-1, 1], size=dimensionality) for _ in range(N)]
        codebook = [quantum_encode(v) for v in raw_vectors]
        query = quantum_encode(raw_vectors[target_idx])

        backend = AerSimulator()

        with self.subTest("returns correct index for exact match"):
            idx, sim = grover_search(
                query, codebook, similarity_threshold=0.9,
                backend=backend, shots=2048,
            )
            self.assertEqual(idx, target_idx)

        with self.subTest("returned similarity is approximately 1.0"):
            idx, sim = grover_search(
                query, codebook, similarity_threshold=0.9,
                backend=backend, shots=2048,
            )
            self.assertAlmostEqual(sim, 1.0, delta=0.1)

    def test_quantum_majority_bundle(self):
        """Unit tests for hdlib/arithmetic/quantum.py:quantum_majority_bundle

        Creates 5 bipolar vectors where 4 agree on position 0 (+1) and 1
        disagrees (−1).  Verifies that the quantum majority bundle correctly
        assigns +1 to position 0 via quantum interference.
        """

        dimensionality = 16
        n_sys = quantum_encode(np.ones(dimensionality, dtype=int)).num_qubits

        np.random.seed(7)
        # Base vector: position 0 = +1
        base = np.ones(dimensionality, dtype=int)
        base[1:] = np.random.choice([-1, 1], size=dimensionality - 1)

        # 4 copies of base (position 0 = +1) + 1 negated (position 0 = -1)
        vectors = [base.copy() for _ in range(4)]
        vectors.append(-base.copy())  # minority at every position
        oracle_circuits = [quantum_encode(v) for v in vectors]

        # Classical reference: majority vote
        classical_bundled = Vector(
            size=dimensionality, vtype="bipolar",
            vector=np.sign(sum(v.astype(float) for v in vectors)).astype(int),
        )

        with self.subTest("returns an oracle circuit"):
            result_circ = quantum_majority_bundle(oracle_circuits)
            self.assertIsInstance(result_circ, QuantumCircuit)

        with self.subTest("decoded majority at position 0 is +1"):
            result_circ = quantum_majority_bundle(oracle_circuits)
            v_recovered = statevector_to_bipolar(result_circ)
            self.assertEqual(v_recovered[0], 1)

        with self.subTest("full result matches classical bundle (normalised)"):
            result_circ = quantum_majority_bundle(oracle_circuits)
            v_recovered = statevector_to_bipolar(result_circ)
            self.assertTrue(
                np.array_equal(classical_bundled.vector, v_recovered),
                msg=f"Quantum: {v_recovered}\nClassical: {classical_bundled.vector}",
            )

    def test_quantum_contextual_bind(self):
        """Unit tests for hdlib/arithmetic/quantum.py:quantum_contextual_bind

        Tests that the contextual binding circuit:
        (i)  has the correct number of qubits (n_idx + n_sys);
        (ii) the full state has non-zero entanglement between index and system;
        (iii) post-selecting on index = 0 recovers |bind(C, v0)⟩ on the system.
        """

        dimensionality = 4  # 2 qubits per register (n_sys = 2)
        v_ctx = Vector(size=dimensionality, vtype="bipolar", seed=20)
        v_val0 = Vector(size=dimensionality, vtype="bipolar", seed=21)
        v_val1 = Vector(size=dimensionality, vtype="bipolar", seed=22)

        context_circ = quantum_encode(v_ctx.vector)
        val_circs = [quantum_encode(v_val0.vector), quantum_encode(v_val1.vector)]
        K = len(val_circs)
        n_sys = context_circ.num_qubits  # 2
        n_idx = max(1, math.ceil(math.log2(K))) if K > 1 else 1  # 1

        qc_ctx = quantum_contextual_bind(context_circ, val_circs)

        with self.subTest("circuit has n_idx + n_sys qubits"):
            self.assertEqual(qc_ctx.num_qubits, n_idx + n_sys)

        with self.subTest("state has non-zero entanglement between idx and sys"):
            sv = Statevector.from_instruction(qc_ctx.decompose().decompose())
            # Trace over system qubits to get index-only density matrix
            rho_idx = partial_trace(sv, list(range(n_idx, n_idx + n_sys)))
            ent = entropy(rho_idx)
            self.assertGreater(ent, 0.0)

        with self.subTest("post-select index=0 recovers |bind(C, v0)⟩"):
            sv = Statevector.from_instruction(qc_ctx.decompose().decompose())
            sv_data = np.asarray(sv.data)

            # idx occupies the lowest n_idx bits; idx=0 → indices divisible by 2^n_idx
            step = 2 ** n_idx
            sys_amps = sv_data[::step]  # amplitudes for idx=0, length = 2^n_sys
            norm = np.linalg.norm(sys_amps)
            self.assertGreater(norm, 0.0)
            sys_amps /= norm

            # Reference: |bind(C, v0)⟩ = O_C · O_v0 |+⟩^n
            bound_op = quantum_bind([context_circ, val_circs[0]])
            qc_ref = QuantumCircuit(n_sys)
            qc_ref.h(range(n_sys))
            qc_ref.compose(bound_op, inplace=True)
            sv_ref = Statevector.from_instruction(qc_ref.decompose().decompose())

            fidelity = abs(np.dot(np.conj(sv_ref.data), sys_amps)) ** 2
            self.assertAlmostEqual(fidelity, 1.0, delta=0.05)

if __name__ == "__main__":
    unittest.main()
