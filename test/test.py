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
from qiskit.quantum_info import Statevector

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
    statevector_to_bipolar
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
        num_qubits = int(math.log2(dimensionality))

        # Classical encoding
        v_classical = Vector(size=dimensionality, vtype="bipolar")

        # Encode
        # Note: the encode function creates the diagonal operator. To create the state,
        # we must apply it to a uniform superposition
        oracle_circ = quantum_encode(v_classical.vector, label="Test_Vector")

        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        qc.append(oracle_circ, range(num_qubits))

        # Decode
        v_recovered = statevector_to_bipolar(qc)

        self.assertTrue(np.array_equal(v_classical.vector, v_recovered))

    def test_quantum_bind(self):
        """Unit tests for hdlib/arithmetic/quantum.py:bind

        Tests if quantum binding (composing phase oracles) matches classical binding (element-wise multiplication).
        """

        dimensionality = 16
        num_qubits = int(math.log2(dimensionality))

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

        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        qc.append(bound_op, range(num_qubits))

        # Decode
        v_recovered = statevector_to_bipolar(qc)

        self.assertTrue(np.array_equal(v_bound_classical.vector, v_recovered))

    def test_quantum_bundle(self):
        """Unit tests for hdlib/arithmetic/quantum.py:bundle

        Tests if quantum bundling (LCU+OAA) produces a state proportional to the classical bundle (element-wise addition).
        """

        dimensionality = 16
        num_qubits = int(math.log2(dimensionality))

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
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        qc.append(quantum_encode(v_classical.vector), range(num_qubits))

        # Apply permute
        qc = quantum_permute(qc, num_qubits, shift=shift)

        # Decode
        v_recovered = statevector_to_bipolar(qc)

        self.assertTrue(np.array_equal(v_permuted_classical.vector, v_recovered))


if __name__ == "__main__":
    unittest.main()
