#!/usr/bin/env python3
"""
Unit tests for hdlib
"""

import errno
import os
import sys
import tempfile
import unittest

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

# Define the test root directory
TESTS_DIR = os.path.dirname(os.path.realpath(__file__))

# Define the hdlib root directory
HDLIB_DIR = os.path.join(os.path.dirname(TESTS_DIR), "hdlib")

# This is required to import the functions we need to test
sys.path.append(HDLIB_DIR)

# Finally import the space, vector, and arithmetic operators
from space import Space, Vector
from arithmetic import bundle, bind, permute
from model import Model


class TestHDLib(unittest.TestCase):
    """
    Unit tests for hdlib
    """

    def test_vector(self):
        """
        Unit tests for hdlib/space.py:Vector class
        """

        # Create a random binary numpy.ndarray
        ndarray = np.random.randint(2, size=10000)

        # Create a Vector object with ndarray
        binary_vector = Vector(vector=ndarray)

        with self.subTest():
            # Test the vector type: must be binary here
            self.assertEqual(binary_vector.vtype, "binary")

        # Create a Vector object with a bipolar numpy.ndarray
        bipolar_vector = Vector(vector=(2 * ndarray - 1))

        with self.subTest():
            # Test the vector type: must be bipolar here
            self.assertEqual(bipolar_vector.vtype, "bipolar")

        with self.subTest():
            # Cannot create Vectors with size < 10000
            self.assertRaises(ValueError, Vector, size=10)

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
        """
        Unit tests for hdlib/space.py:Space class
        """

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

        with self.subTest():
            # Cannot create a Space with vector size < 10000
            self.assertRaises(ValueError, Space, size=10)

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

    def test_model(self):
        """
        Unit tests for hdlib/model.py:Model class
        """

        # Create a model with vector dimensionality 10000, vector type bipolar, and 10 levels
        model = Model(size=10000, levels=10, vtype="bipolar")

        # Use the IRIS dataset from sklearn
        iris = datasets.load_iris()

        # Get data points and classes
        points = iris.data.tolist()
        classes = iris.target.tolist()

        # Fit the model
        model.fit(points, classes)

        with self.subTest():
            # There should be N data points plus 10 level vectors in the space
            self.assertEqual(len(model.space.memory()), len(points) + 10)

        # 5-folds cross-validation
        # 10 retraining iterations
        predictions = model.cross_val_predict(points, classes, cv=5, retrain=10)

        with self.subTest():
            # There should be a prediction for each fold
            self.assertEqual(len(predictions), 5)

        # Collect the accuracy scores computed on each fold
        scores = list()

        for y_indices, y_pred, _ in predictions:
            y_true = [label for position, label in enumerate(classes) if position in y_indices]
            accuracy = accuracy_score(y_true, y_pred)
        
            with self.subTest():
                self.assertTrue(accuracy > 0.0)

    def test_dollar_of_mexico(self):
        """
        Reproduce the "What is the Dollar of Mexico?"

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


if __name__ == "__main__":
    unittest.main()
