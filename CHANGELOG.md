# Changelog

## Version 2.1.0

[@cumbof/hdlib@2.1.0](https://github.com/cumbof/hdlib/releases/tag/2.1.0)

## New features

- `Space` and `Vector` objects are not constrained on a 10,000 dimensionality anymore;
- The `predict` function in `hdlib.model.classification.ClassificationModel` now returns the actual distances between the test samples and the class prototypes too;
- The classical arithmetic has been moved to `hdlib.arithmetic`;
- `hdlib.arithmetic.quantum` provides the quantum arithmetic at the base of Quantum Hyperdimensional Computing;
- The `hdlib.model.classification` module now contains a new `QuantumClassificationModel` class;
- New reasoning and classification examples about the [Quantum Hyperdimensional Computing](https://github.com/cumbof/hdlib/blob/main/examples/quantum) are available under the examples folder.

## Version 2.0.0

[@cumbof/hdlib@2.0.0](https://github.com/cumbof/hdlib/releases/tag/2.0.0)

## New features

- The `Vector` class is now in `hdlib.vector`;
- `hdlib.model.MLModel` is now `hdlib.model.classification.ClassificationModel`;
- `hdlib.graph.Graph` is now `hdlib.model.graph.GraphModel`;
- The `hdlib.parser` module has been suppressed and its functions have been moved to [examples/chopin2/chopin2.py](https://github.com/cumbof/hdlib/blob/main/examples/chopin2/chopin2.py);
- New `hdlib.model.graph.GraphModel._error_rate` and `hdlib.model.graph.GraphModel._predict` static methods for the estimation of the model error rate and the evaluation of a graph model in multiprocessing;
- New `hdlib.model.regression.RegressionModel` and `hdlib.model.clustering.ClusteringModel` classes for building regression and clustering models respectively;
- New examples about the [classification of chemical structures](https://github.com/cumbof/hdlib/blob/main/examples/tox21) and the [classification of viral species via graph encoding](https://github.com/cumbof/hdlib/blob/main/examples/pangenome).

## Fixes

- `hdlib.model.graph.GraphModel.error_mitigation` does not take into account for false positives;
- `hdlib.model.graph.GraphModel.error_rate` and `hdlib.model.graph.GraphModel.predict` now check whether an edge exist using weight-specific thresholds.

## Version 0.1.20

[@cumbof/hdlib@0.1.20](https://github.com/cumbof/hdlib/releases/tag/0.1.20)

### Fixes

- Replace `numpy.NINF` with `-numpy.inf` as a change introduced in Numpy 2.0.

## Version 0.1.19

[@cumbof/hdlib@0.1.19](https://github.com/cumbof/hdlib/releases/tag/0.1.19)

### Fixes

- Replace `numpy.Inf` with `numpy.inf` as a change introduced in Numpy 2.0;
- `hdlib.graph` and the corresponding unittest are now working properly.

## Version 0.1.18

[@cumbof/hdlib@0.1.18](https://github.com/cumbof/hdlib/releases/tag/0.1.18)

### New features

- The space dictionary as part of a `Space` object is now an `OrderedDict`, making `Space` objects iterable over their set of vectors;
- Python examples under the `examples` folder are now available as part of the `hdlib` package.

### Fixes

- Fix dumping and loading `Vector` and `Space` objects to and from pickle files;
- `space.Space.bulk_insert` function now checks whether the names of the input vectors are instances of `bool`, `int`, `float`, `str`, and `None` before creating and inserting vectors into the space;
- Distance thresholds in `space.Space.find` and `space.Space.find_all` are now set to `numpy.Inf` by default.

## Version 0.1.17

[@cumbof/hdlib@0.1.17](https://github.com/cumbof/hdlib/releases/tag/0.1.17)

### New features

- Add the `subtraction` operator to the `arithmetic` module;
- Add `__sub__` to `space.Vector` that makes use of `arithmetic.subtraction` to element-wise subtract two Vector objects;
- Add `space.Vector.subtraction` to element-wise subtract a vector from a Vector object inplace;
- Add `graph.Graph` to build vector-symbolic representations of directed and undirected, weighted and unweighted graphs;
- Extend [test/test.py](https://github.com/cumbof/hdlib/blob/main/test/test.py) with new unit tests.

## Version 0.1.16

[@cumbof/hdlib@0.1.16](https://github.com/cumbof/hdlib/releases/tag/0.1.16)

### New features

- Add `__add__` and `__mul__` to `space.Vector`;
- `model.MLModel.predict` now returns the model error rate.

### Fixes

- `model.Model` is now `model.MLModel`;
- `parser.kfolds_split` has been deprecated and removed;
- `model.MLModel.cross_val_predict` now uses `sklearn.model_selection.StratifiedKFold` for the generation of balanced folds;
- Fix the order of the test real labels before computing the model metrics in [examples/chopin2/chopin2.py](https://github.com/cumbof/hdlib/blob/main/examples/chopin2/chopin2.py).

## Version 0.1.15

[@cumbof/hdlib@0.1.15](https://github.com/cumbof/hdlib/releases/tag/0.1.15)

### New features

- Add [examples/chopin2/chopin2_iris.sh](https://github.com/cumbof/hdlib/blob/main/examples/chopin2/chopin2_iris.sh) as a test case for [examples/chopin2/chopin2.py](https://github.com/cumbof/hdlib/blob/main/examples/chopin2/chopin2.py);
- Add new unit tests to [test/test.py](https://github.com/cumbof/hdlib/blob/main/test/test.py).

### Fixes

- `space.Space.bulk_insert` has been refactored to make use of `space.Space.insert`;
- `parser.load_dataset` now throws a `ValueError` in case of non-numerical datasets;
- Add missing `import os` in `space.Model`.

## Version 0.1.14

[@cumbof/hdlib@0.1.14](https://github.com/cumbof/hdlib/releases/tag/0.1.14)

### Fixes

- `model.Model.fit` now correctly generates both bipolar and binary level vectors;
- `space.Vector.dist` automatically converts the cosine similarity into a distance measure;
- `model.Model.predict` and `model.Model.error_rate` are now compatible with all the supported distance metrics (euclidean, hamming, and cosine).

## Version 0.1.13

[@cumbof/hdlib@0.1.13](https://github.com/cumbof/hdlib/releases/tag/0.1.13)

### Fixes

- Fix the retraining process in `model.Model.predict`.

## Version 0.1.12

[@cumbof/hdlib@0.1.12](https://github.com/cumbof/hdlib/releases/tag/0.1.12)

### New features

- `examples/chopin2/chopin2.py` now reports the Accuracy, F1, Precision, Recall, and the Matthews correlation coefficient for each of the folds in addition to the average of these scores as evaluation metrics of the hyperdimensional computing models;
- `model.Model` class functions now raise different exceptions based on multiple checks on the input parameters.

## Version 0.1.11

[@cumbof/hdlib@0.1.11](https://github.com/cumbof/hdlib/releases/tag/0.1.11)

### Fixes

- The `model.Model.stepwise_regression` function now report the importance corresponding to the best score;
- The `model.Model._init_fit_predict` function uses `average="weighted"` for computing a score different from the accuracy to account for label imbalance;
- `examples/chopin2/chopin2.py` now computes different scores on the resulting predictions, prints the list of selected features based on the best score, and finally reports the confusion matrices.

## Version 0.1.10

[@cumbof/hdlib@0.1.10](https://github.com/cumbof/hdlib/releases/tag/0.1.10)

### New features

- Add `error_rate` as `model.Model` class method for computing the error rate of a classification model.

### Fixes

- The `model.Model.predict` function computes the error rate before retraining the classification model.

## Version 0.1.9

[@cumbof/hdlib@0.1.9](https://github.com/cumbof/hdlib/releases/tag/0.1.9)

### Fixes

- Fix the retraining process in `model.Model.predict` to avoid overfitting.

:warning: Avoid using previous versions of `hdlib`.

## Version 0.1.8

[@cumbof/hdlib@0.1.8](https://github.com/cumbof/hdlib/releases/tag/0.1.8)

### Fixes

- Fix the initialization of Vector objects with a specific seed;
- `model.Model._init_fit_predict` and `model.Model._stepwise_regression_iter` are now private;
- Improving docstring adopting the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) documentation format.

## Version 0.1.7

[@cumbof/hdlib@0.1.7](https://github.com/cumbof/hdlib/releases/tag/0.1.7)

### Fixes

- Fix the break condition in `model.Model.stepwise_regression` for both the `backward` and `forward` methods.

## Version 0.1.6

[@cumbof/hdlib@0.1.6](https://github.com/cumbof/hdlib/releases/tag/0.1.6)

### New features

- Add `stepwise_regression` as `model.Model` class method for performing the feature selection as backward variable elimination or forward variable selection.

## Version 0.1.5

[@cumbof/hdlib@0.1.5](https://github.com/cumbof/hdlib/releases/tag/0.1.5)

### New features

- Add `bind`, `bundle`, and `permute` as `Vector` class methods for applying arithmetic operations inplace;
- Rename `split_dataset` in the `parser` module into `kfolds_split`;
- Add `percentage_split` function to the `parser` module;
- Integrate [chopin2](https://github.com/cumbof/chopin2) ML model into the `model` module withe the `Model` class.

## Version 0.1.4

[@cumbof/hdlib@0.1.4](https://github.com/cumbof/hdlib/releases/tag/0.1.4)

### New features

- Add `parser` module with utility functions for dealing with input datasets;
- Check if the input pickle file exists before initializing `Vector` and `Space` objects with `from_file`;
- Report `Vector` and `Space`objects information when calling `print`;
- Add `examples/chopin2/chopin2.py`: reimplementation of the [chopin2](https://github.com/cumbof/chopin2) ML model with `hdlib`.

### Fixes

- Tags are maintained as sets when applying arithmetic operators.

## Version 0.1.3

[@cumbof/hdlib@0.1.3](https://github.com/cumbof/hdlib/releases/tag/0.1.3)

### New features

- Check for version compatibility when loading a pickle file;
- Add `normalize()` function to the `Vector` class;
- Link vectors in space with `parents` and `children` sets as class attributes;
- Add `link()` and `set_root()` functions to the `Space` class.

## Version 0.1.2

[@cumbof/hdlib@0.1.2](https://github.com/cumbof/hdlib/releases/tag/0.1.2)

### New features

- New distance metrics: `cosine`, `hamming`, and `euclidean`;
- Tags are inherited after applying the arithmetic operators;
- Unit tests for `Vector` and `Space` classes and for `bundle`, `bind`, and `permute` arithmetic operators;
- `What is the Dollar of Mexico?` as a unit test.

## Version 0.1.1

[@cumbof/hdlib@0.1.1](https://github.com/cumbof/hdlib/releases/tag/0.1.1)

### New features

- Vectors can be tagged;
- Tags can be used to retrieve groups of Vectors in the Space with the `get` method of the `Space` class;
- The performances of the `remove` and `findAll` methods of the `Space` class have been improved.

## Version 0.1.0

[@cumbof/hdlib@0.1.0](https://github.com/cumbof/hdlib/releases/tag/0.1.0)

First public release of `hdlib`.

Hyperdimensional Computing Library for building Vector Symbolic Architectures in Python 3
