# hdlib

Hyperdimensional Computing Library for building Vector Symbolic Architectures in Python 3.

![Conda](https://img.shields.io/conda/dn/conda-forge/hdlib?label=hdlib%20on%20Conda)

## Install

It is available through `pip` and `conda`.
Please, use one of the following commands to start playing with `hdlib`:

```bash
# Install with pip
pip install hdlib

# Install with conda
conda install -c conda-forge hdlib
```

## Usage

The `hdlib` library provides two main modules, `space` and `arithmetic`. The first one contains constructors of `Space` and `Vector` objects that can be used to build vectors and the space that hosts them. The second module, called `arithmetic`, contains a bunch of functions to operate on vectors.

```python
from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, bundle, permute
```

### Hyperdimensional Vectors

Vector objects in `hdlib` can be created through the `Vector` class whose constructor requires the following parameters:

| Parameter   | Default   | Mandatory | Description  |
|:------------|:----------|:---------:|:-------------|
| `name`      |           |           | Name of the vector. It is automatically generated in case it is not specified |
| `size`      | `10000`   | ⚑         | Vector dimensionality usually in the order of 10,000 |
| `vector`    |           |           | `numpy.ndarray` object. If specified, `size` and `vtype` are automatically inferred from the vector itself |
| `vtype`     | `bipolar` | ⚑         | Vector type: `bipolar` or `binary` |
| `tags`      |           |           | List of tags used to characterize vectors. Useful to easily retrieve vector with specific tags |
| `seed`      |           |           | Seed for reproducibility purposes |
| `warning`   | `False`   |           | Print warning messages if `True` |
| `from_file` |           |           | Path to a pickle file to load a precomputed Vector |

There are three different ways to initialize Vector objects:

```python
# With no spacific parameters
# This creates a random bipolar vector with size 10,000 by default
vector = Vector()

# By creating a numpy.ndarray object first
# A binary vector in this case
import numpy as np
ndarray = np.random.randint(2, size=size)
vector = Vector(vector=ndarray)

# By loading a precomputed Vector object
vector = Vector(from_file="~/vector.pkl")
```

> **Note**
> In this last example, a Vector object is built by loading the content of a pickle file. Vector objects can be saved to pickle files with the `dump()` method as in the following example: `vector.dump(to_file="~/vector.pkl")`

Here is the list of Vector class methods:

| Method | Signature                     | Description  |
|:-------|:------------------------------|:-------------|
| `dist` | `vector: Vector, method: str` | Compute the `cosine`, `hamming`, or `euclidean` distance with another Vector object |
| `dump` | `to_file: str`                | Dump the Vector object to a pickle file |

### Hyperdimensional Space

Vectors are stored into a so called hyperdimensional space that can be defined through the `Space` constructor that requires the following parameters:

| Parameter   | Default   | Mandatory | Description  |
|:------------|:----------|:---------:|:-------------|
| `size`      | `10000`   | ⚑         | Used to create vectors of the same length that all share the same hyperdimensional space |
| `vtype`     | `bipolar` | ⚑         | Vectors in the space must have all the same type: `bipolar` or `binary` |
| `from_file` |           |           | Path to a pickle file to load a precomputed Space |

There are two ways to initialize Space objects:

```python
# With no specific parameters
# This creates a space that can host random bipolar vectors with size 10,000 by default
space = Space()

# By loading a precomputed Space object
space = Space(from_file="~/space.pkl")
```

> **Note**
> In this last example, similarly to Vector objects, a Space object is built by loading the content of a pickle file. Space objects can be saved to pickle files with the `dump()` method as in the following example: `space.dump(to_file="~/space.pkl")`

Here is the list of Space class methods:

| Method       | Signature                                       | Description  |
|:-------------|:------------------------------------------------|:-------------|
| `memory`     |                                                 | Return a list with Vector IDs |
| `get`        | `names: list, tags: list`                       | Return a list of Vector objects based on a list of Vector IDs or tags |
| `insert`     | `vector: Vector`                                | Insert a Vector object into the Space |
| `bulkInsert` | `names: list, tags: list`                       | Automatically create a Vector object for each of the ID in the input `names` list and finally insert them into the Space. Also tag vectors based on tags in the `tags` list of lists. Tags in position `i` are assigned to the Vector object whose name is in position `i` of the `vectors` list |
| `remove`     | `name: str`                                     | Remove a Vector object from the Space based on its ID |
| `add_tag`    | `name: str, tag: str`                           | Assign a tag to a Vector object in the Space |
| `remove_tag` | `name: str, tag: str`                           | Remove a tag to a Vector object in the Space |
| `link`       | `name1: str, name2: str`                        | Link two vectors in the Space. Note that links are directed |
| `set_root`   | `name: str`                                     | Vector links can be used to define a tree structure. Set a specific vector as root |
| `find`       | `vector: Vector, threshold: float, method: str` | Given a specific Vector object, search for the closest Vector in the Space according to a specific distance metric: `cosine`, `hamming`, or `euclidean` |
| `find_all`   | `vector: Vector, threshold: float, method: str` | Report the distance between the input Vector object and all the other Vectors in the Space |
| `dump`       | `to_file: str`                                  | Dump the Space object to a pickle file |

### Arithmetic Operations

A Vector Symbolic Architecture (a.k.a. Hyperdimensional Computing) is composed of vectors in the hyperdimensional space and a series of arithmetic operations to manipulate vectors.

The `hdlib` library provides three operators under the `arithmetic` module: `bundle`, `bind`, and `permute`.

Here are the characteristics of these operators:

| Operator       | Properties  |
|:---------------|:------------|
| `bundle`       | (i) The resulting vector is similar to the input vectors, (ii) the more vectors are involved in bundling, the harder it is to determine the component vectors, and (iii) if several copies of any vector are included in bundling, the resulting vector is closer to the dominant vector than to the other components |
| `bind`         | (i) Invertible (unbind), (ii) it distributes over bundling, (iii) it preserves the distance, and (iv) the resulting vector is dissimilar to the input vectors |
| `permute`      | (i) Invertible, (ii) it distributes over bundling and any elementwise operation, (iii) it preserves the distance, and (iv) the resulting vector is dissimilar to the input vectors |

## Contributing

Long-term discussion and bug reports are maintained via GitHub Issues, while code review is managed via GitHub Pull Requests.

Please, (i) be sure that there are no existing issues/PR concerning the same bug or improvement before opening a new issue/PR; (ii) write a clear and concise description of what the bug/PR is about; (iii) specifying the list of steps to reproduce the behavior in addition to versions and other technical details is highly recommended.
