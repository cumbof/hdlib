"""
Implementation of hyperdimensional Vector and Space
"""

import os
import pickle
import uuid

import numpy as np
from typing import List, Optional, Set, Tuple, Union

from hdlib import __version__


class Vector(object):
    """
    Vector object
    """

    def __init__(
        self,
        name: Optional[str]=None,
        size: int=10000,
        vector: Optional[np.ndarray]=None,
        vtype: Optional[str]="bipolar",
        tags: Optional[Set[Union[str, int, float]]]=None,
        seed: Optional[int]=None,
        warning: bool=False,
        from_file: Optional[os.path.abspath]=None,
    ) -> "Vector":
        """
        Initialize a Vector object

        :param name:        Vector name or ID
        :param size:        Vector size
        :param vector:      Vector as numpy.ndarray
        :param vtype:       Vector type: bipolar or binary
        :param tags:        Set of tags for grouping vectors
        :param seed:        Random seed for reproducibility
        :param warning:     Print warning messages if True
        :param from_file:   Load a vector from pickle file
        :return:            A Vector object
        """

        # Conditions on vector name or ID
        # Vector name is casted to string. For this reason, only Python primitives are allowed
        # A random name is assigned if not specified
        try:
            if name is None:
                name = str(uuid.uuid4())

            else:
                name = str(name)

            self.name = name

        except:
            raise TypeError("Vector name must be instance of a primitive")

        # Register random seed for reproducibility 
        self.seed = seed

        # Add tags
        self.tags = tags if tags else set()

        # Add links
        # Used to link Vectors by their names or IDs
        self.parents = set()
        self.children = set()

        # Conditions on vector
        # It must be a numpy.ndarray
        # A random vector is generated if not specified
        if vector is not None:
            if not isinstance(vector, np.ndarray):
                raise TypeError("Vector must be instance of numpy.ndarray")

            self.vector = vector

            self.size = len(self.vector)

            if self.size < 10000:
                raise ValueError("Vector size must be greater than or equal to 10000")

            self.vtype = vtype

            # Try to infer the vector type from the content of the vector itself
            if ((self.vector == 0) | (self.vector == 1)).all():
                self.vtype = "binary"

            elif ((self.vector == -1) | (self.vector == 1)).all():
                self.vtype = "bipolar"

            else:
                if warning:
                    print("Vector type can be binary or bipolar only")

        elif from_file and os.path.isfile(from_file):
            # Load vector from pickle file
            with open(from_file, "rb") as pkl:
                version, self.name, self.size, self.vector, self.vtype, self.parents, self.children, self.tags, self.seed = pickle.load(pkl)

            if version != __version__:
                print("Warning: the specified Space has been created with a different version of hdlib")

        else:
            # Conditions on vector size
            # It must be an integer number greater than or equal to 10000
            # This size makes sure that vectors are quasi-orthogonal in space
            if not isinstance(size, int):
                raise TypeError("Vector size must be an integer number")

            if size < 10000:
                raise ValueError("Vector size must be greater than or equal to 10000")

            self.size = size

            # Add vector type
            self.vtype = vtype.lower()

            if vtype not in ("bipolar", "binary"):
                raise ValueError("Vector type can be binary or bipolar only")

            if seed is None:
                rand = np.random

            else:
                # Conditions on random seed for reproducibility
                # numpy allows integers as random seeds
                if not isinstance(seed, int):
                    raise TypeError("Seed must be an integer number")

                rand = np.random.default_rng(seed=self.seed)

            # Build a random binary vector
            self.vector = rand.randint(2, size=size)

            if vtype == "bipolar":
                # Build a random bipolar vector
                self.vector = 2 * self.vector - 1

    def __len__(self) -> int:
        """
        Get the vector size

        :return:    The length of vector
        """

        return self.size

    def dist(self, vector: "Vector", method: str="cosine") -> float:
        """
        Compute distance between vectors

        :param vector:      Vector object
        :param method:      Distance method: cosine, hamming, euclidean
        :return:            Distance
        """

        if self.size != vector.size:
            raise Exception("Vectors must have the same size")

        if self.vtype != vector.vtype:
            raise Exception("Vectors must be of the same type")

        if method.lower() == "cosine":
            return np.dot(self.vector, vector.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(vector.vector))

        elif method.lower() == "hamming":
            return np.count_nonzero(self.vector != vector.vector)

        elif method.lower() == "euclidean":
            return np.linalg.norm(self.vector - vector.vector)

        else:
            raise ValueError("Distance method \"{}\" is not supported".format(method))

    def normalize(self) -> None:
        """
        Normalize a vector after a bind or bundle operation
        """

        if self.vtype not in ("bipolar", "binary"):
            raise Exception("Vector type is not supported")

        self.vector[self.vector > 0] = 1

        self.vector[self.vector <= 0] = 0 if self.vtype == "binary" else -1

    def dump(self, to_file: Optional[os.path.abspath]=None) -> None:
        """
        Dump the hyperdimensional vector to a pickle file

        :param to_file:     Path to the output pickle file
        """

        if not to_file:
            # Dump the vector to a pickle file in the current working directory if not file path is provided
            to_file = os.path.join(os.getcwd, "{}.pkl".format(self.name))

        if os.path.isfile(to_file):
            raise Exception("The output file already exists!\n{}".format(to_file))

        with open(to_file, "wb") as pkl:
            pickle.dump((__version__, self.name, self.size, self.vector, self.vtype, self.parents, self.children, self.tags, self.seed), pkl)


class Space(object):
    """
    Vectors space
    """

    def __init__(self, size: int=10000, vtype: Optional[str]="bipolar", from_file: Optional[os.path.abspath]=None) -> "Space":
        """
        Initialize the vectors space as a dictionary of Vector objects

        :param size:        Size of vectors in the space
        :param vtype:       Vector type: bipolar or binary
        :param from_file:   Load a space from pickle file
        :return:            A Space object
        """

        self.space = dict()

        self.size = size

        if self.size < 10000:
            raise ValueError("Size of vectors in space must be greater than or equal to 10000")

        self.vtype = vtype.lower()

        if self.vtype not in ("binary", "bipolar"):
            raise ValueError("Vector type not supported")

        self.tags = dict()

        # Vector links can be used to define a tree structure
        # Use this flag to mark a vector as root
        self.root = None

        if from_file and os.path.isfile(from_file):
            with open(from_file, "rb") as pkl:
                version, self.size, self.vtype, self.space, self.root = pickle.load(pkl)

            if version != __version__:
                print("Warning: the specified Space has been created with a different version of hdlib")

            for name in self.space:
                if self.space[name].tags:
                    for tag in self.space[name].tags:
                        if tag not in self.tags:
                            self.tags[tag] = set()

                        self.tags[tag].add(name)

    def __len__(self) -> int:
        """
        Get the space size

        :return:    The number of vectors in the space
        """

        return len(self.space)

    def memory(self) -> List[str]:
        """
        Return names or IDs of vectors in space

        :return:    Names or IDs of vectors in space
        """

        return list(self.space.keys())

    def get(
        self,
        names: Optional[List[str]]=None,
        tags: Optional[List[Union[str, int, float]]]=None
    ) -> List[Vector]:
        """
        Get vectors by names or tags

        :param names:   List of vector names or IDs
        :param tags:    List of tags
        :return:        List of vectors in space
        """

        if not names and not tags:
            raise Exception("No names or tags provided!")            

        vectors = set()

        if names:
            try:
                names = [str(name) for name in names]

            except:
                raise TypeError("Vector name must be instance of a primitive")

            for vector_name in names:
                if vector_name in self.space:
                    vectors.add(self.space[vector_name])

        elif tags:
            for tag in tags:
                if not isinstance(tag, str) and not isinstance(tag, int) and not isinstance(tag, float):
                    raise TypeError("Tags must be string, integer, or float")

                if tag in self.tags:
                    for vector_name in self.tags[tag]:
                        vectors.add(self.space[vector_name])

        return list(vectors)

    def insert(self, vector: Vector) -> None:
        """
        Add a Vector object to the space

        :param vector:      Vector object
        """

        if self.size != vector.size:
            raise Exception("Space and vectors with different size are not compatible")

        if self.vtype != vector.vtype:
            raise Exception("Attempting to insert a {} vector into a {} space: failed".format(vector.vtype, self.vtype))

        if vector.name in self.space:
            raise Exception("Vector \"{}\" already in space".format(vector.name))

        self.space[vector.name] = vector

        for tag in vector.tags:
            if tag not in self.tags:
                self.tags[tag] = set()

            self.tags[tag].add(vector.name)

    def bulkInsert(
        self,
        names: List[str],
        tags: Optional[List[List[Union[str, int, float]]]]=None,
        ignoreExisting: bool=True
    ) -> None:
        """
        Add vectors to the space

        :param names:           List of vector unique names or IDs
        :param tags:            List of tags with the same size of vectors: tags of vectors[i] is in tags[i]
        :param ignoreExisting:  Do not throw an error in case a vector already exists in the space
        """

        if not isinstance(names, list):
            raise TypeError("Input must be a list of strings")

        if tags and not isinstance(tags, list):
            raise TypeError("tags must be a list of lists of strings")

        if tags and len(names) != len(tags):
            raise Exception("The number of vector IDs must match the size of the tags list")

        names = list(set(names))

        for pos, name in enumerate(names):
            try:
                name = str(name)

            except:
                raise TypeError("Entries in input list must be instances of primitives")

            if name in self.space:
                if not ignoreExisting:
                    raise Exception("Vector \"{}\" already in space".format(name))

                else:
                    continue

            vector_tags = set(tags[pos]) if tags else set()

            vector = Vector(name=name, size=self.size, tags=vector_tags, vtype=self.vtype)

            self.space[vector.name] = vector

            for tag in vector_tags:
                if tag not in self.tags:
                    self.tags[tag] = set()

                self.tags[tag].add(name)

    def remove(self, name: str) -> Vector:
        """
        Remove a vector from the space

        :param name:    Vector name or ID
        :return:        Vector removed from the space
        """

        try:
            name = str(name)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if name in self.space:
            vector = self.space[name]

            del self.space[name]

            for tag in vector.tags:
                self.tags[tag].remove(vector.name)

                if not self.tags[tag]:
                    del self.tags[tag]

            return vector

        raise Exception("Vector not in space")

    def add_tag(self, name: str, tag: Union[str, int, float]) -> None:
        """
        Tag a vector

        :param name:    Vector name or ID
        :param tag:     Vector tag
        """

        try:
            name = str(name)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if name not in self.space:
            raise Exception("Vector not in space")

        if not isinstance(tag, str) and not isinstance(tag, int) and not isinstance(tag, float):
            raise TypeError("Tags must be string, integer, or float")

        self.space[name].tags.add(tag)

        if tag not in self.tags:
            self.tags[tag] = set()

        self.tags[tag].add(name)

    def remove_tag(self, name: str, tag: Union[str, int, float]) -> None:
        """
        Untag a vector

        :param name:    Vector name or ID
        :param tag:     Vector tag
        """

        try:
            name = str(name)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if not isinstance(tag, str) and not isinstance(tag, int) and not isinstance(tag, float):
            raise TypeError("Tags must be string, integer, or float")

        if name not in self.space:
            raise Exception("Vector not in space")

        if tag in self.tags:
            self.space[name].tags.remove(tag)

            self.tags[tag].remove(name)

            if not self.tags[tag]:
                del self.tags[tag]

    def link(self, name1: str, name2: str) -> None:
        """
        Link the current Vector object with another vector through its name.
        Links are directed

        :param name1:   Vector #1 name or ID
        :param name2:   Vector #2 name or ID
        """

        try:
            name1 = str(name1)

            name2 = str(name2)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if name1 not in self.space:
            raise Exception("Vector \"{}\" not in space".format(name1))

        if name2 not in self.space:
            raise Exception("Vector \"{}\" not in space".format(name2))

        self.space[name1].children.add(name2)

        self.space[name2].parents.add(name1)

    def set_root(self, name: str) -> None:
        """
        Vector links can be used to define a tree structure.
        Set a specific vector as root

        :param name:    Vector name or ID
        """

        try:
            name = str(name)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if name not in self.space:
            raise Exception("Vector \"{}\" not in space".format(name))

        self.root = name

    def find(self, vector: Vector, threshold: float=-1.0, method: str="cosine") -> Tuple[str, float]:
        """
        Search for the closest vector in space

        :param vector:      Vector object
        :param threshold:   Do not consider entries in memory with a distance lower than this threshold
        :param method:      Distance method: cosine, hamming, euclidean
        :return:            Name of the closest vector in space and its distance from the input vector
        """

        # Exploit self.find_all() to seach for the best match
        # It will take care of raising exceptions in case of problems with input arguments
        distances, best = self.find_all(vector, threshold=threshold, method=method)

        return best, distances[best]

    def find_all(self, vector: Vector, threshold: float=-1.0, method: str="cosine") -> Tuple[dict, str]:
        """
        Compute distance of the input vector against all vectors in space

        :param vector:      Vector object
        :param threshold:   Do not consider entries in memory with a distance lower than this threshold
        :param method:      Distance method: cosine, hamming, euclidean
        :return:            Dictionary with distances of the input vector against all the other vectors in space, and the name of the closest vector in space
        """

        if self.size != vector.size:
            raise Exception("Space and vectors with different size are not compatible")

        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("Threshold cannot be lower than -1.0 or higher than 1.0")

        distances = dict()

        distance = -1.0

        best = None

        for v in self.space:
            # Compute distance
            dist = self.space[v].dist(vector, method=method)

            if dist >= threshold:
                distances[v] = dist

                if distances[v] > distance:
                    best = v

                    distance = distances[v]

        return distances, best

    def dump(self, to_file: Optional[os.path.abspath]=None) -> None:
        """
        Dump the hyperdimensional space to a pickle file

        :param to_file:     Path to the output pickle file
        """

        if not to_file:
            # Dump the space to a pickle file in the current working directory if not file path is provided
            to_file = os.path.join(os.getcwd, "space.pkl")

        if os.path.isfile(to_file):
            raise Exception("The output file already exists!\n{}".format(to_file))

        with open(to_file, "wb") as pkl:
            pickle.dump((__version__, self.size, self.vtype, self.space, self.root), pkl)
