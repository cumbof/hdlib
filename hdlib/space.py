"""
Implementation of hyperdimensional Vector and Space
"""

import os
import pickle
import uuid

import numpy as np
from typing import List, Optional, Tuple, Union


class Vector(object):
    """
    Vector object
    """

    def __init__(
        self,
        name: Optional[str]=None,
        size: int=1000,
        vector: Optional[np.ndarray]=None,
        tags: Optional[List[Union[str, int, float]]]=None,
        seed: Optional[int]=None
    ) -> "Vector":
        """
        Initialize a Vector object

        :param name:        Vector name or id
        :param size:        Vector size
        :param vector:      Vector as numpy.ndarray
        :param tags:        Tags for grouping vectors
        :param seed:        Random seed for reproducibility
        :return:            A Vector object
        """

        # Conditions on vector name or id
        # Vector name is casted to string. For this reason, only Python primitives are allowed
        # A random name is assigned if not specified
        try:
            if name is None:
                name = str(uuid.uuid4())

            else:
                name = str(name)

            self.name = name

        except:
            raise TypeError("Vector name must be an instance of a primitive")

        # Conditions on vector size
        # It must be an integer number greater than 1000
        if not isinstance(size, int):
            raise TypeError("Vector size must be an integer number")

        if size < 1000:
            raise ValueError("Vector size must be greater than 1000")

        self.size = size        

        # Register random seed for reproducibility 
        self.seed = seed

        # Add tags
        self.tags = tags if tags else list()

        # Conditions on vector
        # It must be a Numpy ndarray
        # A random bipolar vector is generated if not specified
        if vector is not None:
            if not isinstance(vector, np.ndarray):
                raise TypeError("Vector must be an instance of numpy.ndarray")

            self.vector = vector

        else:
            if seed is None:
                rand = np.random

            else:
                # Conditions on random seed for reproducibility
                # Numpy allows integers as random seeds
                if not isinstance(seed, int):
                    raise TypeError("Seed must be an integer number")

                rand = np.random.default_rng(seed=self.seed)

            # Build a random bipolar vector
            self.vector = 2*rand.randint(2, size=size)-1

    def __len__(self) -> int:
        """
        Get the vector size

        :return:    The length of vector
        """

        return self.size

    def dist(self, vector: "Vector") -> float:
        """
        Compute distance between vectors as cosine similarity

        :param vector:      Vector object
        :return:            Cosine similarity
        """

        if not isinstance(vector, Vector):
            raise TypeError("Input must be a Vector object")

        if self.size != vector.size:
            raise Exception("Vectors must have the same size")

        return np.dot(self.vector, vector.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(vector.vector))


class Space(object):
    """
    Vectors space
    """

    def __init__(self, size: int=1000, from_file: Optional[os.path.abspath]=None) -> "Space":
        """
        Initialize the vectors space as a dictionary of Vector objects

        :param size:        Size of vectors in the space
        :param from_file:   Load a space from file
        :return:            A Space object
        """

        self.space = dict()

        self.size = size

        if from_file and os.path.isfile(from_file):
            self.size, self.space = pickle.load(open(from_file, "rb"))

    def __len__(self) -> int:
        """
        Get the space size

        :return:    The number of vectors in the space
        """

        return len(self.space)

    def memory(self) -> List[str]:
        """
        Return ids of vectors in space

        :return:    Names or ids of vectors in space
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

        try:
            vectors = list()

            if names:
                names = [str(name) for name in names]

                for vector_name in names:
                    if vector_name in self.space:
                        vectors.append(self.space[vector_name])

            elif tags:
                for tag in tags:
                    if not isinstance(tag, str) and not isinstance(tag, int) and not isinstance(tag, float):
                        raise TypeError("Tags must be string, integer, or float")

                for vector_name in self.space:
                    if set(tags).intersection(self.space[vector_name].tags):
                        vectors.append(self.space[vector_name])

            return vectors

        except:
            raise TypeError("Input must be an instance of a primitive")

    def insert(self, vector: Vector) -> None:
        """
        Add a Vector object to the space

        :param vector:      Vector object
        """

        if not isinstance(vector, Vector):
            raise TypeError("Input must be a Vector object")

        if self.size != vector.size:
            raise Exception("Space and vectors with different size are not compatible")

        if vector.name in self.space:
            raise Exception("Vector \"{}\" already in space".format(vector.name))

        self.space[vector.name] = vector

    def bulkInsert(
        self,
        vectors: List[str],
        tags: Optional[List[List[Union[str, int, float]]]]=None,
        ignoreExisting: bool=True
    ) -> None:
        """
        Add vectors to the space

        :param vectors:         List of vector unique names or ids
        :param tags:            List of tags with the same size of vectors: tags of vectors[i] is in tags[i]
        :param ignoreExisting:  Do not throw an error in case a vector already exists in the space
        """

        if not isinstance(vectors, list):
            raise TypeError("Input must be a list of strings")

        if tags and not isinstance(tags, list):
            raise TypeError("tags must be a list of lists of strings")

        if tags and len(vectors) != len(tags):
            raise Exception("The number of vectors must match the size of the tags list")

        vectors = list(set(vectors))

        for pos, name in enumerate(vectors):
            try:
                name = str(name)

                if name in self.space:
                    if not ignoreExisting:
                        raise Exception("Vector \"{}\" already in space".format(name))

                    else:
                        continue

                vector_tags = tags[pos] if tags else list()

                vector = Vector(name=name, size=self.size, tags=vector_tags)

                self.space[vector.name] = vector

            except:
                raise TypeError("Entries in input list must be instances of primitives")

    def remove(self, name: str) -> Vector:
        """
        Remove a vector from the space

        :param name:    Vector name or id
        :return:        Vector removed from the space
        """

        try:
            name = str(name)

            if name in self.space:
                vector = self.space[name]

                del self.space[name]

                return vector

            raise Exception("Vector not in space")

        except:
            raise TypeError("Input must be an instance of a primitive")

    def find(self, vector: Vector, threshold: float=-1.0) -> Tuple[str, float]:
        """
        Search for the closest vector in space

        :param vector:      Vector object
        :param threshold:   Do not consider entries in memory with a distance lower than this threshold
        :return:            Name of the closest vector in space and its distance from the input vector
        """

        # Exploit self.findAll() to seach for the best match
        # It will take care of raising exceptions in case of problems with input arguments
        distances, best = self.findAll(vector, threshold=threshold)

        return best, distances[best]

    def findAll(self, vector: Vector, threshold: float=-1.0) -> Tuple[dict, str]:
        """
        Compute distance of the input vector against all vectors in space

        :param vector:      Vector object
        :param threshold:   Do not consider entries in memory with a distance lower than this threshold
        :return:            Dictionary with distances of the input vector against all the other vectors in space, and the name of the closest vector in space
        """

        if not isinstance(vector, Vector):
            raise TypeError("Input must be a Vector object")

        if self.size != vector.size:
            raise Exception("Space and vectors with different size are not compatible")

        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("Threshold cannot be lower than -1.0 or higher than 1.0")

        distances = dict()

        distance = -1.0

        best = None

        for v in self.space:
            # Compute cosine similarity
            dist = self.space[v].dist(vector)

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
            # Dump the space to a pickle file in the current working directory
            # if not file path is provided
            to_file = os.path.join(os.getcwd, "space.pkl")

        if os.path.isfile(to_file):
            raise Exception("The output file already exists!\n{}".format(to_file))

        with open(to_file, "wb") as pkl:
            pickle.dump((self.size, self.space), pkl)
