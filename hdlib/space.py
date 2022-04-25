__authors__ = ('Fabio Cumbo (fabio.cumbo@gmail.com)')
__version__ = '0.1.0'
__date__ = 'Apr 24, 2022'

import uuid
import numpy as np
from typing import Tuple

class Vector(object):
    """
    Vector object
    """

    def __init__(self, name: str=None, size: int=1000, vector: np.ndarray=None, seed: int=None) -> None:
        """
        Initialize a Vector object

        :param name:        Vector name or id
        :param size:        Vector size
        :param vector:      Vector as numpy.ndarray
        :param seed:        Random seed for reproducibility
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
    
    # Forward reference of Vector class
    def dist(self, vector: 'Vector') -> float:
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

    def __init__(self, size: int=1000) -> None:
        """
        Initialize the vectors space as a dictionary of Vector objects

        :size:int:      Vector size
        """

        self.space = dict()
        self.size = size

    def memory(self) -> list:
        """
        Return ids of vectors in space

        :return:    Names or ids of vectors in space
        """

        return list(self.space.keys())

    def get(self, name: str) -> Vector:
        """
        Get a vector from the space

        :param name:    Vector name or id
        :return:        Vector in space
        """

        try:
            name = str(name)
            if name in self.space:
                return self.space[name]
            raise Exception("Vector not in space")
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

    def bulkInsert(self, vectors: list, ignoreExisting: bool=True) -> None:
        """
        Add vectors to the space

        :param vectors:     List of vector unique names or ids
        """

        if not isinstance(vectors, list):
            raise TypeError("Input must be a list")

        vectors = list(set(vectors))
        for name in vectors:
            try:
                name = str(name)
                if name in self.space:
                    if not ignoreExisting:
                        raise Exception("Vector \"{}\" already in space".format(name))
                    else:
                        continue
                vector = Vector(name=name, size=self.size)
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
                vector = self.get(name)
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
            dist = self.get(v).dist(vector)
            if dist >= threshold:
                distances[v] = dist
                if distances[v] > distance:
                    best = v
                    distance = distances[v]
        return distances, best
