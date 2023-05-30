"""
Implementation of the arithmetic operators
"""

import numpy as np

from hdlib.space import Vector


def bundle(vector1: Vector, vector2: Vector) -> Vector:
    """
    Bundle vectors

    Properties:
    > The resulting vector is similar to the input vectors
    > The more vectors are involved in bundling, the harder it is to determine the component vectors
    > If several copies of any vector are included in bundling, the resulting vector is closer to the dominant vector than to the other component

    :param vector1:     Vector object
    :param vector2:     Vector object
    :return:            Vector resulted from bundling vector1 and vector2
    """

    if not isinstance(vector1, Vector) or not isinstance(vector2, Vector):
        raise TypeError("Input is not a valid Vector object")

    if vector1.size != vector2.size:
        raise Exception("Vectors must have the same size")

    vector = vector1.vector + vector2.vector
    return Vector(size=vector1.size, vector=vector, seed=vector1.seed)

def bind(vector1: Vector, vector2: Vector) -> Vector:
    """
    Bind vectors

    Properties:
    > Invertible (unbind)
    > It distributes over bundling
    > It preserves distance
    > The resulting vector is dissimilar to the input vectors

    :param vector1:     Vector object
    :param vector2:     Vector object
    :return:            Vector resulted from binding vector1 and vector2
    """

    if not isinstance(vector1, Vector) or not isinstance(vector2, Vector):
        raise TypeError("Input is not a valid Vector object")

    if vector1.size != vector2.size:
        raise Exception("Vectors must have the same size")

    vector = vector1.vector * vector2.vector
    return Vector(size=vector1.size, vector=vector, seed=vector1.seed)

def permute(vector: Vector, rotateby: int=1) -> Vector:
    """
    Permute vector

    Properties:
    > Invertible
    > It distribute over bundling and any elementwise operation
    > It preserves distance
    > The resulting vector is dissimilar to the input vectors

    :param vector:      Vector object
    :param rotateby:    How many rotations
    :return:            Input vector rotated "rotateby" times
    """

    if not isinstance(vector, Vector):
        raise TypeError("Input is not a valid Vector object")

    rolled = np.roll(vector.vector, rotateby, axis=0)
    return Vector(size=vector.size, vector=rolled, seed=vector.seed)
