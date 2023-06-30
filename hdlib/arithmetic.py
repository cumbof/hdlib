"""Implementation of the arithmetic operators."""

import numpy as np

from hdlib.space import Vector


def bind(vector1: Vector, vector2: Vector) -> Vector:
    """Bind vectors.

    Parameters
    ----------
    vector1 : Vector
        The first vector object.
    vector2: Vector
        The second vector object.

    Returns
    -------
    Vector
        A new vector object as the result of the bind operator on the two input vectors.

    Raises
    ------
    Exception
        If vectors have different sizes or different vector types.

    Notes
    -----
    The bind operator has the following properties:
    - invertible (unbind);
    - it distributes over bundling;
    - it preserves the distance;
    - the resulting vector is dissimilar to the input vectors.

    Examples
    --------
    >>> from hdlib.space import Vector
    >>> from hdlib.arithmetic import bind
    >>> vector1 = Vector(size=10000, vtype="binary")
    >>> vector2 = Vector(size=10000, vtype="binary")
    >>> vector3 = bind(vector1, vector2)
    >>> type(vector3)
    <class 'hdlib.space.Vector'>

    The bind function returns a new Vector object whose content is computed as the element-wise 
    multiplication of the two input vectors.

    >>> vector1 = Vector(size=10000, vtype="binary")
    >>> vector2 = Vector(size=10000, vtype="bipolar")
    >>> vector3 = bind(vector1, vector2)
    Exception: Vector types are not compatible

    The vector type of the two input vector is different and thus the binding cannot be performed.

    >>> vector1 = Vector(size=10000, vtype="bipolar")
    >>> vector2 = Vector(size=15000, vtype="bipolar")
    >>> vector3 = bind(vector1, vector2)
    Exception: Vectors must have the same size

    It also throws an exception in case the size of the two input vectors is not the same.
    """

    if vector1.size != vector2.size:
        raise Exception("Vectors must have the same size")

    if vector1.vtype != vector2.vtype:
        raise Exception("Vector types are not compatible")

    vector = vector1.vector * vector2.vector

    tags = set(vector1.tags).union(set(vector2.tags))

    return Vector(size=vector1.size, vector=vector, tags=tags, vtype=vector1.vtype, seed=vector1.seed)


def bundle(vector1: Vector, vector2: Vector) -> Vector:
    """Bundle vectors.

    Parameters
    ----------
    vector1 : Vector
        The first vector object.
    vector2: Vector
        The second vector object.

    Returns
    -------
    Vector
        A new vector object as the result of the bundle operator on the two input vectors.

    Raises
    ------
    Exception
        If vectors have different sizes or different vector types.

    Notes
    -----
    The bundle operator has the following properties:
    - the resulting vector is similar to the input vectors;
    - the more vectors are involved in bundling, the harder it is to determine the component vectors;
    - if several copies of any vector are included in bundling, the resulting vector is closer to the 
      dominant vector than to the other components.

    Examples
    --------
    >>> from hdlib.space import Vector
    >>> from hdlib.arithmetic import bundle
    >>> vector1 = Vector(size=10000, vtype="binary")
    >>> vector2 = Vector(size=10000, vtype="binary")
    >>> vector3 = budle(vector1, vector2)
    >>> type(vector3)
    <class 'hdlib.space.Vector'>

    The bundle function returns a new Vector object whose content is computed as the element-wise sum 
    of the two input vectors.

    >>> vector1 = Vector(size=10000, vtype="binary")
    >>> vector2 = Vector(size=10000, vtype="bipolar")
    >>> vector3 = budle(vector1, vector2)
    Exception: Vector types are not compatible

    The vector type of the two input vector is different and thus the bundling cannot be performed.

    >>> vector1 = Vector(size=10000, vtype="bipolar")
    >>> vector2 = Vector(size=15000, vtype="bipolar")
    >>> vector3 = budle(vector1, vector2)
    Exception: Vectors must have the same size

    It also throws an exception in case the size of the two input vectors is not the same.
    """

    if vector1.size != vector2.size:
        raise Exception("Vectors must have the same size")

    if vector1.vtype != vector2.vtype:
        raise Exception("Vector types are not compatible")

    vector = vector1.vector + vector2.vector

    tags = set(vector1.tags).union(set(vector2.tags))

    return Vector(size=vector1.size, vector=vector, tags=tags, vtype=vector1.vtype, seed=vector1.seed)


def permute(vector: Vector, rotate_by: int=1) -> Vector:
    """Permute a vector

    Parameters
    ----------
    vector : Vector
        The input vector object.
    rotate_by: int
        Rotate the input vector by `rotate_by` positions (the default is 1).

    Returns
    -------
    Vector
        A new vector object as the result of the permute operator on the input vector.

    Notes
    -----
    The permute operator has the following properties:
    - invertible;
    - it distributes over bundling and any elementwise operation;
    - it preserves the distance;
    - tThe resulting vector is dissimilar to the input vectors.

    Examples
    --------
    >>> from hdlib.space import Vector
    >>> from hdlib.arithmetic import permute
    >>> vector1 = Vector(size=10000, vtype="binary")
    >>> vector2 = permute(vector1, rotate_by=2)
    >>> type(vector2)
    <class 'hdlib.space.Vector'>

    The permute function returns a new Vector object whose content is the same of the input 
    vector rotated by 2 positions.
    """

    rolled = np.roll(vector.vector, rotate_by, axis=0)

    return Vector(size=vector.size, vector=rolled, tags=vector.tags, vtype=vector.vtype, seed=vector.seed)
