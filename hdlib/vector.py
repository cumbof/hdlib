"""Implementation of hyperdimensional Vector.

__hdlib__ provides the _Vector_ class under _hdlib.vector_ for building the abstract representation of hyperdimensional vectors."""

import errno
import os
import pickle
import uuid
from typing import Optional, Set, Union

import numpy as np

from hdlib import __version__


class Vector(object):
    """Vector object."""

    def __init__(
        self,
        name: Optional[str]=None,
        size: int=10000,
        vector: Optional[np.ndarray]=None,
        vtype: str="bipolar",
        tags: Optional[Set[Union[str, int, float]]]=None,
        seed: Optional[int]=None,
        warning: bool=False,
        from_file: Optional[os.path.abspath]=None,
    ) -> "Vector":
        """Initialize a Vector object.

        Parameters
        ----------
        name : str, optional
            The unique identifier of the Vector object. A random UUID v4 is generated if not specified.
        size : int, optional, default 10000
            The size of the vector. It is 10,000 by default.
        vector : numpy.ndarray, optional, default None
            The actual vector. A random vector is created if not specified.
        vtype : {'binary', 'bipolar'}, default 'bipolar'
            The vector type.
        tags : set, default None
            An optional set of vector tags. Tags can be str, int, and float.
        seed : int, default None
            An optional seed for reproducibly generating the vector numpy.ndarray randomly.
        warning : bool, default False
            Print warning messages if True.
        from_file : str, default None
            Path to a pickle file. Used to load a Vector object from file.

        Returns
        -------
        Vector
            A new Vector object.

        Raises
        ------
        Exception
            If the pickle object in `from_file` is not instance of Vector.
        FileNotFoundError
            If `from_file` is not None but the file does not exist.
        TypeError
            - if the vector name is not instance of a primitive;
            - if `tags` is not an instance of set;
            - if `vector` is not an instance of numpy.ndarray;
            - if `size` is not an integer number.
        ValueError
            If `vtype` is different than 'binary' or 'bipolar'.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector = Vector()
        >>> type(vector)
        <class 'hdlib.vector.Vector'>

        A new bipolar vector with a size of 1,000 is created by default.

        >>> vector1 = Vector()
        >>> vector1.dump(to_file='~/my_vector.pkl')
        >>> vector2 = Vector(from_file='~/my_vector.pkl')
        >>> type(vector2)
        <class 'hdlib.vector.Vector'>

        This creates a random bipolar vector `vector1`, dumps the object to a pickle file under the home directory,
        and finally create a new vector object `vector2` from the pickle file.
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

        # Take track of the hdlib version
        self.version = __version__

        if tags and not isinstance(tags, set):
            raise TypeError("Tags must be a set")

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

            self.vtype = vtype

        elif from_file:
            if not os.path.isfile(from_file):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), from_file)

            else:
                # Load vector from pickle file
                with open(from_file, "rb") as pkl:
                    from_file_obj = pickle.load(pkl)

                    if not isinstance(from_file_obj, type(self)):
                        raise Exception("Pickle object is not instance of {}".format(type(self)))

                    self.__dict__.update(from_file_obj.__dict__)

                    if self.version != __version__:
                        print("Warning: the specified Vector has been created with a different version of hdlib")

        else:
            if not isinstance(size, int):
                raise TypeError("Vector size must be an integer number")

            self.size = size

            if vtype not in ("bipolar", "binary"):
                raise ValueError("Vector type can be binary or bipolar only")

            # Add vector type
            self.vtype = vtype.lower()

            if seed is None:
                rand = np.random.default_rng()

            else:
                # Conditions on random seed for reproducibility
                # numpy allows integers as random seeds
                if not isinstance(seed, int):
                    raise TypeError("Seed must be an integer number")

                rand = np.random.default_rng(seed=self.seed)

            """
            # Use the following instructions to generate random vectors with real numbers
            self.vector = rand.uniform(low=-1.0, high=1.0, size=(self.size,))
            self.vector /= np.linalg.norm(self.vector)
            """

            # Build a random binary vector
            self.vector = rand.integers(2, size=size)

            if vtype == "bipolar":
                # Build a random bipolar vector
                self.vector = 2 * self.vector - 1

    def __len__(self) -> int:
        """Get the vector size.

        Returns
        -------
        int
            The vector size.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector = Vector()
        >>> len(vector)
        10000

        Return the vector size, which is 10,000 by default here
        """

        return self.size

    def __str__(self) -> str:
        """Print the Vector object properties.

        Returns
        -------
        str
            A description of the Vector object. It reports the name, seed, size,
            vector type, tags, and the actual vector.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector = Vector()
        >>> print(vector)

                Class:   hdlib.vector.Vector
                Version: 0.1.17
                Name:    89ea628b-3d29-47e1-9d10-34bdbfce8d40
                Seed:    None
                Size:    10000
                Type:    bipolar
                Tags:

                []

                Vector:

                [ 1 -1 -1 ... -1  1 -1]

        Print the Vector object properties. The name has been generated as a UUID v4, while
        the vector size and type are 10,000 and 'bipolar' by default. No tags have been specified.
        Thus, the set of vector tags is empty.
        """

        return f"""
            Class:   hdlib.vector.Vector
            Version: {self.version}
            Name:    {self.name}
            Seed:    {self.seed}
            Size:    {self.size}
            Type:    {self.vtype}
            Tags:

            {np.array(list(self.tags))}

            Vector:

            {self.vector}
        """

    def __add__(self, vector: "Vector") -> "Vector":
        """Implement the addition operator between two Vector objects as bundle.

        Returns
        -------
        Vector
            A new vector object as the result of the bundle operator on the two input vectors.

        Raises
        ------
        TypeError
            If the input `vector` is not instance of the Vector class.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector3 = vector1 + vector2
        >>> type(vector3)
        <class 'hdlib.vector.Vector'>

        The bundle function returns a new Vector object whose content is computed as the element-wise sum 
        of the two input vectors.
        """

        if not isinstance(vector, type(self)):
            raise TypeError("Cannot apply the bundle operator to non-Vector objects")

        # Import arithmetic.bundle here to avoid circular imports
        from hdlib.arithmetic import bundle as bundle_operator

        return bundle_operator(self, vector)

    def __sub__(self, vector: "Vector") -> "Vector":
        """Implement the subtraction operator between two Vector objects.

        Returns
        -------
        Vector
            A new vector object as the result of the subtraction operator on the two input vectors.

        Raises
        ------
        TypeError
            If the input `vector` is not instance of the Vector class.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector3 = vector1 - vector2
        >>> type(vector3)
        <class 'hdlib.vector.Vector'>

        The subtraction operation returns a new Vector object whose content is computed as the element-wise 
        subtraction of the two input vectors.
        """

        if not isinstance(vector, type(self)):
            raise TypeError("Cannot apply the subtraction operator to non-Vector objects")

        # Import arithmetic.bind here to avoid circular imports
        from hdlib.arithmetic import subtraction as subtraction_operator

        return subtraction_operator(self, vector)

    def __mul__(self, vector: "Vector") -> "Vector":
        """Implement the multiplication operator between two Vector objects as bind.

        Returns
        -------
        Vector
            A new vector object as the result of the bind operator on the two input vectors.

        Raises
        ------
        TypeError
            If the input `vector` is not instance of the Vector class.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector3 = vector1 * vector2
        >>> type(vector3)
        <class 'hdlib.vector.Vector'>

        The bind function returns a new Vector object whose content is computed as the element-wise 
        multiplication of the two input vectors.
        """

        if not isinstance(vector, type(self)):
            raise TypeError("Cannot apply the bind operator to non-Vector objects")

        # Import arithmetic.bind here to avoid circular imports
        from hdlib.arithmetic import bind as bind_operator

        return bind_operator(self, vector)

    def dist(self, vector: "Vector", method: str="cosine") -> float:
        """Compute distance between vectors.

        Parameters
        ----------
        vector : Vector
            A Vector object from which the distance must be computed.
        method : {'cosine', 'euclidean', 'hamming'}, optional, default 'cosine'
            The distance method.

        Returns
        -------
        float
            The distance between the current Vector object and the input `vector`.

        Raises
        ------
        Exception
            If the current vector has a different size or vector type than the input vector.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector1 = Vector(seed=1)
        >>> vector2 = Vector(seed=2)
        >>> vector1.dist(vector2, method='cosine')
        0.996

        Generate two random bipolar vectors and compute the distance between them.
        """

        if self.size != vector.size:
            raise Exception("Vectors must have the same size")

        if self.vtype != vector.vtype:
            raise Exception("Vectors must be of the same type")

        if method.lower() == "cosine":
            return 1 - np.dot(self.vector, vector.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(vector.vector))

        elif method.lower() == "hamming":
            return np.count_nonzero(self.vector != vector.vector)

        elif method.lower() == "euclidean":
            return np.linalg.norm(self.vector - vector.vector)

        else:
            raise ValueError("Distance method \"{}\" is not supported".format(method))

    def normalize(self) -> None:
        """Normalize a vector after a binding or bundling with another vector.

        Raises
        ------
        Exception
            If the vector type is not supported (i.e., is different from binary and bipolar).

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> from hdlib.arithmetic import bind
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector3 = bind(vector1, vector2)
        >>> vector3.normalize()
        >>> ((vector3.vector == -1) | (vector3.vector == 1)).all()
        True

        Binding or bundling two vectors can produce a new vector whose vtype is different from the
        one of the two input vector. This function normalizes the vector content in accordance to
        its vector type.
        """

        if self.vtype not in ("bipolar", "binary"):
            raise Exception("Vector type is not supported")

        self.vector[self.vector > 0] = 1

        self.vector[self.vector <= 0] = 0 if self.vtype == "binary" else -1

    def bind(self, vector: "Vector") -> None:
        """Bind the current vector with another vector object inplace.

        Parameters
        ----------
        vector : Vector
            The input Vector object.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector1.bind(vector2)

        It overrides the actual vector content of `vector1` with the result of the binding with `vector2`.
        Refers to hdlib.arithmetic.bind for additional information.
        """

        # Import arithmetic.bind here to avoid circular imports
        from hdlib.arithmetic import bind as bind_operator

        self.__override_object(bind_operator(self, vector))

    def bundle(self, vector: "Vector") -> None:
        """Bundle the current vector with another vector object inplace.

        Parameters
        ----------
        vector : Vector
            The input Vector object.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector1.bundle(vector2)

        It overrides the actual vector content of `vector1` with the result of the bundling with `vector2`.
        Refers to hdlib.arithmetic.bundle for additional information.
        """

        # Import arithmetic.bundle here to avoid circular imports
        from hdlib.arithmetic import bundle as bundle_operator

        self.__override_object(bundle_operator(self, vector))

    def subtraction(self, vector: "Vector") -> None:
        """Subtract a vector from the current vector object inplace.

        Parameters
        ----------
        vector : Vector
            The input Vector object.

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector1.subtract(vector2)

        It overrides the actual vector content of `vector1` with the result of the subtraction with `vector2`.
        Refers to hdlib.arithmetic.subtraction for additional information.
        """

        # Import arithmetic.subtraction here to avoid circular imports
        from hdlib.arithmetic import subtraction as subtraction_operator

        self.__override_object(subtraction_operator(self, vector))

    def permute(self, rotate_by: int=1) -> None:
        """Permute the current vector inplace.

        Parameters
        ----------
        rotate_by : int
            Rotate the input vector by `rotate_by` positions (the default is 1).

        Examples
        --------
        >>> from hdlib.vector import Vector
        >>> vector = Vector()
        >>> vector.permute(rotate_by=2)

        It overrides the actual vector content of `vector` with the result of applying the permute function inplace.
        Refers to hdlib.arithmetic.permute for additional information.
        """

        # Import arithmetic.permute here to avoid circular imports
        from hdlib.arithmetic import permute as permute_operator

        self.__override_object(permute_operator(self, rotate_by=rotate_by))

    def __override_object(self, vector: "Vector") -> None:
        """Override the Vector object with another Vector object. This is a private method.

        Parameters
        ----------
        vector : Vector
            The input vector from which properties are inherited to the current vector.
        """

        self.name = vector.name
        self.size = vector.size
        self.seed = vector.seed
        self.tags = vector.tags

        self.parents = vector.parents
        self.children = vector.children

        self.vtype = vector.vtype
        self.vector = vector.vector

        self.version = vector.version

    def dump(self, to_file: Optional[os.path.abspath]=None) -> None:
        """Dump the Vector object to a pickle file.

        Parameters
        ----------
        to_file
            Path to the file used to dump the Vector object to.

        Raises
        ------
        Exception
            If the `to_file` file already exists.

        Examples
        --------
        >>> import os
        >>> from hdlib.vector import Vector
        >>> vector = Vector()
        >>> vector.dump(to_file='~/my_vector.pkl')
        >>> os.path.isfile('~/my_vector.pkl')
        True

        Create a Vector object and dump it to a pickle file under the home directory.
        """

        if not to_file:
            # Dump the vector to a pickle file in the current working directory if not file path is provided
            to_file = os.path.join(os.getcwd(), "{}.pkl".format(self.name))

        if os.path.isfile(to_file):
            raise Exception("The output file already exists!\n{}".format(to_file))

        with open(to_file, "wb") as pkl:
            pickle.dump(self, pkl)
