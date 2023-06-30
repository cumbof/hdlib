"""Implementation of hyperdimensional Vector and Space."""

import errno
import os
import pickle
import uuid
from typing import List, Optional, Set, Tuple, Union

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
            The size of the vector. It is 10,000 by default and cannot be less than that.
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
        TypeError
            - if the vector name is not instance of a primitive;
            - if `tags` is not an instance of set;
            - if `vector` is not an instance of numpy.ndarray;
            - if `size` is not an integer number.
        ValueError
            - if `vtype` is different than 'binary' or 'bipolar';
            - if `size` is lower than 10,000.
        FileNotFoundError
            If `from_file` is not None but the file does not exist.

        Examples
        --------
        >>> from hdlib.space import Vector
        >>> vector = Vector()
        >>> type(vector)
        <class 'hdlib.space.Vector'>

        A new bipolar vector with a size of 10,000 is created by default.

        >>> vector = Vector(size=10)
        ValueError: Vector size must be greater than or equal to 10000

        This throws a ValueError since the vector size cannot be less than 10,000.

        >>> vector1 = Vector()
        >>> vector1.dump(to_file='~/my_vector.pkl')
        >>> vector2 = Vector(from_file='~/my_vector.pkl')
        >>> type(vector2)
        <class 'hdlib.space.Vector'>

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

        elif from_file:
            if not os.path.isfile(from_file):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), from_file)

            else:
                # Load vector from pickle file
                with open(from_file, "rb") as pkl:
                    self.version, self.name, self.size, self.vector, self.vtype, self.parents, self.children, self.tags, self.seed = pickle.load(pkl)

                if self.version != __version__:
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
        >>> from hdlib.space import Vector
        >>> vector = Vector()
        >>> len(vector)
        10000

        Return the vector size, which is 10,000 by default here
        """

        return self.size

    def __str__(self) -> None:
        """Print the Vector object properties.

        Returns
        -------
        str
            A description of the Vector object. It reports the name, seed, size,
            vector type, tags, and the actual vector.

        Examples
        --------
        >>> from hdlib.space import Vector
        >>> vector = Vector()
        >>> print(vector)

                Class:   hdlib.space.Vector
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

        return """
            Class:   hdlib.space.Vector
            Version: {}
            Name:    {}
            Seed:    {}
            Size:    {}
            Type:    {}
            Tags:
            
            {}
            
            Vector:

            {}
        """.format(
            self.version,
            self.name,
            self.seed,
            self.size,
            self.vtype,
            np.array(list(self.tags)),
            self.vector
        )

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
        >>> from hdlib.space import Vector
        >>> vector1 = Vector(seed=1)
        >>> vector2 = Vector(seed=2)
        >>> vector1.dist(vector2, method='cosine')
        0.0034

        Generate two random bipolar vectors and compute the cosine similarity between them.
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
        """Normalize a vector after a binding or bundling with another vector.

        Raises
        ------
        Exception
            If the vector type is not supported (i.e., is different from binary and bipolar).

        Examples
        --------
        >>> from hdlib.space import Vector
        >>> from hdlib.arithmetic import bind
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector3 = bind(vector1, vector2)
        >>> vector3.normalize()
        >>> ((vector3.vector == 0) | (vector3.vector == 1)).all()
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
        >>> from hdlib.space import Vector
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
        >>> from hdlib.space import Vector
        >>> vector1 = Vector()
        >>> vector2 = Vector()
        >>> vector1.bundle(vector2)

        It overrides the actual vector content of `vector1` with the result of the bundling with `vector2`.
        Refers to hdlib.arithmetic.bundle for additional information.
        """

        # Import arithmetic.bundle here to avoid circular imports
        from hdlib.arithmetic import bundle as bundle_operator

        self.__override_object(bundle_operator(self, vector))
    
    def permute(self, rotate_by: int=1) -> None:
        """Permute the current vector inplace.

        Parameters
        ----------
        rotate_by : int
            Rotate the input vector by `rotate_by` positions (the default is 1).

        Examples
        --------
        >>> from hdlib.space import Vector
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

        self.parent = vector.parent
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
        >>> from hdlib.space import Vector
        >>> vector = Vector()
        >>> vector.dump(to_file='~/my_vector.pkl')
        >>> os.path.isfile('~/my_vector.pkl')
        True

        Create a Vector object and dump it to a pickle file under the home directory.
        """

        if not to_file:
            # Dump the vector to a pickle file in the current working directory if not file path is provided
            to_file = os.path.join(os.getcwd, "{}.pkl".format(self.name))

        if os.path.isfile(to_file):
            raise Exception("The output file already exists!\n{}".format(to_file))

        with open(to_file, "wb") as pkl:
            pickle.dump((self.version, self.name, self.size, self.vector, self.vtype, self.parents, self.children, self.tags, self.seed), pkl)


class Space(object):
    """Vectors space."""

    def __init__(self, size: int=10000, vtype: str="bipolar", from_file: Optional[os.path.abspath]=None) -> "Space":
        """Initialize the vectors space as a dictionary of Vector objects.

        Parameters
        ----------
        size : int, optional, default 10000
            Size of vectors in the space. It is 10,000 by default and cannot be less than that.
        vtype : {'binary', 'bipolar'}, default 'bipolar'
            The type of vectors in space.
        from_file : str, default None
            Path to a pickle file. Used to load a Space object from file.

        Returns
        -------
        Space
            A new Space object.

        Raises
        ------
        ValueError
            - if `vtype` is different than 'binary' or 'bipolar';
            - if `size` is lower than 10,000.
        FileNotFoundError
            If `from_file` is not None but the file does not exist.

        Examples
        --------
        >>> from hdlib.space import Space
        >>> space = Space()
        <class 'hdlib.space.Space'>

        Create a Space object that can host bipolar vectors with a size of 10,000 by default.

        >>> Space(size=10)
        ValueError: Size of vectors in space must be greater than or equal to 10000

        This throws a ValueError since the vector size cannot be less than 10,000.

        >>> space1 = Space()
        >>> space1.dump(to_file='~/my_space.pkl')
        >>> space2 = Space(from_file='~/my_space.pkl')
        >>> type(space2)
        <class 'hdlib.space.Space'>

        This creates an empty space `space1`, dumps the object to a pickle file under the home directory,
        and finally create a new space object `space2` from the pickle file.
        """

        self.space = dict()

        self.version = __version__

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

        if from_file:
            if not os.path.isfile(from_file):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), from_file)

            else:
                with open(from_file, "rb") as pkl:
                    self.version, self.size, self.vtype, self.space, self.root = pickle.load(pkl)

                if self.version != __version__:
                    print("Warning: the specified Space has been created with a different version of hdlib")

                for name in self.space:
                    if self.space[name].tags:
                        for tag in self.space[name].tags:
                            if tag not in self.tags:
                                self.tags[tag] = set()

                            self.tags[tag].add(name)

    def __len__(self) -> int:
        """Get the number of vectors in space.

        Returns
        -------
        int
            The number of vectors in space.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector = Vector()
        >>> space.insert(vector)
        >>> len(space)
        1

        Create a Space object, add a Vector object to the space, and check the total number
        of Vector objects in the space.
        """

        return len(self.space)

    def __str__(self) -> None:
        """Print the Space object properties.

        Returns
        -------
        str
            A description of the Space object. It reports the size, vector type,
            the number of vectors in space, the set of vectors tags, and the set of vectors names.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector = Vector(name='my_vector')
        >>> space.insert(vector)
        >>> print(space)

                Class:   hdlib.space.Space
                Size:    10000
                Type:    bipolar
                Vectors: 1
                Tags:

                []

                IDs:

                ['my_vector']

        Print the Space object properties. It contains only one vector.
        The vector size and type are 10,000 and 'bipolar' by default.
        """

        return """
            Class:   hdlib.space.Space
            Version: {}
            Size:    {}
            Type:    {}
            Vectors: {}
            Tags:
            
            {}
            
            IDs:

            {}
        """.format(
            self.version,
            self.size,
            self.vtype,
            len(self.space),
            np.array(list(self.tags.keys())),
            np.array(list(self.space.keys()))
        )

    def memory(self) -> List[str]:
        """Return names or IDs of vectors in space.

        Returns
        -------
        list
            A list with vectors names or IDs

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector = Vector(name='my_vector')
        >>> space.insert(vector)
        >>> space.memory()
        ['my_vector']

        Create a Space and add a Vector called 'my_vector'. The memory function returns
        the list of vector names. In this case a list with one element only 'my_vector'.
        """

        return list(self.space.keys())

    def get(
        self,
        names: Optional[List[str]]=None,
        tags: Optional[List[Union[str, int, float]]]=None
    ) -> List[Vector]:
        """Get vectors by names or tags.

        Parameters
        ----------
        names : list, optional
            A list with vector names. It is required in case no tags are specified.
        tags : list, optional
            A list with vector tags. It is required in case no names are specified.

        Returns
        -------
        list
            A list of Vector objects in the space according to the specified names or tags.

        Raises
        ------
        TypeError
            If names or tags in the input lists are not instance of primitives.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector1 = Vector(name='my_vector_1', tags={'tag1', 'tag2'})
        >>> vector2 = Vector(name='my_vector_2', tags={'tag2', 'tag3', 'tag4'})
        >>> space.insert(vector1)
        >>> space.insert(vector2)
        >>> vectors = space.get(tags=['tag2'])
        >>> for vector in vectors:
        ...     print(vector.name)
        my_vector_1
        my_vector_2

        This creates two Vector objects with a few tags and add them to a Space.
        It then retrieves a list of vectors by searching for a specific tag which is in common between
        the two vectors in this case. It finally prints the vector names.
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
        """Add a Vector object to the space.

        Parameters
        ----------
        vector : Vector
            The input Vector object that must be added to the Space

        Raises
        ------
        Exception
            - if the vector size or type is not compatible with the space;
            - if a vector with the same name of the input one is already in the space.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> vector = Vector()
        >>> space = Space()
        >>> space.insert(vector)

        It creates a random bipolar vector with size 10,000 and adds it to a space that by default can host 
        bipolar vectors with size 10,000.

        >>> vector = Vector(size=15000)
        >>> space = Space()
        >>> space.insert(vector)
        Exception: Space and vectors with different size are not compatible

        By default, the space can host bipolar vectors with size 10,000, while here we explicitly created a 
        Vector object with size 15,000 which is not compatible with the space.
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

    def bulk_insert(
        self,
        names: List[str],
        tags: Optional[List[List[Union[str, int, float]]]]=None,
        ignore_existing: bool=True
    ) -> None:
        """Add vectors to the space in bulk.

        Parameters
        ----------
        names : list
            A list with vector names.
        tags : list, optional
            An optional list of lists with vector tags.
        ignore_existing : bool, default True
            Do not raise an exception in case the space contains a vector with the same name specified in `names`.

        Raises
        ------
        TypeError
            - if `names` or `tags` are not instance of list;
            - if the elements of the `names` list are not instance of a primitive.
        Exception
            - if the number of elements in `names` doesn't match with the number of elements in `tags`;
            - if there already a vector in the space with the same name in `names`.

        Examples
        --------
        >>> from hdlib.space import Space
        >>> space = Space()
        >>> space.bulk_insert(names=['my_vector_1', 'my_vector_2'])
        >>> space.memory()
        ['my_vector_1', 'my_vector_2']

        Create two random bipolar vectors with size 10,000 just by specifying a list with vector names.
        The vector type and size is inherited by the space that by default can host bipolar vectors with size 10,000.

        >>> space.bulk_insert(names=['my_vector_3', 'my_vector_4'], tags=[['tag1'], ['tag1', 'tag2']])
        >>> vectors = space.get(tags=['tag1'])
        >>> for vector in vectors:
        ...     print(vector.name)
        my_vector_3
        my_vector_4

        Add other two vectors and assigned them a few tags, then retrieve the vectors with tag 'tag1'.
        Both 'my_vector_3' and 'my_vector_4' contain 'tag1' in their set of tags.
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
                if not ignore_existing:
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
        """Remove a vector from the space by its name.

        Parameters
        ----------
        name : str
            The name of the vector that must be removed from the space.

        Returns
        -------
        Vector
            Returns the Vector object.

        Raises
        ------
        TypeError
            If the vector name is not an instance of a primitive.
        Exception
            If there is not a vector with that specific name in the space.

        Examples
        --------
        >>> form hdlib.space import Space, Vector
        >>> vector = Vector(name='my_vector')
        >>> space = Space()
        >>> space.insert(vector)
        >>> space.remove('my_vector')
        >>> len(space)
        0

        Create a vector called 'my_vector', add it to the space and then remove it.
        Finally check how many vectors are in the space.
        """

        try:
            name = str(name)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if name not in self.space:
            raise Exception("Vector not in space")

        vector = self.space[name]

        del self.space[name]

        for tag in vector.tags:
            self.tags[tag].remove(vector.name)

            if not self.tags[tag]:
                del self.tags[tag]

        return vector

    def add_tag(self, name: str, tag: Union[str, int, float]) -> None:
        """Tag a vector.

        Parameters
        ----------
        name : str
            The vector name or ID.
        tag : str, int, float
            The tag must be a primitive.

        Raises
        ------
        TypeError
            If the name or tag are not instance of primitives.
        Exception
            If there is not a vector in the space with that specific name or ID.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> my_vector = Vector(name='my_vector')
        >>> space.insert(my_vector)
        >>> space.add_tag('my_vector', 'tag')
        >>> for vector in space.get(tags['tag']):
        ...     print(vector.name)
        my_vector

        This creates a Vector object add it to a Space. It then assigns a tag to the vector and searches
        for vector with that specific tag within the space. It finally prints the vector names.
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
        """Untag a vector.

        Parameters
        ----------
        name : str
            The vector name or ID.
        tag : str, int, float
            The tag must be a primitive.

        Raises
        ------
        TypeError
            If the name or tag are not instance of primitives.
        Exception
            If there is not a vector in the space with that specific name or ID.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> my_vector = Vector(name='my_vector', tags={'tag'})
        >>> space.insert(my_vector)
        >>> space.remove_tag('my_vector', 'tag')
        >>> len(space.get(tags['tag']))
        0

        This initializes a space, inserts a vector with a tag into the space, then untags the vector, and
        finally searches for vectors with that specific tag. No vectors are returned since there was only
        one vector with that tag that has been untagged.
        """

        try:
            name = str(name)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if name not in self.space:
            raise Exception("Vector not in space")

        if not isinstance(tag, str) and not isinstance(tag, int) and not isinstance(tag, float):
            raise TypeError("Tags must be string, integer, or float")

        if tag in self.tags:
            self.space[name].tags.remove(tag)

            self.tags[tag].remove(name)

            if not self.tags[tag]:
                del self.tags[tag]

    def link(self, name1: str, name2: str) -> None:
        """Link two vectors in the space through by their names. Links are directed edges.

        Parameters
        ----------
        name1 : str
            Name or ID of the first vector.
        name2 : str
            Name or ID of the second vector.

        Raises
        ------
        TypeError
            If vectors names are not instance of a primitive.
        Exception
            If there are no vectors in space named `name1` and `name2`.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector1 = Vector(name='vector1')
        >>> vector2 = Vector(name='vector2')
        >>> space.insert(vector1)
        >>> space.insert(vector2)
        >>> space.link('vector2', 'vector1')
        >>> vector2 = space.get(names=['vector2'])[0]
        >>> 'vector1' in vector2.children
        True

        Define a space with two vectors 'vector1' and 'vector2'. Link 'vector2' with 'vector1'.
        Retrieve 'vector2' from the space and check whether 'vector1' is in its set of linked nodes.
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
        """Vector links can be used to define a tree structure. Set a specific vector as root.

        Parameters
        ----------
        name : str
            Name or ID of vector in space.

        Raises
        ------
        TypeError
            If the vector name or ID is not instance of a primitive.
        Exception
            If there are no vectors in the space with the specified name.

        Examples
        --------
        >>> from hdlib.space import Space
        >>> space = Space()
        >>> space.bulk_insert(names=['vector1', 'vector2', 'vector3'])
        >>> space.link('vector1', 'vector2')
        >>> space.link('vector1', 'vector3')
        >>> space.set_root('vector1')
        >>> vector1 = space.get(names=['vector1'])[0]
        >>> for vector in vector1.children:
        ...     print(vector)
        vector2
        vector3

        Create a space and add three vectors in bulk. Link 'vector1' to 'vector2' and 'vector3', and
        set 'vector1' as root. Finally, print the name of the nodes linked to the root.
        """

        try:
            name = str(name)

        except:
            raise TypeError("Vector name must be instance of a primitive")

        if name not in self.space:
            raise Exception("Vector \"{}\" not in space".format(name))

        self.root = name

    def find(self, vector: Vector, threshold: float=-1.0, method: str="cosine") -> Tuple[str, float]:
        """Search for the closest vector in space.

        Parameters
        ----------
        vector : Vector
            Input Vector object. Search for the closest vector to this Vector in the space.
        threshold : float, default -1.0
            Threshold on distance/similarity.
        method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Distance/similarity measure.

        Returns
        -------
        tuple
            A tuple with the name of the closest vector in space and its distance/similarity with the input vector.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector1 = Vector(name='vector1')
        >>> vector2 = Vector(name='vector2')
        >>> vector3 = Vector(name='vector3')
        >>> space.insert(vector1)
        >>> space.insert(vector2)
        >>> space.insert(vector3)
        >>> space.find(vector1)
        ('vector1', 1.0)

        Create a space with three vectors 'vector1', 'vector2', and 'vector3', and search for the closest vector to 'vector1'.
        The result is obviously itself, 'vector1', with a cosine similarity of 1.0.
        """

        # Exploit self.find_all() to seach for the best match
        # It will take care of raising exceptions in case of problems with input arguments
        distances, best = self.find_all(vector, threshold=threshold, method=method)

        return best, distances[best]

    def find_all(self, vector: Vector, threshold: float=-1.0, method: str="cosine") -> Tuple[dict, str]:
        """Compute distance of the input vector against all vectors in space.

        Parameters
        ----------
        vector : Vector
            Input Vector object. Search for the closest vector to this Vector in the space.
        threshold : float, default -1.0
            Threshold on distance/similarity.
        method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Distance/similarity measure.

        Returns
        -------
        dict
            A dictionary the distances/similarities between the input vector and all the other vectors in the space,
            in addition to the name of the closest vector.

        Raises
        ------
        ValueError
            If the threshold is lower than -1.0 or higher than 1.0.
        Exception
            If the size of the input vector is not compatible with the size of vectors in the space.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector1 = Vector(name='vector1', seed=1)
        >>> vector2 = Vector(name='vector2', seed=2)
        >>> vector3 = Vector(name='vector3', seed=3)
        >>> space.insert(vector1)
        >>> space.insert(vector2)
        >>> space.insert(vector3)
        >>> space.find_all(vector1)
        ({'vector1': 1.0, 'vector2': 0.004, 'vector3': 0.015}, 'vector1')

        Create a space with three vectors 'vector1', 'vector2', and 'vector3', and compute the cosine similarity between 'vector1'
        and all the other vectors in space (including itseld). The closest vector is obviously itself, 'vector1', with a cosine 
        similarity of 1.0. Use a seed for reproducing the same distances.
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
        """Dump the Space object to a pickle file.

        Parameters
        ----------
        to_file
            Path to the file used to dump the Space object to.

        Raises
        ------
        Exception
            If the `to_file` file already exists.

        Examples
        --------
        >>> import os
        >>> from hdlib.space import Space
        >>> space = Space()
        >>> space.dump(to_file='~/my_space.pkl')
        >>> os.path.isfile('~/my_space.pkl')
        True

        Create a Space object and dump it to a pickle file under the home directory.
        """

        if not to_file:
            # Dump the space to a pickle file in the current working directory if not file path is provided
            to_file = os.path.join(os.getcwd, "space.pkl")

        if os.path.isfile(to_file):
            raise Exception("The output file already exists!\n{}".format(to_file))

        with open(to_file, "wb") as pkl:
            pickle.dump((self.version, self.size, self.vtype, self.space, self.root), pkl)
