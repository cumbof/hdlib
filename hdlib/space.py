"""Implementation of hyperdimensional Space.

__hdlib__ provides the _Space_ class under _hdlib.space_ for building the abstract representation of 
a hyperdimensional space which acts as a container for a multitude of vectors."""

import errno
import os
import pickle
from collections import OrderedDict
from typing import List, Optional, Set, Tuple, Union

import numpy as np

from hdlib import __version__, Vector


class Space(object):
    """Vectors space."""

    def __init__(self, size: int=10000, vtype: str="bipolar", from_file: Optional[os.path.abspath]=None) -> "Space":
        """Initialize the vectors space as a dictionary of Vector objects.

        Parameters
        ----------
        size : int, optional, default 10000
            Size of vectors in the space.
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
        Exception
            If the pickle object in `from_file` is not instance of Space.
        FileNotFoundError
            If `from_file` is not None but the file does not exist.
        ValueError
            - if `vtype` is different than 'binary' or 'bipolar';
            - if `size` is lower than 1,000.

        Examples
        --------
        >>> from hdlib.space import Space
        >>> space = Space()
        <class 'hdlib.space.Space'>

        Create a Space object that can host bipolar vectors with a size of 10,000 by default.

        >>> Space(size=10)
        ValueError: Size of vectors in space must be greater than or equal to 1000

        This throws a ValueError since the vector size cannot be less than 1,000.

        >>> space1 = Space()
        >>> space1.dump(to_file='~/my_space.pkl')
        >>> space2 = Space(from_file='~/my_space.pkl')
        >>> type(space2)
        <class 'hdlib.space.Space'>

        This creates an empty space `space1`, dumps the object to a pickle file under the home directory,
        and finally create a new space object `space2` from the pickle file.
        """

        # We may want to iterate over the Space object
        # Thus, we need to maintain the order of the vectors into the space dictionary
        self.space = OrderedDict()

        # Used to iterate over vectors in the space
        self._vector_index = 0

        self.version = __version__

        self.size = size

        if self.size < 1000:
            raise ValueError("Size of vectors in space must be greater than or equal to 1000")

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
                    from_file_obj = pickle.load(pkl)

                    if not isinstance(from_file_obj, type(self)):
                        raise Exception("Pickle object is not instance of {}".format(type(self)))

                    self.__dict__.update(from_file_obj.__dict__)

                    if self.version != __version__:
                        print("Warning: the specified Space has been created with a different version of hdlib")

    def __iter__(self) -> "Space":
        """Required to make the Space object iterable."""

        return self

    def __next__(self) -> str:
        """Used to iterate over the vector objects into the Space.

        Returns
        -------
        str
            The vector name at a specific position.
        """

        if self._vector_index >= len(self.space):
            # Set the vector index back to the first position.
            # Redy to start iterating again over the vectors in the space
            self._vector_index = 0

            raise StopIteration

        else:
            # Retrieve the vector name at a specific position in the space
            # Vectors are all ordered in the space since the space is defined as an OrderedDict
            vector = self.memory()[self._vector_index]

            # Increment the vector index for the next iteration
            self._vector_index += 1

            # This returns the vector name or ID
            # It is enough, since the space is a hashmap and we can retrieve the Vector object in O(1)
            return vector

    def __contains__(self, vector: str) -> bool:
        """Check whether a vector is in the space.

        Parameters
        ----------
        vector : str
            The vector name or ID.

        Returns
        -------
        bool
            True if `vector` is in the space, False otherwise.

        Examples
        --------
        >>> from hdlib.space import Space, Vector
        >>> space = Space()
        >>> vector = Vector(name="my_vector")
        >>> space.insert(vector)
        >>> "my_vector" in space
        True

        Create a Space object, add a Vector object into the space, and check whether the
        vector is actually in the space by searching for its name.
        """

        return True if vector in self.space else False

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

        Create a Space object, add a Vector object into the space, and check the total number
        of Vector objects in the space.
        """

        return len(self.space)

    def __str__(self) -> str:
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
                Version: 0.1.17
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
        Exception
            - if no `names` or `tags` are provided in input;
            - if both `names` and `tags` are provided in input.
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
            raise Exception("No names or tags provided")

        if names and tags:
            raise Exception("Cannot search for vectors by their names and tags at the same time")

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
                    raise TypeError("A tags must be string, integer, or float")

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
        ignore_existing: bool=False
    ) -> None:
        """Add vectors to the space in bulk.

        Parameters
        ----------
        names : list
            A list with vector names.
        tags : list, optional
            An optional list of lists with vector tags.
        ignore_existing : bool, default False
            If True, do not raise an exception in case the space contains a vector with the same name specified in `names`.

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

        names = set(names)

        for pos, name in enumerate(names):
            if not isinstance(name, (bool, str, int, float, None)):
                raise TypeError("Entries in input list must be instances of primitives")

            name = str(name)

            if name in self.space:
                if not ignore_existing:
                    raise Exception("Vector \"{}\" already exists in the space".format(name))

                else:
                    continue

            vector_tags = set(tags[pos]) if tags else set()

            vector = Vector(name=name, size=self.size, tags=vector_tags, vtype=self.vtype)

            self.insert(vector)

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

    def find(self, vector: Vector, threshold: float=np.inf, method: str="cosine") -> Tuple[str, float]:
        """Search for the closest vector in space.

        Parameters
        ----------
        vector : Vector
            Input Vector object. Search for the closest vector to this Vector in the space.
        threshold : float, default numpy.Inf
            Threshold on distance between vectors.
        method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Distance metric.

        Returns
        -------
        tuple
            A tuple with the name of the closest vector in space and its distance with the input vector.

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
        ('vector1', 0.0)

        Create a space with three vectors 'vector1', 'vector2', and 'vector3', and search for the closest vector to 'vector1'.
        The result is obviously itself, 'vector1', with a cosine distance of 0.0.
        """

        # Exploit self.find_all() to seach for the best match
        # It will take care of raising exceptions in case of problems with input arguments
        distances, best = self.find_all(vector, threshold=threshold, method=method)

        return best, distances[best]

    def find_all(self, vector: Vector, threshold: float=np.inf, method: str="cosine") -> Tuple[dict, str]:
        """Compute distance of the input vector against all vectors in space.

        Parameters
        ----------
        vector : Vector
            Input Vector object. Search for the closest vector to this Vector in the space.
        threshold : float, default numpy.Inf
            Threshold on distance between vectors.
        method : {'cosine', 'euclidean', 'hamming'}, default 'cosine'
            Distance metric.

        Returns
        -------
        dict
            A dictionary the distances between the input vector and all the other vectors in the space,
            in addition to the name of the closest vector.

        Raises
        ------
        ValueError
            If the threshold is lower than 0.0.
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
        ({'vector1': 0.0, 'vector2': 0.996, 'vector3': 0.985}, 'vector1')

        Create a space with three vectors 'vector1', 'vector2', and 'vector3', and compute the cosine distance between 'vector1'
        and all the other vectors in space (including itseld). The closest vector is obviously itself, 'vector1', with a cosine 
        distance of 0.0. Use a seed for reproducing the same distances.
        """

        if self.size != vector.size:
            raise Exception("Space and vectors with different size are not compatible")

        if threshold < 0.0:
            raise ValueError("Threshold cannot be lower than 0.0")

        distances = dict()

        distance = np.inf

        best = None

        for v in self.space:
            # Compute distance
            dist = self.space[v].dist(vector, method=method)

            if dist <= threshold:
                distances[v] = dist

                if distances[v] < distance:
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
            to_file = os.path.join(os.getcwd(), "space.pkl")

        if os.path.isfile(to_file):
            raise Exception("The output file already exists!\n{}".format(to_file))

        with open(to_file, "wb") as pkl:
            pickle.dump(self, pkl)
