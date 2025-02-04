"""Directed and undirected, weighted and unweighted graphs with hdlib.

It implements the __hdlib.graph.Graph__ class object which allows to represent directed and undirected, weighted and unweighted graphs
built according to the Hyperdimensional Computing (HDC) paradigm as described in _Poduval et al. 2022_ https://doi.org/10.3389/fnins.2022.757125."""

import copy
import itertools
import multiprocessing as mp
from typing import Optional, Set, Tuple, Union

import numpy as np

from hdlib import __version__
from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, permute

# Private graph vector name
GRAPH_ID = "__graph__"

# Private weight vector name prefix in case of weighted graph only
WEIGHT_ID = "__weight__"


class Graph(object):
    """Hyperdimensional Graph representation."""

    def __init__(
        self,
        size: int=10000,
        weights: int=100,
        directed: bool=False,
        weighted: bool=False,
        seed: Optional[int]=None
    ) -> "Graph":
        """Initialize a Graph object.

        Parameters
        ----------
        size : int, default 10000
            The size of vectors used to create a Space and define Vector objects.
        weights : int, default 100
            The number of weight level vectors.
        directed : bool, default False
            Directed or undirected.
        weighted : bool, default False
            Weighted or unweighted.
        seed : int, optional
            Random seed for reproducibility of results.

        Raises
        ------
        TypeError
            - If the vector size is not an integer number;
            - If the number of weight level vectors is not an integer.
        ValueError
            - If the vector size is lower than 10,000;
            - If the number of weight level vectors is lower than 2.

        Examples
        --------
        >>> from hdlib.graph import Graph
        >>> graph = Graph(size=10000, vtype='bipolar', directed=False, weighted=False)
        >>> type(graph)
        <class 'hdlib.graph.Graph'>

        This creates a new Graph object around a Space that can host random bipolar Vector objects with size 10,000.
        The represented graph is undirected and unweighted.
        """

        if not isinstance(size, int):
            raise TypeError("Vectors size must be an integer number")

        if size < 10000:
            raise ValueError("Vectors size must be greater than or equal to 10000")

        if not isinstance(weights, int):
            raise TypeError("The number of weight level vectors must be an integer")

        if weights < 2:
            raise ValueError("The number of weight level vectors must be greater than or equal to 2")

        # Register vectors dimensionality
        self.size = size

        # Register the number of weight level vectors
        self.weights = weights

        # Register vectors type
        self.vtype = "bipolar"

        # Register whether the graph is directed or undirected
        self.directed = directed

        # Register whether the graph is weighted or unweighted
        self.weighted = weighted

        # Keep track of the number of nodes
        self.nodes_counter = 0

        # Keep track of the number of edges
        self.edges_counter = 0

        # Hyperdimensional space
        self.space = Space(size=self.size, vtype=self.vtype)

        self.seed = seed

        if self.seed is None:
            self.rand = np.random.default_rng()

        else:
            # Conditions on random seed for reproducibility
            # numpy allows integers as random seeds
            if not isinstance(seed, int):
                raise TypeError("Seed must be an integer number")

            self.rand = np.random.default_rng(seed=self.seed)

        # Keep track of hdlib version
        self.version = __version__

    def __str__(self) -> None:
        """Print the Graph object properties.

        Returns
        -------
        str
            A description of the Graph object. It reports the vectors size, the vector type,
            the number of nodes, the number of edges, and whether it is directed or undirected,
            and weighted or unweighted.

        Examples
        --------
        >>> from hdlib.graph import Graph
        >>> graph = Graph()
        >>> print(graph)

                Class:    hdlib.graph.Graph
                Version:  0.1.17
                Size:     10000
                Type:     bipolar
                Directed: False
                Weighted: False
                Nodes:    0
                Edges:    0
                Seed:     None

        Print the MLModel object properties. By default, the size of vectors in space is 10,000,
        their types is bipolar, and the number of level vectors is 2. The number of data points 
        and the number of class labels are empty here since no dataset has been processed yet.
        """

        return """
            Class:    hdlib.graph.Graph
            Version:  {}
            Size:     {}
            Type:     {}
            Directed: {}
            Weighted: {}
            Nodes:    {}
            Edges:    {}
            Seed:     {}
        """.format(
            self.version,
            self.size,
            self.vtype,
            self.directed,
            self.weighted,
            self.nodes_counter,
            self.edges_counter,
            self.seed
        )

    def _add_edge(
        self,
        node1: str,
        node2: str,
        weight: Optional[float]=None
    ) -> None:
        """Add an edge to the graph and automatically build nodes if they do not exist in the space.

        Parameters
        ----------
        node1 : str
            Node name.
        node2 : str
            Node name.
        weight : float, optional
            The edge weight as a float between 0.0 and 1.0.

        Raises
        ------
        TypeError
            If `weight` is not a float.
        ValueError
            - if `node1` or `node2` is equals to `GRAPH_ID`;
            - if the graph is weighted but `weight` is None;
            - if `weight` is <0.0 or >1.0.
        """

        if node1 == GRAPH_ID or node2 == GRAPH_ID:
            raise ValueError("Node names cannot match with the private graph ID `{}`".format(GRAPH_ID))

        edge_exists = False

        if node1 in self.space.space and node2 in self.space.space:
            # Check whether an edge between node1 and node2 already exists
            if self.directed and node2 in self.space.space[node1].children:
                edge_exists = True

            elif not self.directed and node2 in self.space.space[node1].children and node1 in self.space.space[node2].children:
                edge_exists = True

        if not edge_exists:
            # Check whether the graph is weighted and the edge weight is a valid number
            # In case the graph is unweighted, ignore the weight
            if self.weighted:
                if weight is None:
                    raise ValueError("The edge weight cannot be None in a weighted graph")

                elif not isinstance(weight, float):
                    raise TypeError("Weight must be a float number")

                elif weight < 0.0 or weight > 1.0:
                    raise ValueError("Weight must be between 0.0 and 1.0")

            for node in [node1, node2]:
                # Build node if it is not in the space
                if node not in self.space.space:
                    # Build a random binary vector
                    base = self.rand.integers(2, size=self.size)

                    if self.vtype == "bipolar":
                        # Build a random bipolar vector
                        base = 2 * base - 1

                    vector = Vector(
                        name=node,
                        size=self.size,
                        vtype=self.vtype,
                        vector=copy.deepcopy(base)
                    )

                    # Define a new property called memory to store information
                    # about current node neighbors
                    setattr(vector, "memory", None)

                    if self.weighted:
                        # Define a new property called weights to store
                        # edge weights in case of a weighted graph
                        setattr(vector, "weights", dict())

                    # Register the node into the space
                    self.space.insert(vector)

                    # Increment the nodes counter
                    self.nodes_counter += 1

            # Take track of the edge as a link between the two nodes
            self.space.link(node1, node2)

            # Increment the edges counter
            self.edges_counter += 1

            if not self.directed:
                # Take track of the same edge again in case of an undirected graph
                self.space.link(node2, node1)

                # Increment the edges counter
                self.edges_counter += 1

            if self.weighted:
                # Keep track of the edge weight
                self.space.space[node1].weights[node2] = round(weight, 5)

                if not self.directed:
                    self.space.space[node2].weights[node1] = round(weight, 5)

    def _node_memory(self, node: str) -> None:
        """Build the node memory as a vector containing information about its neighbors.

        Parameters
        ----------
        node : str
            The node for which we want to build the memory.

        Raises
        ------
        Exception
            - if the input `node` is not in the graph space;
            - if the input `node` does not have any neighbors.
        """

        if node not in self.space.space:
            raise Exception("Node `{}` is not in the graph space".format(node))

        neighbors = self.space.space[node].children

        node_memory = None

        for neighbor in neighbors:
            if self.weighted:
                # Get the real weight from vector tags
                weight = self.space.space[node].weights[neighbor]

                # Retrieve the weight vector from the space
                weight_vector = self.space.space["{}__{}".format(WEIGHT_ID, weight)]

                if node_memory is None:
                    # Initialize the node memory with the first neighbor
                    # multiplied by its weight vector
                    node_memory = weight_vector * self.space.space[neighbor]

                else:
                    # Multiply each neighbor with its weight vector and
                    # bundle all the resulting vectors together to build the node memory
                    node_memory = node_memory + (weight_vector * self.space.space[neighbor])

            else:
                if node_memory is None:
                    # Initialize the node memory with the first neighbor
                    node_memory = self.space.space[neighbor]

                else:
                    # Bundle all the node's neighbors together to build the node memory
                    node_memory = node_memory + self.space.space[neighbor]

        # Store the node memory into the memory property of the node vector object
        self.space.space[node].memory = node_memory

    def _weight_memory(self, start: float, end: float, step: float=0.1) -> None:
        """Build the weights memory.

        Parameters
        ----------
        start : float
            Initial point of the weights interval.
        end : float
            Final point of the weight interval.
        step : float
            Interval step for iterating over the weight interval.
        """

        index_vector = range(self.size)
        next_level = int((self.size / 2 / len(np.arange(start, end, step))))
        change = int(self.size / 2)

        for weight in np.arange(start, end, step):
            weight = round(weight, 5)

            if weight == start:
                base = np.full(self.size, -1 if self.vtype == "bipolar" else 0)
                to_one = self.rand.permutation(index_vector)[:change]

            else:
                to_one = self.rand.permutation(index_vector)[:next_level]

            for index in to_one:
                base[index] = base[index] * -1 if self.vtype == "bipolar" else base[index] + 1

            weight_vector = Vector(
                name="{}__{}".format(WEIGHT_ID, weight),
                size=self.size,
                vtype=self.vtype,
                vector=copy.deepcopy(base)
            )

            self.space.insert(weight_vector)

    def error_rate(
        self,
        edges: Union[Set[Tuple[str, str]], Set[Tuple[str, str, Optional[float]]]],
        threshold: float=0.7
    ) -> Tuple[float, Union[Set[Tuple[str, str]], Set[Tuple[str, str, Optional[float]]]], Union[Set[Tuple[str, str]], Set[Tuple[str, str, Optional[float]]]]]:
        """Compute the error rate defined as the number of mispredicted edges on the total number of edges.
        Note that the error rate depends on the set of edges in input to this function which could be different
        from the actual set of edges used to build the graph model.

        Parameters
        ----------
        edges : set
            The set of edges used to mitigate the graph model error rate.
            Note that the edges in this set do not necessarily have to be present in the graph.
        threshold : float
            The distance threshold for establishing whether an edge exists in the graph.

        Returns
        -------
        tuple
            A tuple with the error rate, and the sets of flase positive and false negative edges
            among those in the input `edges`.

        Raises
        ------
        Exception
            - if the graph is weighted but the input edges do not have a weight;
            - if the graph is unweighted but the input edges have a weight;
            - if the tuples that define the edges contain less than 2 elements or more than 3.
        """

        # Compute the error rate as the number of mispredicted edges over the total number of edges
        false_positives = set()
        false_negatives = set()

        for edge in edges:
            weight = None

            if len(edge) == 2:
                if self.weighted:
                    raise Exception("Graph is weghted but no weights are specified")

                node1, node2 = edge

            elif len(edge) == 3:
                if not self.weighted:
                    raise Exception("Graph is unweghted but weights are specified")

                node1, node2, weight = edge

            else:
                raise Exception("Malformed edge {}".format(edge))

            exists, _ = self.edge_exists(node1, node2, weight=weight, threshold=threshold)

            if exists and node2 not in self.space.space[node1].children:
                false_positives.add(edge)

            elif not exists and node2 in self.space.space[node1].children:
                false_negatives.add(edge)

        return (len(false_positives) + len(false_negatives)) / len(edges), false_positives, false_negatives

    def error_mitigation(
        self,
        edges: Union[Set[Tuple[str, str]], Set[Tuple[str, str, Optional[float]]]],
        threshold: float=0.7,
        max_iter: int=10,
        prev_error_rate: Optional[float]=None
    ) -> None:
        """Mitigate the error rate of the graph model.

        Parameters
        ----------
        edges : set
            The set of edges used to mitigate the graph model error rate.
            Note that the edges in this set do not necessarily have to be present in the graph.
        threshold : float, default 0.7
            The distance threshold for establishing whether an edge exists in the graph.
        max_iter : int, deafult 10
            This is an iterative process that is repeated for up to `max_iter` iterations.
        prev_error_rate : float, optional
            Used to compare the error rate of the graph model with the error rate computed
            at the previous iteration. This must be initially set to `1.0`.
        """

        # Compute the graph model error rate
        error_rate, false_positives, false_negatives = self.error_rate(edges, threshold=threshold)

        if (prev_error_rate is None or error_rate < prev_error_rate) and max_iter > 0:
            # Rebuild the mispredicted node memories
            for edge in false_positives.union(false_negatives):
                weight_vector = None

                if len(edge) == 2:
                    node1, node2 = edge

                elif len(edge) == 3:
                    node1, node2, weight = edge

                    if self.weighted and weight:
                        # Retrieve the weight vector from the space
                        weight_vector = self.space.space["{}__{}".format(WEIGHT_ID, weight)]

                if self.space.space[node1].memory:
                    if edge in false_positives:
                        # Reduce the signal of node2 in the memory of node1
                        if weight_vector is not None:
                            self.space.space[node1].memory -= (weight_vector * self.space.space[node2])

                        else:
                            self.space.space[node1].memory -= self.space.space[node2]

                    elif edge in false_negatives:
                        if weight_vector is not None:
                            self.space.space[node1].memory += (weight_vector * self.space.space[node2])

                        else:
                            # Increase the signal of node2 in the memory of node1
                            self.space.space[node1].memory += self.space.space[node2]

            # Rebuild the graph model
            # Do not rebuild the nodes memory since they have been overwritten earlier
            self.fit(self.edges, build_nodes_memory=False)

            # Recursively mitigate the error rate
            self.error_mitigation(edges, threshold=threshold, max_iter=max_iter-1, prev_error_rate=error_rate)

    def fit(
        self,
        edges: Union[Set[Tuple[str, str]], Set[Tuple[str, str, Optional[float]]]],
        build_nodes_memory: bool=True
    ) -> None:
        """Build the graph memory and store it into the space.

        Parameters
        ----------
        edges : set
            The set of edges defined as tuples `<source, target, weight>`.
            Note that `weight` is optional in case of unweighted graphs.
        build_nodes_memory : bool, default True
            Build nodes and weight memories by default.
            This must be set to False in case this is invoked to mitigate the error rate.

        Raises
        ------
        ValueError
            If no edges are provided in input.
        Exception
            - if the tuple contains two elements but the graph is weighted;
            - if the tuple contains three elements but the graph is unweighted;
            - if the tuple representing the edge contain less than 2 elements or more than 3.
        """

        if not edges:
            raise ValueError("Must provide at least one edge")

        for edge in edges:
            if len(edge) == 2:
                if self.weighted:
                    raise Exception("Graph is weghted but no weights are specified")

                node1, node2 = edge

                # This edge is unweighted
                self._add_edge(node1, node2)

            elif len(edge) == 3:
                if not self.weighted:
                    raise Exception("Graph is unweighted but weights are specified")

                node1, node2, weight = edge

                # This edge is weighted
                self._add_edge(node1, node2, weight=weight)

            else:
                raise Exception("Malformed edge {}".format(edge))

        if self.weighted and build_nodes_memory:
            # Build the vector representation of the edges weight
            # Weights are float numbers between 0.0 and 1.0, here limited to the second decimal point
            self._weight_memory(0.0, 1.0, 1 / self.weights)

        graph = None

        for node in self.space.space:
            # Check whether the current node is not the actual graph memory
            # Also, check whether the current node is not a weight vector
            # This is required because this function can be run multiple times
            if node != GRAPH_ID and not node.startswith(WEIGHT_ID):
                if build_nodes_memory:
                    # Build the node memory
                    self._node_memory(node)

                if self.directed:
                    if graph is None:
                        if self.space.space[node].memory:
                            graph = self.space.space[node] * permute(self.space.space[node].memory, rotate_by=1)

                        else:
                            graph = self.space.space[node]

                    else:
                        if self.space.space[node].memory:
                            # Build the graph memory as the bundle of all the node memories rotated by 1 position
                            graph = graph + (self.space.space[node] * permute(self.space.space[node].memory, rotate_by=1))

                        else:
                            graph = graph + self.space.space[node]

                else:
                    if graph is None:
                        if self.space.space[node].memory:
                            graph = self.space.space[node] * self.space.space[node].memory

                        else:
                            graph = self.space.space[node]

                    else:
                        if self.space.space[node].memory:
                            # Build the graph memory as the bundle of all the node memories
                            graph = graph + (self.space.space[node] * self.space.space[node].memory)

                        else:
                            graph = graph + self.space.space[node]

        if not self.directed:
            # Introduce a factor 1/2 because if we expand the node memory, then
            # H(i)*H(j) and H(j)*H(i) will be counted twice
            graph.vector = graph.vector / 2

        # Check whether a graph is already present in the space
        if GRAPH_ID in self.space.space:
            self.space.remove(GRAPH_ID)

        # Rename the graph vector
        graph.name = GRAPH_ID

        # Store the graph vector into the space
        self.space.insert(graph)

        # Also keep track of the edges
        # This is used in case of the error_mitigation()
        self.edges = edges

    def edge_exists(
        self,
        node1: str,
        node2: str,
        weight: Optional[float]=None,
        threshold: float=0.7
    ) -> Tuple[bool, float]:
        """Check whether an edge exists between `node1` and `node2` according to a specified distance `threshold`.

        node1 : str
            The source node name or ID.
        node2 : str
            The target node name or ID.
        weight : float, optional
            The edge weight.
        threshold : float, default 0.7
            The distance threshold on vectors to establish the presence of the edge.

        Returns
        -------
        Tuple
            True in case an edge between `node1` and `node2` exists in the graph space,
            otherwise False. It also returns the actual distance between the two vectors.

        Raises
        ------
        Exception
            - if there is no graph vector in the space;
            - if `node1` and `node2` are not in the graph space.
        TypeError
            If `weight` is not a float.
        ValueError
            - if the graph is weighted but `weight` is None;
            - if `weight` is <0.0 or >1.0.
        """

        if GRAPH_ID not in self.space.space:
            raise Exception("There is no graph in the space")

        for node in [node1, node2]:
            if node not in self.space.space:
                raise Exception("Node '{}' is not in the space".format(node))

        # Check whether the graph is weighted and the edge weight is a valid number
        # In case the graph is unweighted, ignore the weight
        if self.weighted:
            if weight is None:
                raise ValueError("The edge weight cannot be None in a weighted graph")

            elif not isinstance(weight, float):
                raise TypeError("Weight must be a float number")

            elif weight < 0.0 or weight > 1.0:
                raise ValueError("Weight must be between 0.0 and 1.0")

        # Retrieve the vector representation of the graph
        graph = self.space.space[GRAPH_ID]

        # Also retrieve the vector representations of the two input nodes
        node1_vector = self.space.space[node1]
        node2_vector = self.space.space[node2]

        # Retrieve the node1 memory by binding the vector representation of
        # node1 with the vector representation of the graph
        # The resulting vector equals to the actual node1 memory plus noise
        node1_memory = bind(node1_vector, graph)

        if self.directed:
            # In case of directed graphs, nodes memory are rotated by 1 position
            # Thus, it must be rotated back in order to preserve the similarity
            node1_memory = permute(node1_memory, rotate_by=-1)

        # Check whether there is a edge between node1 and node2 by computing the
        # distance between node2 and node1 memory
        # A distance close to 0 means the edge exists
        # A distance close to 1 means the edge does not exist
        if self.weighted:
            # Retrieve the weight vector from the space
            weight_vector = self.space.space["{}__{}".format(WEIGHT_ID, weight)]

            distance = (weight_vector * node2_vector).dist(node1_memory, method="cosine")

        else:
            distance = node2_vector.dist(node1_memory, method="cosine")

        if distance < threshold:
            return True, distance

        return False, distance
