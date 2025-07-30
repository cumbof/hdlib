"""Directed and undirected, weighted and unweighted graphs with hdlib.

It implements the __hdlib.graph.Graph__ class object which allows to represent weighted directed and undirected graphs
built according to the Hyperdimensional Computing (HDC) paradigm as described in _Poduval et al. 2022_ https://doi.org/10.3389/fnins.2022.757125."""

import copy
import multiprocessing as mp
from typing import Any, Optional, Set, Tuple

import numpy as np

from hdlib import __version__
from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, permute

# Private graph vector name
GRAPH_ID = "__graph__"

# Private weight vector name prefix
WEIGHT_ID = "__weight__"


class Graph(object):
    """Hyperdimensional Graph representation."""

    def __init__(
        self,
        size: int=10000,
        directed: bool=False,
        seed: Optional[int]=None
    ) -> "Graph":
        """Initialize a Graph object.

        Parameters
        ----------
        size : int, default 10000
            The size of vectors used to create a Space and define Vector objects.
        directed : bool, default False
            Directed or undirected.
        seed : int, optional
            Random seed for reproducibility of results.

        Raises
        ------
        TypeError
            If the vector size is not an integer number.
        ValueError
            If the vector size is lower than 1,000.

        Examples
        --------
        >>> from hdlib.graph import Graph
        >>> graph = Graph(size=10000, vtype='bipolar', directed=False)
        >>> type(graph)
        <class 'hdlib.graph.Graph'>

        This creates a new undirected Graph object around a Space that can host random bipolar Vector objects with size 10,000.
        """

        if not isinstance(size, int):
            raise TypeError("Vectors size must be an integer number")

        if size < 1000:
            raise ValueError("Vectors size must be greater than or equal to 1000")

        # Register vectors dimensionality
        self.size = size

        # Register a set with edge weights (classes)
        self.weights = set()

        # Register vectors type
        self.vtype = "bipolar"

        # Register whether the graph is directed or undirected
        self.directed = directed

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
            the number of nodes, the number of edges, and whether it is directed or undirected.

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
                Weights:  100
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
            Weights:  {}
            Nodes:    {}
            Edges:    {}
            Seed:     {}
        """.format(
            self.version,
            self.size,
            self.vtype,
            self.directed,
            len(self.weights),
            self.nodes_counter,
            self.edges_counter,
            self.seed
        )

    def _add_edge(
        self,
        node1: str,
        node2: str,
        weight: Any,
    ) -> None:
        """Add an edge to the graph and automatically build nodes if they do not exist in the space.

        Parameters
        ----------
        node1 : str
            Node name.
        node2 : str
            Node name.
        weight : Any
            The edge weight.
            This can be numeric or a string used as a class label.

        Raises
        ------
        ValueError
            If `node1` or `node2` is equals to `GRAPH_ID`.
        """

        if node1 == GRAPH_ID or node2 == GRAPH_ID:
            raise ValueError(f"Node names cannot match with the private graph ID `{GRAPH_ID}`")

        edge_exists = False

        if node1 in self.space.space and node2 in self.space.space:
            # Check whether an edge between node1 and node2 already exists
            if self.directed and node2 in self.space.space[node1].children:
                edge_exists = True

            elif not self.directed and node2 in self.space.space[node1].children and node1 in self.space.space[node2].children:
                edge_exists = True

        if not edge_exists:
            for node in [node1, node2]:
                # Build node if it is not in the space
                if node not in self.space.space:
                    # Build a random binary vector
                    vector = Vector(
                        name=node,
                        size=self.size,
                        vtype=self.vtype
                    )

                    # Define a new property called memory to store information
                    # about current node neighbors
                    setattr(vector, "memory", None)

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

            # Keep track of the edge weight
            if node2 not in self.space.space[node1].weights:
                self.space.space[node1].weights[node2] = set()

            if not self.directed:
                if node1 not in self.space.space[node2].weights:
                    self.space.space[node2].weights[node1] = set()

        self.space.space[node1].weights[node2].add(weight)

        if not self.directed:
            self.space.space[node2].weights[node1].add(weight)

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
            raise Exception(f"Node `{node}` is not in the graph space")

        neighbors = self.space.space[node].children

        node_memory = None

        for neighbor in neighbors:
            # Get the real weight from vector tags
            for weight in self.space.space[node].weights[neighbor]:
                # Retrieve the weight vector from the space
                weight_vector = self.space.space[f"{WEIGHT_ID}{weight}"]

                if node_memory is None:
                    # Initialize the node memory with the first neighbor
                    # multiplied by its weight vector
                    node_memory = weight_vector * self.space.space[neighbor]

                else:
                    # Multiply each neighbor with its weight vector and
                    # bundle all the resulting vectors together to build the node memory
                    node_memory = node_memory + (weight_vector * self.space.space[neighbor])

        # Store the node memory into the memory property of the node vector object
        self.space.space[node].memory = node_memory

    def _weight_memory(self) -> None:
        """Build the weights memory.
        """

        # Recover edge weights from the space
        for node in self.space.space:
            # Check whether the current node is not the actual graph memory
            # Also, check whether the current node is not a weight vector
            if node != GRAPH_ID and not node.startswith(WEIGHT_ID):
                for neighbor in self.space.space[node].weights:
                    # Retrieve the weights on these edges
                    self.weights.update(self.space.space[node].weights[neighbor])

        for weight in self.weights:
            # Build a random vector
            weight_vector = Vector(name=f"{WEIGHT_ID}{weight}", size=self.size, vtype=self.vtype)

            self.space.insert(weight_vector)

    @staticmethod
    def _error_rate(
        instance: "Database",
        edges: Set[Tuple[str, str, Any]],
    ) -> Tuple[float, Set[Tuple[str, str, float]], Set[Tuple[str, str, float]]]:
        """Just a wrapper around the `error_rate` function to make it callable in multiprocessing.
        It is safe to run in multiprocessing because it does not modify any instance attributes.

        Parameters
        ----------
        instance : Database
            A Graph instance.
        edges : set
            The set of edges used to mitigate the graph model error rate.
            Note that the edges in this set do not necessarily have to be present in the graph.

        Returns
        -------
        tuple
            A tuple with the error rate, and the sets of flase positive and false negative edges
            among those in the input `edges`.
        """

        return instance.error_rate(edges)

    def error_rate(
        self,
        edges: Set[Tuple[str, str, Any]],
    ) -> Tuple[float, Set[Tuple[str, str, float]], Set[Tuple[str, str, float]]]:
        """Compute the error rate defined as the number of mispredicted edges on the total number of edges.
        Note that the error rate depends on the set of edges in input to this function which could be different
        from the actual set of edges used to build the graph model.

        Parameters
        ----------
        edges : set
            The set of edges used to mitigate the graph model error rate.
            Note that the edges in this set do not necessarily have to be present in the graph.

        Returns
        -------
        tuple
            A tuple with the error rate, and the sets of flase positive and false negative edges
            among those in the input `edges`.
        """

        # Compute the error rate as the number of mispredicted edges over the total number of edges
        false_positives = set()
        false_negatives = set()

        # We estimate a proper threshold to establish whether an edge exists between two nodes
        # Keep track of these thresholds to avoid recomputing them
        # This is also weight dependant
        weight_to_node_specific_thresholds = dict()

        for edge in edges:
            node1, node2, weight_true = edge

            weight_pred = list()

            for weight in self.weights:
                node1_threshold = weight_to_node_specific_thresholds[weight].get(node1) if weight in weight_to_node_specific_thresholds else None

                _, weight_dist, dist_threshold = self.edge_exists(node1, node2, weight, threshold=node1_threshold)

                weight_pred.append((weight, weight_dist))

                if weight not in weight_to_node_specific_thresholds:
                    weight_to_node_specific_thresholds[weight] = dict()

                if node1 not in weight_to_node_specific_thresholds[weight]:
                    weight_to_node_specific_thresholds[weight][node1] = dist_threshold

            if len(weight_pred) > 1:
                # Sort weights based on their distances
                weight_pred = sorted(weight_pred, key=lambda w: w[1])

                # Get the difference between the closest and the farthest distances
                dist_diff = weight_pred[-1][1] - weight_pred[0][1]

                # Use the difference in distances to select the top closest weights
                weight_pred = [w[0] for w in weight_pred if w[1] - weight_pred[0][1] < dist_diff]

                if weight_true in weight_pred and node2 not in self.space.space[node1].children:
                    false_positives.add(edge)

                elif weight_true not in weight_pred and node2 in self.space.space[node1].children:
                    false_negatives.add(edge)

            else:
                weight_pred = weight_pred[0]

                if weight_true == weight_pred and node2 not in self.space.space[node1].children:
                    false_positives.add(edge)

                elif weight_true != weight_pred and node2 in self.space.space[node1].children:
                    false_negatives.add(edge)

        return (len(false_positives) + len(false_negatives)) / len(edges), false_positives, false_negatives

    def error_mitigation(
        self,
        edges: Set[Tuple[str, str, Any]],
        max_iter: int=10,
        nproc: int=1
    ) -> None:
        """Mitigate the error rate of the graph model.

        Parameters
        ----------
        edges : set
            The set of edges used to mitigate the graph model error rate.
            Note that the edges in this set do not necessarily have to be present in the graph.
        max_iter : int, deafult 10
            This is an iterative process that is repeated for up to `max_iter` iterations.
        nproc : int, default 1
            Maximum number of jobs for multiprocessing.
        """

        if max_iter > 0:
            # Redefine the number of CPUs for multiprocessing
            if nproc < 1:
                nproc = 1

            elif nproc > mp.cpu_count():
                nproc = mp.cpu_count()

            # Split the set of edges into equally sized chunks
            chunk_size = len(edges) // nproc

            # Very inefficient but required
            edges_list = list(edges)

            edges_subsets = [(self, set(edges_list[i:i+chunk_size])) for i in range(0, len(edges_list), chunk_size)]

            # Keep track of misclassified edges
            false_positives = set()
            false_negatives = set()

            with mp.Pool(processes=nproc) as pool:
                error_estimation = pool.starmap(self.__class__._error_rate, edges_subsets)

                # Retrieve the set of misclassified edges
                for _, false_positives_partial, false_negatives_partial in error_estimation:
                    false_positives = false_positives.union(false_positives_partial)
                    false_negatives = false_negatives.union(false_negatives_partial)

            prev_error_rate = (len(false_positives) + len(false_negatives)) / len(edges)

            print(f"(base)\tError rate: {prev_error_rate}\tFalse positives: {len(false_positives)}\tFalse negatives: {len(false_negatives)}")

            # Error mitigate the model if the error rate is > 0
            if prev_error_rate > 0:
                for i in range(max_iter):
                    # Work on a copy of self here
                    graph = copy.deepcopy(self)

                    # Rebuild the mispredicted node memories
                    for edge in false_positives.union(false_negatives):
                        node1, node2, weight = edge

                        # Retrieve the weight vector from the space
                        weight_vector = graph.space.space[f"{WEIGHT_ID}{weight}"]

                        if graph.space.space[node1].memory:
                            if edge in false_positives:
                                # Reduce the signal of node2 in the memory of node1
                                graph.space.space[node1].memory -= (weight_vector * graph.space.space[node2])

                            elif edge in false_negatives:
                                graph.space.space[node1].memory += (weight_vector * graph.space.space[node2])

                    # Rebuild the graph model
                    # Do not rebuild the nodes memory since they have been overwritten earlier
                    graph.fit(graph.edges, build_nodes_memory=False)

                    # Reset `false_positives` and `false_negatives`
                    false_positives = set()
                    false_negatives = set()

                    # Recompute the error rate
                    with mp.Pool(processes=nproc) as pool:
                        error_estimation = pool.starmap(graph.__class__._error_rate, edges_subsets)

                        # Retrieve the set of misclassified edges
                        for _, false_positives_partial, false_negatives_partial in error_estimation:
                            false_positives = false_positives.union(false_positives_partial)
                            false_negatives = false_negatives.union(false_negatives_partial)

                    error_rate = (len(false_positives) + len(false_negatives)) / len(edges)

                    print(f"(iter {i+1})\tError rate: {error_rate}\tFalse positives: {len(false_positives)}\tFalse negatives: {len(false_negatives)}")

                    if error_rate > prev_error_rate:
                        # Stop the error mitigating the graph model here
                        break

                    # Update the error rate
                    prev_error_rate = error_rate

                    # Update the graph model
                    # Make the error mitigated graph persistent
                    self.__dict__.update(graph.__dict__)

                    if error_rate == 0.0:
                        # There is nothing else to do here
                        break

    def fit(
        self,
        edges: Set[Tuple[str, str, Any]],
        build_nodes_memory: bool=True
    ) -> None:
        """Build the graph memory and store it into the space.

        Parameters
        ----------
        edges : set
            The set of edges defined as tuples `<source, target, weight>`.
        build_nodes_memory : bool, default True
            Build nodes and weight memories by default.
            This must be set to False in case this is invoked to mitigate the error rate.

        Raises
        ------
        ValueError
            If no edges are provided in input.
        """

        if not edges:
            raise ValueError("Must provide at least one edge")

        for edge in edges:
            node1, node2, weight = edge

            self._add_edge(node1, node2, weight)

        if build_nodes_memory:
            # Build the vector representation of the edges weight
            self._weight_memory()

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
        weight: Any,
        threshold: Optional[float]=None
    ) -> Tuple[bool, float, float]:
        """Check whether an edge exists between `node1` and `node2` according to a specified distance `threshold`.

        node1 : str
            The source node name or ID.
        node2 : str
            The target node name or ID.
        weight : Any
            The edge weight.
        threshold : float, default None
            The distance threshold on vectors to establish the presence of the edge.
            It is automatically estimated if None.

        Returns
        -------
        Tuple
            True in case an edge between `node1` and `node2` exists in the graph space,
            otherwise False. It also returns the actual distance between the two vectors and
            the threshold used to establish whether the edge exists.

        Raises
        ------
        Exception
            - if there is no graph vector in the space;
            - if `node1` and `node2` are not in the graph space.
        """

        if GRAPH_ID not in self.space.space:
            raise Exception("There is no graph in space")

        for node in [node1, node2]:
            if node not in self.space.space:
                raise Exception(f"Node '{node}' not in space")

        if f"{WEIGHT_ID}{weight}" not in self.space.space:
            raise Exception(f"Weight vector '{weight}' not in space")

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

        # Check whether there is a edge between node1 and node2 by computing the distance between node2 and node1 memory
        # A distance close to 0 means the edge exists
        # A distance close to 1 means the edge does not exist
        # Retrieve the weight vector from the space
        weight_vector = self.space.space[f"{WEIGHT_ID}{weight}"]

        if threshold == None:
            if not self.space.space[node1].children:
                # We cannot do much if node1 has no children
                threshold = 1.0

            else:
                # Retrieve node1 children
                # There could be hundreds of thousands of nodes here, so we focus on a random selection (5%)
                selection = len(self.space.space[node1].children) if len(self.space.space[node1].children) < 100 else int(len(self.space.space[node1].children)*5.0//100.0)

                neighbors = set(self.rand.choice(list(self.space.space[node1].children), size=selection, replace=False))

                # Pick the same number of random non-neighbors
                # This is very inefficient!
                non_neighbors = set(self.rand.choice(list(self.space.space.keys()), size=len(neighbors), replace=False))

                # Keep track of cosine distances between node1 and its neighbors plus random non-neighbors
                distances = list()

                for node in neighbors.union(non_neighbors):
                    if node == GRAPH_ID or node.startswith(WEIGHT_ID):
                        continue

                    # Retrieve the node vector from the space
                    # node is numpy.str_
                    node_vector = self.space.space[str(node)]

                    with np.errstate(invalid="ignore", divide="ignore"):
                        distances.append((weight_vector * node_vector).dist(node1_memory, method="cosine"))

                # Use the 5th percentile as threshold
                # This is a node-specific threshold for node1
                threshold = float(np.percentile(distances, 5))

        with np.errstate(invalid="ignore", divide="ignore"):
            distance = (weight_vector * node2_vector).dist(node1_memory, method="cosine")

        if distance < threshold:
            return True, distance, threshold

        return False, distance, threshold

    def predict(
        self,
        edges: Set[Tuple[str, str, Any]]
    ) -> Tuple[Any, float]:
        """Predict the weight (class) of a specific set of edges.

        Parameters
        ----------
        edges : set
            Set of edges.

        Returns
        -------
        tuple
            A tuple with the predicted class and it's accuracy in terms of percentage of edges
            that matched the predicted class.
        """

        hits = {weight: 0 for weight in self.weights}

        # We estimate a proper threshold to establish whether an edge exists between two nodes
        # Keep track of these thresholds to avoid recomputing them
        # This is also weight dependant
        weight_to_node_specific_thresholds = dict()

        for node1, node2, edge_weight in edges:
            if node1 in self.space.space and node2 in self.space.space:
                weight_pred = list()

                for weight in self.weights:
                    node1_threshold = weight_to_node_specific_thresholds[weight].get(node1) if weight in weight_to_node_specific_thresholds else None

                    _, weight_dist, dist_threshold = self.edge_exists(node1, node2, weight, threshold=node1_threshold)

                    weight_pred.append((weight, weight_dist))

                    if weight not in weight_to_node_specific_thresholds:
                        weight_to_node_specific_thresholds[weight] = dict()

                    if node1 not in weight_to_node_specific_thresholds[weight]:
                        weight_to_node_specific_thresholds[weight][node1] = dist_threshold

                if len(weight_pred) > 1:
                    # Sort weights based on their distances
                    weight_pred = sorted(weight_pred, key=lambda w: w[1])

                    # Get the difference between the closest and the farthest distances
                    dist_diff = weight_pred[-1][1] - weight_pred[0][1]

                    # Use the difference in distances to select the top closest weights
                    weight_pred = [w[0] for w in weight_pred if w[1] - weight_pred[0][1] < dist_diff]

                    for weight in weight_pred:
                        hits[weight] += 1

                else:
                    weight_pred = weight_pred[0]

                    hits[weight_pred] += 1

        # Get the best hit
        prediction = max(hits, key=hits.get)

        accuracy = (hits[prediction] / len(edges)) * 100.0

        return (prediction, accuracy)
