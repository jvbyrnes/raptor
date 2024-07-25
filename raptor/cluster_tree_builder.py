import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    """
    Configuration class for ClusterTreeBuilder.

    Attributes:
        reduction_dimension (int): The dimension to which data should be reduced.
        clustering_algorithm (ClusteringAlgorithm): The algorithm to use for clustering.
        clustering_params (dict): Additional parameters for the clustering algorithm.
    """

    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary

class ClusterTreeBuilder(TreeBuilder):
    """
    ClusterTreeBuilder is responsible for constructing a hierarchical tree structure
    using clustering algorithms. It extends the TreeBuilder class and utilizes a 
    specified clustering algorithm to group nodes at each layer of the tree.

    Attributes:
        reduction_dimension (int): The dimension to which data should be reduced.
        clustering_algorithm (ClusteringAlgorithm): The algorithm to use for clustering.
        clustering_params (dict): Additional parameters for the clustering algorithm.

    Methods:
        construct_tree(current_level_nodes, all_tree_nodes, layer_to_nodes, use_multithreading):
            Constructs the hierarchical tree by clustering nodes at each layer.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        """
        Constructs a hierarchical tree by clustering nodes at each layer. Starts at 'bottom layer' (original chunks) and goes up self.num_layers.

        Args:
            current_level_nodes (Dict[int, Node]): Nodes at the current level of the tree.
            all_tree_nodes (Dict[int, Node]): All nodes in the tree.
            layer_to_nodes (Dict[int, List[Node]]): Mapping of layers to their respective nodes.
            use_multithreading (bool, optional): Whether to use multithreading for processing clusters. Defaults to False.

        Returns:
            Dict[int, Node]: Updated nodes at the current level after constructing the tree.
        """
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            """
            Processes a single cluster of nodes by summarizing their texts and creating a new parent node.

            Args:
                cluster (List[Node]): The cluster of nodes to process.
                new_level_nodes (Dict[int, Node]): The dictionary to store new level nodes.
                next_node_index (int): The index for the next node to be created.
                summarization_length (int): The maximum length for the summarized text.
                lock (Lock): A threading lock to ensure thread-safe operations.

            Returns:
                None
            """
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        def should_stop_layer_construction(node_list_current_layer: List[Node], layer: int) -> bool:
            """
            Determines if layer construction should stop based on the number of nodes in the current layer.

            Args:
                node_list_current_layer (List[Node]): List of nodes in the current layer.
                layer (int): The current layer index.

            Returns:
                bool: True if layer construction should stop, False otherwise.
            """
            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                return True
            return False

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            # Check if we should stop layer construction
            if should_stop_layer_construction(node_list_current_layer, layer):
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
