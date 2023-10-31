from abc import abstractmethod, ABC


class Partitioner(ABC):
    def __init__(self, padding, **kwargs):
        self.padding = padding
        super().__init__(**kwargs)

    @abstractmethod
    def get_partition(self, data, num_edge_types=1):
        """Given a torch_geometric Data object, return
        a partition of the graph.
        "padding" is a parameter indicating whether and how much
        to expand each cluster. If padding is 0, a true partition
        of the graph is returned (i.e., no overlap).
        For "padding" = k, the k-hop neighborhood will be added
        to each cluster.
        The partition is represented
        as a list of clusters, where each element is a triple
        (node_list, edge_mask, internal_node_mask), where node_list
        is a torch tensor listing all the nodes in the cluster,
        edge_mask is a boolean tensor indicating which edges of the
        "data" object are inside the cluster and internal_node_mask
        is a boolean tensor indicating which nodes were in the
        original cluster (1) and which ones were added as part of
        the padding (0)
        """
        pass

    @abstractmethod
    def __str__(self):
        pass
