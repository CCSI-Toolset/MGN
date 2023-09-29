import torch
from GNN.partitioning.partitioner import Partitioner


class NullPartitioner(Partitioner):
    """
    This class returns a "null" partition, defined as the entire
    dataset. That is, no partition is applied.
    """

    def __init__(self, **kwargs):
        padding = 0
        super().__init__(padding, **kwargs)

    def get_partition(self, data, num_edge_types=1):
        num_nodes = data.pos.shape[0]
        # the cell is the entire graph
        cell = torch.arange(num_nodes)
        # all nodes are internal
        internal_node_mask = torch.ones_like(cell, dtype=bool)
        # all edges are present
        edge_masks = []
        for i in range(num_edge_types):
            edge_index = data["edge_index_%s" % i]
            edge_mask = torch.ones_like(edge_index[0], dtype=bool)
            edge_masks.append(edge_mask)
        if num_edge_types == 1:
            edge_masks = edge_masks[0]
        cells = [(cell, edge_masks, internal_node_mask)]
        return cells

    def __str__(self):
        return "null"
