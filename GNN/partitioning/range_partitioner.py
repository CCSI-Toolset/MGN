import torch

from GNN.utils.data_utils import k_hop_subgraph
from GNN.partitioning.partitioner import Partitioner


class RangePartitioner(Partitioner):
    """
    This class generates partitions of a torch_geometric Dataset object,
    where the partitions are hyperrectangles pre-specified by the user
    """

    def __init__(self, ranges, padding, **kwargs):
        """
        ranges is a list of d lists, where d is the dimension of the hyperrectangle
        each list range_i in lists contains 2-tuples, defining the
        vertices of the hyperrectangles
        Graph will be partitioned into N patches, where N is the size of
        all the range_is.
        """
        # check that the same number of coordinates are specified for
        # every dimensions
        num_partitions = len(ranges[0])
        for dim in ranges:
            assert len(dim) == num_partitions
        self.ranges = ranges
        self.num_dimensions = len(ranges)
        super().__init__(padding, **kwargs)

    def get_partition(self, data, num_edge_types=1):
        eps = 1e-8
        # check that the number of dimensions matches that of the input data
        if data.pos.shape[1] != self.num_dimensions:
            raise ValueError(
                "Expected %s dimensions, but input data has %s"
                % (self.num_dimensions, data.pos.shape[1])
            )
        num_nodes = data.pos.shape[0]
        cells = []
        # for each hyperrectangle
        for lim in zip(*self.ranges):
            # get the nodes that are within boundaries in all dimensions
            start, end = lim[0][0], lim[0][1] + eps
            cell_mask = (start <= data.pos[:, 0]) & (data.pos[:, 0] < end)
            for dim_ix in range(1, self.num_dimensions):
                start, end = lim[dim_ix][0], lim[dim_ix][1] + eps
                cell_mask = (
                    cell_mask
                    & (start <= data.pos[:, dim_ix])
                    & (data.pos[:, dim_ix] < end)
                )
            nodes_in_cell = torch.arange(num_nodes)[cell_mask]
            padded_cells, edge_masks = [], []
            # first, we get the k-hop subgraph for every edge type
            for k in range(num_edge_types):
                edge_index = data["edge_index_%s" % k]
                padded_cell, _, _ = k_hop_subgraph(
                    nodes_in_cell, self.padding, edge_index
                )
                padded_cells.append(padded_cell)
            # final padded cell is the union of the padded cells for each edge_type
            # note that the set of internal nodes is always the same, only the the
            # padding (i.e., ghost nodes) are potentially different
            padded_cell = torch.cat(padded_cells).unique()
            internal_node_mask = torch.isin(padded_cell, nodes_in_cell)
            # now that we know which nodes go in the cell, get the edges for all edge types
            for k in range(num_edge_types):
                edge_index = data["edge_index_%s" % k]
                _, _, edge_mask = k_hop_subgraph(padded_cell, 0, edge_index)
                edge_masks.append(edge_mask)
            # if we only have one edge type, we return the corresponding mask, not a list
            if num_edge_types == 1:
                edge_masks = edge_masks[0]
            cells.append((padded_cell, edge_masks, internal_node_mask))
        return cells

    def __str__(self):
        return "range_npatches%s_padding%s" % (len(self.ranges[0]), self.padding)
