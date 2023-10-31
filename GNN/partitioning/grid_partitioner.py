import torch

from GNN.utils.data_utils import k_hop_subgraph
from GNN.partitioning.partitioner import Partitioner


class GridPartitioner(Partitioner):
    """
    This class partitions a torch_geometric Dataset object
    into a grid
    """

    def __init__(self, nrows, ncols, padding, **kwargs):
        """
        nrows and ncols are the number of rows and columns in
        the grid, respectively
        both parameters must be positive integers
        """
        assert type(nrows) == int and nrows > 0
        assert type(ncols) == int and ncols > 0
        self.nrows = nrows
        self.ncols = ncols
        super().__init__(padding, **kwargs)

    def get_partition(self, data, num_edge_types=1):
        eps = 1e-6
        # find the boundaries of the graph and use them
        # to compute the height and width of each cell
        x_min, y_min = data.pos.min(axis=0)[0]
        x_max, y_max = data.pos.max(axis=0)[0]
        cell_width = (x_max + eps - x_min) / self.ncols
        cell_height = (y_max + eps - y_min) / self.nrows
        num_nodes = data.pos.shape[0]
        cells = []
        # for each column
        for i in range(self.ncols):
            x_start = x_min + (i * cell_width)
            x_end = x_min + ((i + 1) * cell_width)
            # for each row
            for j in range(self.nrows):
                y_start = y_min + (j * cell_height)
                y_end = y_min + ((j + 1) * cell_height)
                # find nodes in ij-th cell
                cell_mask = (
                    (x_start <= data.pos[:, 0])
                    & (data.pos[:, 0] < x_end)
                    & (y_start <= data.pos[:, 1])
                    & (data.pos[:, 1] < y_end)
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
                # note that the set of internal nodes is always the same, only the
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
        return "grid_ncols%s_nrows%s_padding%s" % (self.ncols, self.nrows, self.padding)
