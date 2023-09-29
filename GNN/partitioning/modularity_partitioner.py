import numpy as np
import torch
from community import best_partition
from scipy.stats import pearsonr
from torch_geometric.utils.convert import to_networkx
from GNN.utils.data_utils import k_hop_subgraph
from GNN.partitioning.partitioner import Partitioner


class ModularityPartitioner(Partitioner):
    """ """

    def __init__(self, padding, **kwargs):
        super().__init__(padding, **kwargs)

    def get_partition(self, data, num_edge_types=1):
        networkx_graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
        self.get_edge_weights(networkx_graph, data.output)
        # get communities by modularity maximization
        # membership is a dictionary of node ID to community ID
        membership = best_partition(networkx_graph, weight="weight")
        num_communities = max(membership.values()) + 1
        communities = [[] for i in range(num_communities)]
        for node_id, community_id in membership.items():
            communities[community_id].append(node_id)
        community_graphs = []
        # create one graph per community
        for nodes in communities:
            padded_graphs, edge_masks = [], []
            # first, we get the k-hop subgraph for every edge type
            for k in range(num_edge_types):
                edge_index = data["edge_index_%s" % k]
                padded_graph, _, _ = k_hop_subgraph(nodes, self.padding, edge_index)
                padded_graphs.append(padded_graph)
            # final padded graph is the union of the padded graphs for each edge_type
            # note that the set of internal nodes is always the same, only the the
            # padding (i.e., ghost nodes) are potentially different
            padded_graph = torch.cat(padded_graphs).unique()
            internal_node_mask = torch.isin(padded_graph, nodes)
            # now that we know which nodes go in the subgraph, get the edges for all edge types
            for k in range(num_edge_types):
                edge_index = data["edge_index_%s" % k]
                _, _, edge_mask = k_hop_subgraph(padded_graph, 0, edge_index)
                edge_masks.append(edge_mask)
            # if we only have one edge type, we return the corresponding mask, not a list
            if num_edge_types == 1:
                edge_masks = edge_masks[0]
            community_graphs.append((padded_graph, edge_masks, internal_node_mask))
        return community_graphs

    def get_edge_weights(self, G, output):
        if torch.is_tensor(output):
            output = output.numpy()
        output = np.linalg.norm(output, axis=2)
        for u, v in G.edges():
            # data for u from 1 to T-1 and v from 2 to T
            output_u = output[:-1, u]
            output_v = output[1:, v]
            # corner cases
            # if the two arrays are equal to each other, the correlation is 1
            if np.all(np.isclose(output_u, output_v)):
                corr = 1.0
            # if one of the arrays is constant, the correlation is zero
            elif np.all(output_u == output_u[0]) or np.all(output_v == output_v[0]):
                corr = 0.0
            else:
                # take the absolute value
                # modularity maximization is not well-defined for negative weights
                corr = np.abs(pearsonr(output_u, output_v)[0])
            assert ~np.isnan(corr)
            G[u][v]["weight"] = corr

    def __str__(self):
        return "modularity_padding%s" % (self.padding)
