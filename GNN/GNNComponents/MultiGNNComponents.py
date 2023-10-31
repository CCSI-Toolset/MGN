from torch import cat
from torch.nn import Module, ModuleList
from torch_scatter import scatter_sum

from .GNNComponents import MLP, EdgeProcessor
from GNN.GNNComponents.skipConnections import (
    FullResidualConnection,
    ResidualConnection,
    InitialConnection,
    DenseConnection,
)
from typing import Optional, Tuple, List
from torch import Tensor


class MetaLayerSCMultigraph(Module):
    """
    MetaLayer for multigraphs
    Based on torch_geometric.nn.MetaLayer; needed for graphs with multiple edge types

    edge_models (list of torch.nn.Module or torch.nn.ModuleList, optional):
        A callable which updates a
        multigraph's edge features based on source and target node features,
        current edge features and global features.
        (default: :obj:`None`)
    node_model (torch.nn.Module, optional): A callable which updates a
        graph's node features based on its current node features, its graph
        connectivity, its edge features and its global features.
        (default: :obj:`None`)
    global_model (torch.nn.Module, optional): A callable which updates a
        graph's global features based on its node features, its graph
        connectivity, its edge features and its current global features.
        (default: :obj:`None`)
    """

    def __init__(
        self,
        edge_models=None,
        node_model=None,
        global_model=None,
        connection_layer=None,
    ):

        super(MetaLayerSCMultigraph, self).__init__()
        self.edge_models = edge_models
        self.node_model = node_model
        self.global_model = global_model
        self.connection_layer = connection_layer
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.node_model, "reset_parameters"):
            self.node_model.reset_parameters()
        if hasattr(self.global_model, "reset_parameters"):
            self.global_model.reset_parameters()
        for item in self.edge_models:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_indices: List[Tensor],
        edge_attrs: Optional[List[Tensor]] = None,
        u: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        edge_feats: Optional[List] = None,
        node_feats: Optional[List] = None,
        global_feats: Optional[List] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        r"""
        Args:
            x (torch.Tensor): The node features.
            edge_indices (list of torch.Tensor): The edge indices.
            edge_attrs (list of torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            u (torch.Tensor, optional): The global graph features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
            edge_feats (list, optional): The edge features for skip connections
                (default: :obj:`None`)
            node_feats (list, optional): Node features for skip connections
                (default: :obj:`None`)
            global_feats (list, optional): Global features for skip connections
                (default: :obj:`None`)
        """
        if edge_attrs is None:
            edge_attrs = [None for i in self.edge_models]

        if self.edge_models is not None:
            for i, (edge_model, edge_index, edge_attr) in enumerate(
                zip(self.edge_models, edge_indices, edge_attrs)
            ):
                row = edge_index[0]
                col = edge_index[1]
                edge_attr = edge_model(
                    x[row], x[col], edge_attr, u, batch if batch is None else batch[row]
                )
                edge_attrs[i] = edge_attr

                if self.connection_layer:
                    # skip connection
                    edge_feats[i].append(edge_attr)
                    edge_attr = self.connection_layer(edge_feats[i])
                    # Update the saved feature values with the post-connection value!
                    edge_feats[i][-1] = edge_attr

        if self.node_model is not None:
            x = self.node_model(x, edge_indices, edge_attrs, u, batch)
            if self.connection_layer:
                # skip connection
                node_feats.append(x)
                x = self.connection_layer(node_feats)
                # Update the saved feature values with the post-connection value!
                node_feats[-1] = x

        if self.global_model is not None:
            u = self.global_model(x, edge_indices, edge_attrs, u, batch)
            if self.connection_layer:
                # skip connection
                global_feats.append(u)
                u = self.connection_layer(global_feats)
                # Update the saved feature values with the post-connection value!
                global_feats[-1] = u

        return x, edge_attrs, u


class NodeProcessor(Module):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):

        """
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension (for now, assume to be the same for all edge models)
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm1d', 'MessageNorm', or None
        """

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(self, x, edge_indices, edge_attrs, u=None, batch=None):
        out = [x]
        for edge_index, edge_attr in zip(edge_indices, edge_attrs):
            out.append(scatter_sum(edge_attr, edge_index[1], dim=0))

        out = cat(out, dim=-1)
        out = self.node_mlp(out)
        # out += x #residual connection

        return out


def build_graph_processor_block(
    num_edge_models=1,
    in_dim_node=128,
    in_dim_edge=128,
    hidden_dim_node=128,
    hidden_dim_edge=128,
    hidden_layers_node=2,
    hidden_layers_edge=2,
    norm_type="LayerNorm",
    connection_layer=None,
):
    """
    Builds a graph processor block for multigraphs (i.e., multiple edges between nodes)
    num_edge_models: number of edge types (equivalently, models),
    in_dim_node: input node feature dimension
    in_dim_edge: input edge feature dimension
    hidden_dim_node: number of nodes in a hidden layer for graph node processing
    hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
    hidden_layers_node: number of hidden layers for graph node processing
    hidden_layers_edge: number of hidden layers for graph edge processing
    norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
    connection_layer: A layer from skipConnections for processing various types of residuals for edge & node features

    """
    edge_models = ModuleList(
        [
            EdgeProcessor(
                in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type
            )
            for _ in range(num_edge_models)
        ]
    )
    node_model = NodeProcessor(
        in_dim_node,
        in_dim_edge * num_edge_models,
        hidden_dim_node,
        hidden_layers_node,
        norm_type,
    )

    return MetaLayerSCMultigraph(
        edge_models=edge_models,
        node_model=node_model,
        connection_layer=connection_layer,
    )


class GraphProcessor(Module):
    def __init__(
        self,
        mp_iterations=15,
        num_edge_models=1,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        norm_type="LayerNorm",
        connection_type="FullResidualConnection",
        connection_alpha=0.5,
        connection_aggregation="concat",
    ):

        """
        Graph processor

        mp_iterations: number of message-passing iterations (graph processor blocks)
        num_edge_models: number of edge types (equivalently, models),
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim_node: number of nodes in a hidden layer for graph node processing
        hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
        hidden_layers_node: number of hidden layers for graph node processing
        hidden_layers_edge: number of hidden layers for graph edge processing
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        connection_type: one of [FullResidualConnection, ResidualConnection, InitialConnection, DenseConnection]
        connection_alpha: float for Residual & Initial connection types only, can be in range [0, 1]
        connection_aggregation: for dense connection types only, can be one of [concat, maxpool, attention]
        """

        super(GraphProcessor, self).__init__()

        self.connection_type = connection_type

        connection_layer = None

        if connection_type == "FullResidualConnection":
            connection_layer = FullResidualConnection()

        elif connection_type == "ResidualConnection":
            connection_layer = ResidualConnection(alpha=connection_alpha)

        elif connection_type == "InitialConnection":
            connection_layer = InitialConnection(alpha=connection_alpha)

        elif connection_type == "DenseConnection":
            # currently dense edge connections have no contribution on loss computation!
            # for concat/attention types, this will create additional parameters. unused parameters will cause ddp errors.
            # solution 1: get rid of dense edge connections
            # solution 2: use find_unused_parameters=True in DDP. this isn't ideal since:
            #  - "This flag results in an extra traversal of the autograd graph every iteration"
            # Since dense connections occur just once at the end of the final mp:
            #  - The output of an edge dense connection won't be tied to any node features between mps
            # This means the only source of dense edge connection gradients would come from the final edge_attr, which we disgard
            #  - (we return node features "x" and edge features "edge_attr", but we only process node features "x" through the node decoder later in the model's forward pass)
            #  - (while we return edge features "edge_attr", they aren't used anywhere)

            self.dense_node_connection_layer = DenseConnection(
                in_dim=hidden_dim_node * (mp_iterations + 1),
                out_dim=hidden_dim_node,
                aggregation=connection_aggregation,
            )
        else:
            raise Exception("Invalid connection type: {0}".format(connection_type))

        self.blocks = ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(
                build_graph_processor_block(
                    num_edge_models,
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type,
                    connection_layer,
                )
            )

    def forward(self, x, edge_indices, edge_attrs):

        edge_feats = [[edge_attr] for edge_attr in edge_attrs]
        node_feats = [x]

        for block in self.blocks:
            x, edge_attrs, _ = block(
                x=x,
                edge_indices=edge_indices,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                node_feats=node_feats,
            )

            # If dense connection type, manually save mp features. Otherwise these are saved by reference within MetaLayerSC forward pass.
            if self.connection_type == "DenseConnection":
                for edge_feat, edge_attr in zip(edge_feats, edge_attrs):
                    edge_feat.append(edge_attr)
                node_feats.append(x)

        # dense skip connection at end of mp's.
        if self.connection_type == "DenseConnection":
            # edge_attr = self.dense_edge_connection_layer(edge_feats)
            x = self.dense_node_connection_layer(node_feats)

        return x, edge_attrs
