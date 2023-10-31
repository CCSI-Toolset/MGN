from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum

from GNN.GNNComponents.skipConnections import (
    FullResidualConnection,
    ResidualConnection,
    InitialConnection,
    DenseConnection,
)
from typing import Optional, Tuple, List
from torch import Tensor


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):

        """
        MLP

        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        normalize_output: if True, normalize output
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm1d', 'MessageNorm', or None
        """

        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm1d",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


#############################

# issue with MessagePassing class:
# Only node features are updated after MP iterations
# Need to use MetaLayer to also allow edge features to update


class EdgeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):

        """
        Edge processor

        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm1d', 'MessageNorm', or None
        """

        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge,
            in_dim_edge,
            hidden_dim,
            hidden_layers,
            norm_type,
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = cat(
            [src, dest, edge_attr], -1
        )  # concatenate source node, destination node, and edge embeddings
        out = self.edge_mlp(out)
        # out += edge_attr #residual connection

        return out


class NodeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):

        """
        Node processor

        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm1d', 'MessageNorm', or None
        """

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        row, col = edge_index
        out = scatter_sum(edge_attr, col, dim=0)  # aggregate edge message by target
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        # out += x #residual connection

        return out


class MetaLayerSC(MetaLayer):
    def __init__(
        self, edge_model=None, node_model=None, global_model=None, connection_layer=None
    ):

        MetaLayer.__init__(
            self,
            edge_model=edge_model,
            node_model=node_model,
            global_model=global_model,
        )

        self.connection_layer = connection_layer

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_feats: Optional[List] = None,
        node_feats: Optional[List] = None,
        u: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """"""
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(
                x[row], x[col], edge_attr, u, batch if batch is None else batch[row]
            )

            if self.connection_layer:
                # skip connection
                edge_feats.append(edge_attr)
                edge_attr = self.connection_layer(edge_feats)
                # Update the saved feature values with the post-connection value!
                edge_feats[-1] = edge_attr

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

            if self.connection_layer:
                # skip connection
                node_feats.append(x)
                x = self.connection_layer(node_feats)
                # Update the saved feature values with the post-connection value!
                node_feats[-1] = x

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u


def build_graph_processor_block(
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
    Builds a graph processor block

    in_dim_node: input node feature dimension
    in_dim_edge: input edge feature dimension
    hidden_dim_node: number of nodes in a hidden layer for graph node processing
    hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
    hidden_layers_node: number of hidden layers for graph node processing
    hidden_layers_edge: number of hidden layers for graph edge processing
    norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm1d', 'MessageNorm', or None
    connection_layer: A layer from skipConnections for processing various types of residuals for edge & node features

    """

    return MetaLayerSC(
        edge_model=EdgeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type
        ),
        node_model=NodeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type
        ),
        connection_layer=connection_layer,
    )


class GraphProcessor(nn.Module):
    def __init__(
        self,
        mp_iterations=15,
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
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim_node: number of nodes in a hidden layer for graph node processing
        hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
        hidden_layers_node: number of hidden layers for graph node processing
        hidden_layers_edge: number of hidden layers for graph edge processing
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm1d', 'MessageNorm', or None
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

        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(
                build_graph_processor_block(
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

    def forward(self, x, edge_index, edge_attr):

        edge_feats = [edge_attr]
        node_feats = [x]

        for block in self.blocks:

            x, edge_attr, _ = block(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_feats=edge_feats,
                node_feats=node_feats,
            )

            # If dense connection type, manually save mp features. Otherwise these are saved by reference within MetaLayerSC forward pass.
            if self.connection_type == "DenseConnection":
                edge_feats.append(edge_attr)
                node_feats.append(x)

        # dense skip connection at end of mp's.
        if self.connection_type == "DenseConnection":
            x = self.dense_node_connection_layer(node_feats)

        return x, edge_attr
