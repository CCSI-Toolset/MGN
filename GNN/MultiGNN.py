from torch.nn import Module, ModuleList

from .GNNComponents.MultiGNNComponents import MLP, GraphProcessor


class GNN(Module):
    # default values based on MeshGraphNets paper/supplement
    def __init__(
        self,
        # data attributes:
        in_dim_node,  # includes data window, node type, inlet velocity
        in_dim_edges,  # distance and relative coordinates
        out_dim,  # includes x-velocity, y-velocity, volume fraction, pressure (or a subset)
        num_edge_types=1,  # number of edge types (for example, mesh edges and real-world edges)
        # encoding attributes:
        out_dim_node=128,
        out_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        # graph processor attributes:
        mp_iterations=15,
        hidden_dim_processor_node=128,
        hidden_dim_processor_edge=128,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
        # decoder attributes:
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
        output_type="acceleration",
        connection_type="FullResidualConnection",
        connection_alpha=0.5,
        connection_aggregation="concat",
        graph_processor_type="Original",
        # other:
        **kwargs
    ):

        super(GNN, self).__init__()

        if isinstance(in_dim_edges, int):
            in_dim_edges = [in_dim_edges for _ in range(num_edge_types)]
        assert (
            len(in_dim_edges) == num_edge_types
        ), "Expected %s in_dim_edges, but found %s" % (
            len(num_edge_types),
            in_dim_edges,
        )
        self.num_edge_types = num_edge_types

        self.node_encoder = MLP(
            in_dim_node,
            out_dim_node,
            hidden_dim_node,
            hidden_layers_node,
            mlp_norm_type,
        )
        self.edge_encoders = ModuleList(
            [
                MLP(
                    in_dim_edge,
                    out_dim_edge,
                    hidden_dim_edge,
                    hidden_layers_edge,
                    mlp_norm_type,
                )
                for in_dim_edge in in_dim_edges
            ]
        )

        if graph_processor_type == "Original":
            self.graph_processor = GraphProcessor(
                mp_iterations,
                num_edge_types,
                out_dim_node,
                out_dim_edge,
                hidden_dim_processor_node,
                hidden_dim_processor_edge,
                hidden_layers_processor_node,
                hidden_layers_processor_edge,
                mlp_norm_type,
                connection_type,
                connection_alpha,
                connection_aggregation,
            )
        else:
            raise Exception(
                "Invalid Graph Processor Type: {0}".format(graph_processor_type)
            )

        self.node_decoder = MLP(
            out_dim_node, out_dim, hidden_dim_decoder, hidden_layers_decoder, None
        )
        self.output_type = output_type

    # graph: torch_geometric.data.Data object with the following attributes:
    #       x: node x feature array (volume fraction, pressure, node type, inlet velocity, etc.)
    #       edge_index: 2 x edge array
    #       edge_attr: edge x feature matrix (distance, relative coordinates)

    def forward(self, graph):
        if self.num_edge_types == 1:
            edge_indices = [graph.edge_index]
            edge_attrs = [graph.edge_attr]
        else:
            edge_indices = [
                graph["edge_index_%s" % i] for i in range(self.num_edge_types)
            ]
            edge_attrs = [
                graph.get("edge_attr_%s" % i) for i in range(self.num_edge_types)
            ]
        out = self.node_encoder(graph.x)
        edge_attrs = [
            edge_encoder(edge_attr)
            for edge_encoder, edge_attr in zip(self.edge_encoders, edge_attrs)
        ]
        out, _ = self.graph_processor(out, edge_indices, edge_attrs)
        out = self.node_decoder(
            out
        )  # paper: corresponds to velocity or acceleration at this point; loss is based on one of these, not the actual state

        return out

    # # implement these in subclasses:

    # def update_state(mgn_output_np,
    #         current_state=None, previous_state=None,
    #         source_data=None):
    #     pass

    # def rollout(device,
    #         initial_window_data,
    #         graph,
    #         node_types,
    #         node_coordinates,
    #         onehot_info = None,
    #         inlet_velocity=None, rollout_iterations=1,
    #         source_data=None):
    #     pass
