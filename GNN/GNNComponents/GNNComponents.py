
from torch import nn, cat
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum

class MLP(nn.Module):
    #MLP with LayerNorm
    def __init__(self, 
            in_dim, 
            out_dim=128, 
            hidden_dim=128,
            hidden_layers=2,
            norm_type='LayerNorm'):

        '''
        MLP

        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        normalize_output: if True, normalize output
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        '''

        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert (norm_type in ['LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm'])
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

    def __init__(self, 
            in_dim_node=128, in_dim_edge=128,
            hidden_dim=128, 
            hidden_layers=2,
            norm_type='LayerNorm'):

        '''
        Edge processor
        
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        '''

        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(2 * in_dim_node + in_dim_edge, 
            in_dim_edge, 
            hidden_dim,
            hidden_layers,
            norm_type)

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = cat([src, dest, edge_attr], -1) #concatenate source node, destination node, and edge embeddings
        out = self.edge_mlp(out)
        out += edge_attr #residual connection

        return out

class NodeProcessor(nn.Module):
    def __init__(self, 
            in_dim_node=128, in_dim_edge=128,
            hidden_dim=128, 
            hidden_layers=2,
            norm_type='LayerNorm'):

        '''
        Node processor
        
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        '''

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(in_dim_node + in_dim_edge,  
            in_dim_node,
            hidden_dim,
            hidden_layers,
            norm_type)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        row, col = edge_index
        out = scatter_sum(edge_attr, col, dim=0) #aggregate edge message by target
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        out += x #residual connection

        return out

def build_graph_processor_block(in_dim_node=128, in_dim_edge=128,
        hidden_dim_node=128, hidden_dim_edge=128, 
        hidden_layers_node=2, hidden_layers_edge=2,
        norm_type='LayerNorm'):

    '''
    Builds a graph processor block
    
    in_dim_node: input node feature dimension
    in_dim_edge: input edge feature dimension
    hidden_dim_node: number of nodes in a hidden layer for graph node processing
    hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
    hidden_layers_node: number of hidden layers for graph node processing
    hidden_layers_edge: number of hidden layers for graph edge processing
    norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None

    '''

    return MetaLayer(
            edge_model=EdgeProcessor(in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type),
            node_model=NodeProcessor(in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type)
        )

class GraphProcessor(nn.Module):
    def __init__(self, 
        mp_iterations=15, 
        in_dim_node=128, in_dim_edge=128,
        hidden_dim_node=128, hidden_dim_edge=128, 
        hidden_layers_node=2, hidden_layers_edge=2,
        norm_type='LayerNorm'):

        '''
        Graph processor

        mp_iterations: number of message-passing iterations (graph processor blocks)
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim_node: number of nodes in a hidden layer for graph node processing
        hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
        hidden_layers_node: number of hidden layers for graph node processing
        hidden_layers_edge: number of hidden layers for graph edge processing
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None

        '''

        super(GraphProcessor, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(build_graph_processor_block(in_dim_node, in_dim_edge,
                hidden_dim_node, hidden_dim_edge, 
                hidden_layers_node, hidden_layers_edge, 
                norm_type))

    def forward(self, x, edge_index, edge_attr):
        for block in self.blocks:
            x, edge_attr, _ = block(x, edge_index, edge_attr)

        return x, edge_attr        