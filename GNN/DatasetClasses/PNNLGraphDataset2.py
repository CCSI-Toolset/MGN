
import os
import pandas as pd
import numpy as np
import pickle
from glob import glob

import torch

# from torch_geometric.utils import to_undirected, sort_edge_index
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RadiusGraph, Cartesian, Distance, Compose, KNNGraph, Delaunay, ToUndirected
from torch_geometric.utils import to_networkx

from networkx import is_weakly_connected
from warnings import warn

from ..utils.mgn_utils import process_node_window, get_sample

class PNNLGraphDataset2(Dataset):

    def __init__(self, 
        root_dir,
        output_type='acceleration', #acceleration, velocity, state
        window_length=5, 
        graph_type = 'radius', # radius, knn, delaunay
        radius=0.1,
        k=10,
        noise=None,
        noise_gamma=0.1,
        apply_onehot=False, 
        boundary_nodes=[1], #list of integer node types corresponding to boundaries
        source_nodes=[2], #list of integer node types corresponding to sources
        **kwargs): 
        # root_dir:
        #   files in this directory
        #        *-simulation.pkl: one file per simulation; contains a dict with 
        #            simulation:  time x node x feature np array; simulations must have the same number of features
        #            coordinates: node x dimension np array
        #            node_types: node x 1 integer np array #future work: or node x encoding dim np array
        #                if apply_onehot, one-hot encoding will be applied
        #            edge_index: optional; contains a 2 x edge np array with type long (an adjacency list). 
        #                Undirected edges should appear twice (a[0,i] == a[1,j] and a[0,j] == a[1,i] for some i,j).
        #                If not available, then a graph will be constructed based on the coordinates
        #            inlet_velocity: optional; float # future work: generalize this to additional simulation information 

        self.output_type = output_type

        assert (output_type != 'acceleration') or (window_length >= 2)
        self.window_length = window_length

        pkl_files = glob(os.path.join(root_dir,'*-simulation.pkl'))
        
        self.datasets = []
        self.node_types = []
        self.boundary_nodes = boundary_nodes
        self.source_nodes = source_nodes
        node_type_dim = None

        self.graphs = []
        self.inlet_velocities = [] 

        for filename in pkl_files:
            with open(filename, 'rb') as f:
                sim_data = pickle.load(f)
            self.datasets.append(sim_data['simulation'])
            self.node_types.append(sim_data['node_types'])
            if not np.issubdtype(self.node_types[-1].dtype, np.integer):
                raise Exception(filename +': node_types not subtype of np.integer')

            node_coordinates = sim_data['coordinates']

            #check if one-hot is possible, save maximum integer + 1
            if node_type_dim is None:
                node_type_dim = self.node_types[-1].shape[-1] 
                apply_onehot_ = (node_type_dim == 1) 
                onehot_dim = -1 if not apply_onehot_ else np.max(self.node_types[-1]) + 1
            elif node_type_dim != self.node_types[-1].shape[-1]:
                raise Exception(filename + ': different node type encoding dimension')

            apply_onehot_ = (np.min(self.node_types[-1]) >= 0)
            onehot_dim = -1 if not apply_onehot_ else max(onehot_dim, np.max(self.node_types[-1]) + 1)

            if apply_onehot and not apply_onehot_:
                raise Exception(filename + ': cannot apply one-hot encoding')

            #graph construction
            transforms = [
                Cartesian(norm=False, cat=True), 
                Distance(norm=False, cat=True)]

            #edge construction
            if 'edge_index' in sim_data.keys():
                graph = Data(pos=torch.from_numpy(node_coordinates), 
                    edge_index=torch.from_numpy(edge_index))
            elif graph_type in ['radius', 'knn', 'delaunay']:
                if graph_type == 'radius':
                    transforms.insert(0, RadiusGraph(radius, max_num_neighbors=len(node_coordinates))) #setting max_num_neighbors to num nodes guarantees undirected
                elif graph_type == 'knn':
                    transforms.insert(0, KNNGraph(k, force_undirected=True))
                else:
                    # transforms.insert(0, FaceToEdge(False))
                    # transforms.insert(0, Delaunay()) 
                    transforms = [Delaunay()]
                graph = Data(pos=torch.from_numpy(node_coordinates))
            else:
                raise Exception('invalid graph type')

            transforms = Compose(transforms)
            graph = transforms(graph)

            #additional graph refinements
            #1) radius: handle thresholding problem
            #2) delaunay: process faces/simplices, remove interior obstacle edges (done after figuring out if one-hot is necessary)
            #3) all: remove other-to-source edges #future work: outlet nodes?

            if 'edge_index' not in sim_data.keys():
                if graph_type == 'radius':
                    drop_edges = graph.edge_attr[:,-1].numpy() > radius #distance is last in list of transforms
                    graph.edge_index = graph.edge_index[:, ~drop_edges].contiguous()
                    graph.edge_attr = graph.edge_attr[~drop_edges].contiguous()
                elif graph_type == 'delaunay':
                    def _getRowEdges(r):
                        L = [(r[0],r[1]), (r[1],r[2]), (r[2],r[0])]
                        return L
                    def _getDelaunayEdges(simplices, boundary_inds):
                        if boundary_inds != []:
                            S = set(boundary_inds)
                            kept_simplices = []
                            for t in simplices:
                                differ = list(set(t).difference(S))
                                if len(differ) > 0:
                                    kept_simplices.append(t)
                            simplices = np.array(kept_simplices)
                        #could simplify this by using FaceToEdge instead
                        edges = list(map(lambda x: _getRowEdges(x), simplices))
                        edges = [i for sublist in edges for i in sublist]
                        return np.array(edges).T
                    boundary_inds = np.where(np.isin(self.node_types[-1].flatten(), boundary_nodes))[0]
                    edges = _getDelaunayEdges(graph.face.numpy().T, boundary_inds)
                    graph.edge_index = torch.tensor(edges)
                    graph.adj_t = None #problem in earlier versions of pt geometric
                    transforms = Compose([ToUndirected(), Cartesian(norm=False, cat=True), Distance(norm=False, cat=True)])
                    graph = transforms(graph)
                    graph.face = None

                #remove other-to-source edges
                sources = np.where(np.isin(self.node_types[-1].flatten(), source_nodes))[0]
                drop_edges = np.isin(graph.edge_index[1].numpy(), sources)
                graph.edge_index = graph.edge_index[:, ~drop_edges].contiguous()
                graph.edge_attr = graph.edge_attr[~drop_edges].contiguous()

            if not is_weakly_connected(to_networkx(graph)):
                warn(filename + ': disconnected graph')

            self.graphs.append(graph)   

            #inlet velocity
            if 'inlet_velocity' in sim_data.keys():
               self.inlet_velocities.append(sim_data['inlet_velocity'])
            elif self.inlet_velocities: #another simulation had an inlet velocity
                raise Exception(filename + ': missing inlet velocity')

        dataset_lengths = np.array([ds.shape[0] - self.window_length for ds in self.datasets], dtype=np.int64)
        self.bins = np.cumsum(dataset_lengths)
        self.output_dim = self.datasets[0].shape[-1]

        self.noise = noise #to do: check none or length==output_dim
        self.noise_gamma = noise_gamma

        #one-hot encoding info
        self.onehot_dim = onehot_dim
        self.apply_onehot = apply_onehot

    def __len__(self):
        return self.bins[-1]

    def __getitem__(self, idx):

        #to do: modify for sources

        #get simulation
        bin = np.digitize(idx, self.bins)
        dataset_idx = idx - (self.bins[bin - 1] if bin > 0 else 0)

        #get inlet velocity, if given
        inlet_velocity = self.inlet_velocities[bin] if self.inlet_velocities else None  

        source_node_idx = np.where(np.isin(self.node_types[bin], self.source_nodes))[0]
        node_data, outputs = get_sample(self.datasets[bin], source_node_idx,
            dataset_idx, self.window_length, self.output_type, self.noise, self.noise_gamma)

        #process data
        node_data = torch.from_numpy(process_node_window(
                node_data, self.graphs[bin].pos.numpy(), 
                self.node_types[bin], 
                self.apply_onehot, self.onehot_dim,
                inlet_velocity).astype(np.float32))
        outputs = torch.from_numpy(outputs.astype(np.float32))

        graph = Data(x=node_data, edge_index=self.graphs[bin].edge_index, edge_attr=self.graphs[bin].edge_attr, y=outputs, num_nodes=len(node_data))

        return graph