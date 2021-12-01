
import os
import pandas as pd
import numpy as np
import pickle
from glob import glob

from torch import from_numpy, no_grad

from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RadiusGraph, Cartesian, Distance, Compose, KNNGraph, Delaunay, ToUndirected
from torch_geometric.utils import to_networkx

from networkx import is_weakly_connected
from warnings import warn

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import numpy as np

from ..utils.mgn_utils import process_node_window, get_sample

class AirFoilDataset(Dataset):

        '''
        Airfoil dataset (using one simulation)
        This class should only be used with the dataset (fn=...) below.

        '''
    def __init__(self, 
        fn='/data/ccsi/airfoil/sim.pkl',
        output_type='velocity', #acceleration, velocity, state
        window_length=5, 
        noise=None,
        noise_gamma=0.1,
        apply_onehot=False, 
        boundary_nodes=[1], #list of integer node types corresponding to boundaries
        source_nodes=[2], #list of integer node types corresponding to sources
        normalize=True,
        **kwargs):

        self.output_type = output_type

        assert (output_type != 'acceleration') or (window_length >= 2)
        self.window_length = window_length

        self.fn = fn
        with open(fn, 'rb') as fid:
            D = pickle.load(fid)
        
        columns = D['columns']
        t = D['t']
        x = D['x']
        y = D['y']
        edges = D['edges'] #note: there are duplicate edges 
        edges = np.unique(edges, axis=0)

        keep_columns = ['Momentum_x', 'Momentum_y', 'Density', 'Pressure']
        keep_columns = [columns.index(i) for i in keep_columns]
        data = D['data']
        data = data[:,:,keep_columns]

        if normalize:
            self.mean = np.mean(data, axis=(0,1))
            self.std = np.std(data, axis=(0,1))
            data = (data - self.mean) / self.std        

        # Find the boundary nodes
        node_coordinates = np.vstack([x,y]).T
        nodeDist = cdist(node_coordinates,np. zeros((1,2)))
        source_inds = np.where(nodeDist > nodeDist.max() - 1e-2)[0]

        # find the air_foil nodes
        xx = np.linspace(0, 1, 10000)
        tt = .12
        yy = 5 * tt * (.2969 * np.sqrt(xx) - .1260 * xx - .3516 * xx ** 2 + .2843 * xx ** 3 - .1015 * xx ** 4)
        airfoil_cords = np.vstack([xx, yy]).T
        e = cdist(node_coordinates, airfoil_cords)

        v = np.argmin(e, axis=0)
        v = np.unique(v)
        airfoil_cords[:,1] = -airfoil_cords[:,1]
        e = cdist(node_coordinates, airfoil_cords)
        w = np.argmin(e, axis=0)
        w = np.unique(w)
        foil_inds = list(set(v).union(set(w)))
        foil_inds = list(set(v).union(set(w)))
        foil_inds = list(set(np.where(x < .9895)[0]).intersection(set(foil_inds)))
        
        xx = np.linspace(.99,1.,1000)
        dx = .01
        dy = .0013 
        m = dy / dx
        yy = m * (xx - 1) 

        airfoil_cords = np.vstack([xx, yy]).T
        e = cdist(node_coordinates, airfoil_cords)
        v = np.argmin(e, axis=0)
        v = np.unique(v)

        airfoil_cords = np.vstack([xx,-yy]).T
        e = cdist(node_coordinates,airfoil_cords)
        w = np.argmin(e,axis=0)
        w = np.unique(w)
        tail_inds = list(set(v).union(set(w)))

        foil_inds = list(set(foil_inds).union(set(tail_inds)))
        
        #save data, node types
        self.data = data
        self.node_types = np.zeros((len(x), 1), dtype='int')
        self.node_types[foil_inds] = boundary_nodes[0]
        self.node_types[source_inds] = source_nodes[0]

        #indices of boundary/source nodes for this class, since there is only one simulation
        self.boundary_nodes = foil_inds
        self.source_nodes = source_inds

        #onehot
        apply_onehot_ = (np.min(self.node_types) >= 0)
        onehot_dim = -1 if not apply_onehot_ else (np.max(self.node_types) + 1)

        if apply_onehot and not apply_onehot_:
            raise Exception(fn + ': cannot apply one-hot encoding')

        self.onehot_dim = onehot_dim
        self.apply_onehot = apply_onehot

        #graph construction
        transforms = [ToUndirected(),
            Cartesian(norm=False, cat=True), 
            Distance(norm=False, cat=True)]
        
        graph = Data(pos=from_numpy(node_coordinates.astype(np.float32)),
                     edge_index=from_numpy(edges.T),
                     adj_t=None) #needed for ToUndirected
        
        transforms = Compose(transforms)
        graph = transforms(graph)

        # #remove other-to-source edges #keep now?
        # sources = np.where(np.isin(self.node_types[-1].flatten(), source_nodes))[0]
        # drop_edges = np.isin(graph.edge_index[1].numpy(), sources)
        # graph.edge_index = graph.edge_index[:, ~drop_edges].contiguous()
        # graph.edge_attr = graph.edge_attr[~drop_edges].contiguous()

        if not is_weakly_connected(to_networkx(graph)):
            warn(fn + ': disconnected graph')

        self.graph = graph

        self.dataset_length = np.array(self.data.shape[0] - self.window_length, dtype=np.int64)
        self.output_dim = self.data.shape[-1]

        self.noise = noise #to do: check none or length==output_dim
        self.noise_gamma = noise_gamma

        #update function; update momentum based on predicted change, predict pressure
        def update_function(mgn_output_np, output_type, 
                current_state=None, previous_state=None,
                source_data=None):

            num_states = current_state.shape[-1]

            with no_grad():
                if output_type == 'acceleration':
                    assert current_state is not None
                    assert previous_state is not None
                    next_state = np.concatentate([2 * current_state - previous_state, np.zeros((len(current_state), 1))], axis=1) + mgn_output_np
                elif output_type == 'velocity':
                    assert current_state is not None
                    next_state = np.concatenate([current_state, np.zeros((len(current_state), 1))], axis=1) + mgn_output_np
                else: #state 
                    next_state = mgn_output_np.copy()

                if type(source_data) is dict:
                    for key in source_data:
                        next_state[key, :num_states] = source_data[key]
                elif type(source_data) is tuple:
                    next_state[source_data[0], :num_states] = source_data[1]
                # else: warning?

            return next_state

        self.update_function = update_function 

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):

        node_data, outputs = get_sample(self.data[:,:,:-1], #only momentum/density
            self.source_nodes,
            idx, self.window_length, self.output_type, self.noise, self.noise_gamma)

        node_data = from_numpy(process_node_window(
                node_data, self.graph.pos.numpy(), 
                self.node_types, 
                self.apply_onehot, self.onehot_dim).astype(np.float32))

        outputs = np.concatenate([outputs, self.data[idx + self.window_length,:,[-1]].reshape((-1, 1))], axis=1) #add pressure back
        outputs = from_numpy(outputs.astype(np.float32))

        graph = Data(x=node_data, edge_index=self.graph.edge_index, edge_attr=self.graph.edge_attr, y=outputs, num_nodes=len(node_data))


        return graph
