import os
import pandas as pd
import numpy as np
import pickle
from glob import glob

from numpy import concatenate

from torch import from_numpy, no_grad

from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RadiusGraph, Cartesian, Distance, Compose, KNNGraph, Delaunay, ToUndirected
from torch_geometric.utils import to_networkx

from networkx import is_weakly_connected
from warnings import warn

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from ..utils.mgn_utils import process_node_window, get_sample

def get_comsol_data(fn = '/data/ccsi/cylinder_flow/cylinder_flow_comsol.csv'):

    '''
    Preprocesses COMSOL cylinder flow simulation output.

    '''

    D = pd.read_csv(fn)
    x = D['x']
    y = D['y']
    D = D.drop(columns=['x','y'])

    X = D.values

    inds = np.arange(0,X.shape[1],4)
    times = X[:,inds]
    t = times[0]

    inds = np.arange(1,X.shape[1],4)
    vel_x = X[:,inds]

    inds = np.arange(2,X.shape[1],4)
    vel_y = X[:,inds]

    inds = np.arange(3,X.shape[1],4)
    p = X[:,inds]

    return x, y, t, vel_x, vel_y, p

def get_comsol_edges(node_coordinates, mesh_file = '/data/ccsi/cylinder_flow/mesh_comsol_output.txt'):

    '''
    Preprocesses COMSOL cylinder flow mesh

    This function is necessary because the node coordinates and comsol mesh are in a different order
    Need to re-order the edge list from the mesh file
    '''

    def splitFloatLine(line):
        return list(map(float, line.split()[:2]))

    def splitElementLine(line):
        return list(map(int,line.split()[:3]))

    def simplexToEdgeList(simp):
        edges = [(simp[0], simp[1]), (simp[1], simp[2]), (simp[2], simp[0])]
        r_edges = [(e[1],e[0]) for e in edges]
        return edges + r_edges

    with open(mesh_file) as fid:
        mesh = fid.readlines()

    #get nodes
    nodeLine = mesh[4]
    numNodes = int(nodeLine.split()[2])
    mesh_nodes = mesh[10:(10+numNodes)]
    mesh_nodes = np.array(list(map(splitFloatLine, mesh_nodes)))

    #get mesh elements
    mesh_elements = mesh[11+numNodes:]
    mesh_elements = np.array(list(map(splitElementLine, mesh_elements)))
    mesh_elements = mesh_elements - 1 # comsol starts from 1 not 0.

    #match mesh and node coordinates
    Y = cdist(mesh_nodes, node_coordinates)
    index = np.argmin(Y, axis=1)
    simplex = index[mesh_elements]
    
    A = list(map(simplexToEdgeList, simplex))
    edge_list = [b for sublist in A for b in sublist]
    edge_list = np.unique(edge_list,axis=1)
    
    return edge_list

class CylinderFlowDataset2(Dataset):

    # diff from CylinderFlowDataset:
    # pressure no longer an input
    # added update_function; output_type only affects v_x, v_y, and pressure is always predicted

    '''
    CylinderFlow dataset (using one simulation)
    Pressure is not included as an input
    output_type only affects v_x and v_y; pressure is always predicted

    '''

    def __init__(self, 
        fn = '/data/ccsi/cylinder_flow/cylinder_flow_comsol.csv',
        mesh_file = '/data/ccsi/cylinder_flow/mesh_comsol_output.txt',
        center = [0.2, 0.2], 
        R = 0.05, 
        output_type='velocity', #acceleration, velocity, state
        window_length=1, 
        noise=None,
        noise_gamma=0.1,
        apply_onehot=False, 
        boundary_nodes=[1, 3], #list of integer node types corresponding to boundaries
        source_nodes=[2], #list of integer node types corresponding to sources
        normalize=True,
        **kwargs): 

        self.output_type = output_type

        assert (output_type != 'acceleration') or (window_length >= 2)
        self.window_length = window_length

        #read/store dataset into class
        self.fn = fn
        x, y, t, vel_x, vel_y, p = get_comsol_data(fn)
        data = []
        for i in range(len(t)):
            data.append([vel_x[:,i],vel_y[:,i],p[:,i]])
        data = np.array(data, dtype=np.float32)
        data = np.rollaxis(data,2,1)

        #normalize data;
        #for larger datasets, replace with online normalization
        if normalize:
            self.mean = np.mean(data, axis=(0,1))
            self.std = np.std(data, axis=(0,1))
            data = (data - self.mean) / self.std
        
        # Find the boundary nodes
        node_coordinates = np.vstack([x,y]).T
        mn, mx = np.min(node_coordinates, axis=0), np.max(node_coordinates, axis=0)
        source_inds = np.where(node_coordinates[:,0] == mn[0])[0]
        bottom_inds = np.where(node_coordinates[:,1] == mn[1])[0]
        top_inds = np.where(node_coordinates[:,1] == mx[1])[0]
        right_inds = np.where(node_coordinates[:,0] == mx[0])[0]
        non_source_boundary_inds = set(bottom_inds).union(set(right_inds)).union(set(top_inds)).difference(source_inds)
        
        #cylinder
        center = np.array(center).reshape(1,2)
        distFromCircleCenter = cdist(node_coordinates,center)
        
        interior_boundary_inds = np.where(distFromCircleCenter <= R)[0]
        boundary_inds = sorted(list(non_source_boundary_inds.union(interior_boundary_inds)))
        
        #save data, node types
        self.data = data
        self.node_types = np.zeros((len(x),1), dtype='int')
        self.node_types[boundary_inds] = boundary_nodes[0] #top, bottom, interior
        self.node_types[right_inds] = boundary_nodes[1] #right #another 'source'?
        self.node_types[source_inds] = source_nodes[0]

        #indices of boundary/source nodes for this class, since there is only one simulation
        self.boundary_nodes = boundary_inds
        self.source_nodes = source_inds

        #one-hot
        apply_onehot_ = (np.min(self.node_types) >= 0) #check if one-hot encoding can be applied
        onehot_dim = -1 if not apply_onehot_ else (np.max(self.node_types) + 1)

        if apply_onehot and not apply_onehot_:
            raise Exception(filename + ': cannot apply one-hot encoding')

        self.onehot_dim = onehot_dim
        self.apply_onehot = apply_onehot

        #graph construction
        transforms = [
            Cartesian(norm=False, cat=True), 
            Distance(norm=False, cat=True)]

        edge_list = get_comsol_edges(node_coordinates, mesh_file)
        
        graph = Data(pos=from_numpy(node_coordinates.astype(np.float32)),
                     edge_index=from_numpy(edge_list.T))
        
        transforms = Compose(transforms)
        graph = transforms(graph)

        # #remove other-to-source edges #keep?
        # sources = np.where(np.isin(self.node_types[-1].flatten(), source_nodes))[0]
        # drop_edges = np.isin(graph.edge_index[1].numpy(), sources)
        # graph.edge_index = graph.edge_index[:, ~drop_edges].contiguous()
        # graph.edge_attr = graph.edge_attr[~drop_edges].contiguous()

        if not is_weakly_connected(to_networkx(graph)):
            warn(filename + ': disconnected graph')

        self.graph = graph

        self.dataset_length = np.array(self.data.shape[0] - self.window_length, dtype=np.int64)
        self.output_dim = self.data.shape[-1]

        self.noise = noise #to do: check none or length==output_dim
        self.noise_gamma = noise_gamma

        #update function; update momentum based on predicted change ('velocity'), predict pressure
        def update_function(mgn_output_np, output_type, 
                current_state=None, previous_state=None,
                source_data=None):

            num_states = current_state.shape[-1]

            with no_grad():
                if output_type == 'acceleration':
                    assert current_state is not None
                    assert previous_state is not None
                    next_state = np.concatentate([2 * current_state - previous_state, np.zeros((len(current_state),1))], axis=1) + mgn_output_np
                elif output_type == 'velocity':
                    assert current_state is not None
                    next_state = np.concatenate([current_state, np.zeros((len(current_state),1))], axis=1) + mgn_output_np
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

        node_data, outputs = get_sample(self.data[:,:,:-1], #only momentum (velocity) 
            self.source_nodes,
            idx, self.window_length, self.output_type, self.noise, self.noise_gamma)

        node_data = from_numpy(process_node_window(
                node_data, self.graph.pos.numpy(), 
                self.node_types, 
                self.apply_onehot, self.onehot_dim).astype(np.float32))

        outputs = concatenate([outputs, self.data[idx + self.window_length,:,[-1]].reshape((-1, 1))], axis=1) #add pressure back
        outputs = from_numpy(outputs.astype(np.float32))

        graph = Data(x=node_data, edge_index=self.graph.edge_index, edge_attr=self.graph.edge_attr, y=outputs, num_nodes=len(node_data))

        return graph
