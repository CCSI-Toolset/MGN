from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose, Cartesian, Distance, KNNGraph
from tqdm import tqdm
from collections import deque
import torch
import numpy as np

class PNNL_Subdomain(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        
        self.sim_range = range(1,51)
        self.time_range = range(1,501)
        
        super(PNNL_Subdomain, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['PNNL_{}_{}.pt'.format(s,t) for s in self.sim_range for t in self.time_range]

    @property
    def processed_file_names(self):
        return ['PNNL_Subdomain.pt']

    def process(self):
        
        data_list = deque([])

        if self.pre_transform is not None:
            for s in tqdm(self.sim_range):
                current_frame = None
                for t in tqdm(self.time_range):
                    next_frame = torch.load(os.path.join(self.root,'PNNL_{}_{}.pt'.format(s,t)))
                    if current_frame is None:
                        edge_index, edge_attr, node_type, pos, relevant_nodes = get_sim_level_info(next_frame)
                        next_frame = next_frame.iloc[relevant_nodes.squeeze(1)]
                    else:
                        next_frame = next_frame.iloc[relevant_nodes.squeeze(1)]
                        data_list.append(self.pre_transform(current_frame, next_frame,
                                                            edge_index, edge_attr, node_type, pos))
                    current_frame = next_frame

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def get_sim_level_info(frame):
    # assuming that every sim has a constant (x,y) location for each node, then
    # we can find the edge_attr and edge_index for eah sim's subdomain just once
    one_hot_dict = OrderedDict()
    for i, key in enumerate(sorted(frame.node_type.unique())):
        one_hot_dict[key] = i
    node_type = torch.nn.functional.one_hot(torch.tensor([one_hot_dict[t] for t in frame.node_type]))
    
    pos = torch.cat([torch.tensor(current_frame['X (m)'].values[:,np.newaxis]),
                torch.tensor(current_frame['Y (m)'].values[:,np.newaxis])],axis=1)
    
    ptg_frame = Data(pos=pos, num_nodes=max(frame.node)+1)

    # graph construction
    transforms = Compose([
        KNNGraph(6, force_undirected=True),
        Cartesian(norm=False, cat=True), 
        Distance(norm=False, cat=True)])
    ptg_frame = transforms(ptg_frame)
    
    # restrict to subdomain
    subdomain = np.logical_and(np.logical_and(-.003<=ptg_frame.pos[:,0], ptg_frame.pos[:,0]<=.004),
                                np.logical_and(.1875<=ptg_frame.pos[:,1], ptg_frame.pos[:,1]<=.2075))
    relevant_indices = torch.nonzero(subdomain)    
    
    # find nodes that have neighbors that aren't in the subdomain, they will become boundaries unless
    # they are inflow nodes
    relevant_source_node = np.isin(ptg_frame.edge_index[0,:],relevant_indices)
    irrelevant_neighbor_node = np.isin(ptg_frame.edge_index[1,:],torch.nonzero(~subdomain.bool()))
    
    relevant_nodes_with_irrelevant_edges = torch.unique(
                ptg_frame.edge_index[0,np.logical_and(relevant_source_node,irrelevant_neighbor_node)])
    
    # inflow nodes will remain inflow nodes, they don't become boundaries because of subsetting
    inflow_nodes = np.nonzero(node_type[:,1]==1)
    relevant_nodes_with_irrelevant_edges = relevant_nodes_with_irrelevant_edges[~np.isin(
                                                relevant_nodes_with_irrelevant_edges,inflow_nodes)]
        
    ptg_frame.edge_index = ptg_frame.edge_index[:,np.logical_and(relevant_source_node,~irrelevant_neighbor_node)]
    ptg_frame.edge_attr = ptg_frame.edge_attr[np.logical_and(relevant_source_node,~irrelevant_neighbor_node)]
    
    # restrict vars to relevant subsets, and update node definition for nodes that are now boundaries (due
    # to subsetting)
    pos = pos[relevant_indices].squeeze()
    
    node_type = torch.cat([node_type, torch.zeros(len(node_type)).view(-1,1)],axis=1)
    node_type[relevant_nodes_with_irrelevant_edges,:] = torch.zeros_like(
                                            node_type[relevant_nodes_with_irrelevant_edges,:])
    node_type[relevant_nodes_with_irrelevant_edges,-1] = 1
    node_type = node_type[relevant_indices].squeeze()
    
    return ptg_frame.edge_index, ptg_frame.edge_attr, node_type, pos, relevant_indices
        
def pre_transform(current_frame, next_frame,
                 edge_index, edge_attr, node_type, pos):
    
    x = torch.cat([torch.tensor(current_frame['Velocity[i] (m/s)'].values[:,np.newaxis]),
                torch.tensor(current_frame['Velocity[j] (m/s)'].values[:,np.newaxis]),
                torch.tensor(current_frame['Volume Fraction of Liq'].values[:,np.newaxis]),
                node_type],axis=1)
    
    def get_delta(next_frame, current_frame, var):
        return (next_frame[var].values[:,np.newaxis] - 
                current_frame[var].values[:,np.newaxis])

    # y attribute is [num_nodes, *] (these are from Section A.1 of https://arxiv.org/pdf/2010.03409.pdf)
    y = torch.cat([torch.tensor(get_delta(next_frame,current_frame,'Velocity[i] (m/s)')),
                    torch.tensor(get_delta(next_frame,current_frame,'Velocity[j] (m/s)')),
                    torch.tensor(get_delta(next_frame,current_frame,'Volume Fraction of Liq')),
                    torch.tensor(next_frame['Pressure (Pa)'].values[:,np.newaxis])], axis=1)
    
    ptg_frame = Data(x=x, y=y, pos=pos, num_nodes= len(x),
                    edge_index= edge_index, edge_attr= edge_attr)
    
    return ptg_frame