from torch_geometric.data import Dataset
import pickle
import numpy as np
import pandas as pd
import torch
import os.path as osp

class PNNL_With_Node_Type(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, saved_dict_loc=None):
        
        self.n_nodes = None #number of nodes in the graph
        self.sim_range = range(1,51)
        self.time_range = range(1,501)
        
        # all the nodes with constant y velocity for a given sim / inlet velocity
        # it comes from the 02_determine_node_types.ipynb file
        self.const_nodes_by_sim = pickle.load(open('/data/ccsi/processed/const_nodes_by_sim.pkl','rb'))
        
        # some dictionaries for tracking statistics about graph data, they get built during processing or 
        # by running self.check_processed_data(True)
        self.min_yvel_by_node_type = {'fluid':1e10, 'side_wall':1e10, 'outflow_bottom':1e10, 
                                     'outflow_top':1e10, 'inflow':1e10, 'packing':1e10}
        self.max_yvel_by_node_type = {'fluid':-1e10, 'side_wall':-1e10, 'outflow_bottom':-1e10, 
                                     'outflow_top':-1e10, 'inflow':-1e10, 'packing':-1e10}
        self.min_xvel_by_node_type = {'fluid':1e10, 'side_wall':1e10, 'outflow_bottom':1e10, 
                                     'outflow_top':1e10, 'inflow':1e10, 'packing':1e10}
        self.max_xvel_by_node_type = {'fluid':-1e10, 'side_wall':-1e10, 'outflow_bottom':-1e10, 
                                     'outflow_top':-1e10, 'inflow':-1e10, 'packing':-1e10}
        self.min_pres_by_node_type = {'fluid':1e10, 'side_wall':1e10, 'outflow_bottom':1e10, 
                                     'outflow_top':1e10, 'inflow':1e10, 'packing':1e10}
        self.max_pres_by_node_type = {'fluid':-1e10, 'side_wall':-1e10, 'outflow_bottom':-1e10, 
                                     'outflow_top':-1e10, 'inflow':-1e10, 'packing':-1e10}
        self.max_frac_by_node_type = {'fluid':-1e10, 'side_wall':-1e10, 'outflow_bottom':-1e10, 
                                     'outflow_top':-1e10, 'inflow':-1e10, 'packing':-1e10}       
        
        if saved_dict_loc!=None:
            for i, old_dict in enumerate([self.min_yvel_by_node_type, self.max_yvel_by_node_type,
                  self.min_xvel_by_node_type, self.max_xvel_by_node_type,
                  self.min_pres_by_node_type, self.max_pres_by_node_type,
                  self.max_frac_by_node_type]):
                new_dict = pickle.load(open('{}_{}.pkl'.format(saved_dict_loc,i),'rb'))
                for k, v in new_dict.items():
                    old_dict[k] = v
        
        # inlet velocities for each sim
        vel_file = '/data/ccsi/pnnl_liquid_inlet/liquid_inlet_velocity.txt'
        with open(vel_file) as fid:
            vels = fid.read().splitlines()
        self.inletVelocity = np.array(list(map(float,vels[1:])))   
        
        super(PNNL_With_Node_Type, self).__init__(root, transform, pre_transform)
        

    @property
    def raw_file_data(self):
        files = OrderedDict()
        self.n_files = 0
        for sim in self.sim_range:
            for time in self.time_range:
                files[self.n_files] = {}
                files[self.n_files]['path'] = '/data/ccsi/pnnl_liquid_inlet/data/0{}/XYZ_Internal_Table_table_{}0.csv'.format(str(sim).zfill(2),  str(time))
                files[self.n_files]['sim'] = sim
                files[self.n_files]['time'] = time
                self.n_files+=1
        return files

    @property
    def raw_file_names(self):
        paths = []
        for sim in self.sim_range:
            for time in self.time_range:
                paths.append('/data/ccsi/pnnl_liquid_inlet/data/0{}/XYZ_Internal_Table_table_{}0.csv'.format(
                        str(sim).zfill(2),  str(time)))
        return paths
    
    def download(self):
        # files already downloaded
        return
        
    @property
    def processed_file_names(self):
        return ['PNNL_{}_{}.pt'.format(s,t) for s in self.sim_range for t in self.time_range]
        

    def process(self):
        i = 0
        for frame in tqdm(self.raw_file_data.values()):
            # sort data by x, then y; confirm all frames have the same # of mesh points m
            data = pd.read_csv(frame['path'])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(self, data, frame['sim'], frame['time'])

            torch.save(data, osp.join(self.processed_dir, 'PNNL_{}_{}.pt'.format(frame['sim'],frame['time'])))
            i += 1

        for i, dic in enumerate([self.min_yvel_by_node_type, self.max_yvel_by_node_type,
              self.min_xvel_by_node_type, self.max_xvel_by_node_type,
              self.min_pres_by_node_type, self.max_pres_by_node_type,
              self.max_frac_by_node_type]):
            pickle.dump(dic, open('/data/ccsi/processed/value_dict_{}.pkl'.format(i),'wb'))
            

        self.check_processed_data()
        
        
    def check_processed_data(self, by_frame_first=False):
    
        # you want this option if the dictionaries like self.min_yvel_by_node_type aren't 
        # updated in this instance yet
        if by_frame_first:
            for sim in tqdm(self.sim_range):
                for time in tqdm(self.time_range):
                    data = torch.load(osp.join(self.processed_dir, 'PNNL_{}_{}.pt'.format(sim,time)))
                    self.check_vals_in_frame(data, sim)
                    
                print(self.min_yvel_by_node_type, self.max_yvel_by_node_type,
                      self.min_xvel_by_node_type, self.max_xvel_by_node_type,
                      self.min_pres_by_node_type, self.max_pres_by_node_type,
                      self.max_frac_by_node_type)    
            
        # running this assumes the dictionaries like self.min_yvel_by_node_type have been updated
        # either through the initial processing of the raw data or the preceding loop
        self.check_vals_across_frames()
            
        
    def check_vals_across_frames(self):
        # check node values using information from across all frames
        
        # every node type, except for outflow_top, can reach volume fraction of 1
        for node_type in ['fluid', 'side_wall', 'outflow_bottom', 'inflow', 'packing', 'outflow_top']:
            assert self.max_frac_by_node_type[node_type] == 1, (node_type, 'frac', self.max_frac_by_node_type[node_type])
            
        # these nodes' velocities can be large
        for node_type in ['fluid', 'outflow_bottom']:
            assert self.min_yvel_by_node_type[node_type] < -1, (node_type, 'min yvel', self.min_yvel_by_node_type[node_type])
            assert self.max_yvel_by_node_type[node_type] > 1, (node_type, 'max yvel', self.min_yvel_by_node_type[node_type])
            assert self.min_xvel_by_node_type[node_type] < -1, (node_type, 'min xvel', self.min_yvel_by_node_type[node_type])
            assert self.max_xvel_by_node_type[node_type] > 1, (node_type, 'max xvel', self.min_yvel_by_node_type[node_type])
        
        # these nodes' velocities are always *almost* zero in x direction...
        for node_type in ['outflow_top', 'inflow']:
            assert self.min_xvel_by_node_type[node_type] > -1e-12, (node_type, 'min xvel', self.min_yvel_by_node_type[node_type])
            assert self.max_xvel_by_node_type[node_type] < 1e-12, (node_type, 'max xvel', self.min_yvel_by_node_type[node_type])
        
        # but in the y direction, they have special properties, enabling scripting of these nodes
        assert self.max_yvel_by_node_type['outflow_top'] == self.min_yvel_by_node_type['outflow_top'] == 0.01963862, (self.min_yvel_by_node_type['outflow_top'])
        assert self.max_yvel_by_node_type['inflow'] == -0.002, ('inflow', 'yvel max')
        assert self.min_yvel_by_node_type['inflow'] == -0.0218, ('inflow', 'yvel min')
            
        # these nodes' velocities are always zero
        for node_type in ['side_wall', 'packing']:
            assert self.min_yvel_by_node_type[node_type] == 0, (node_type, 'min yvel', self.min_yvel_by_node_type[node_type])
            assert self.max_yvel_by_node_type[node_type] == 0, (node_type, 'max yvel', self.min_yvel_by_node_type[node_type])
            assert self.min_xvel_by_node_type[node_type] == 0, (node_type, 'min xvel', self.min_yvel_by_node_type[node_type])
            assert self.max_xvel_by_node_type[node_type] == 0, (node_type, 'max xvel', self.min_yvel_by_node_type[node_type])
               
                
    def check_vals_in_frame(self, data, sim):
        # check node values in a frame

        # at inflow, y velocity is inlet velocity, x velocity is small        
        unique_inflow_vel = data.query('node_type=="inflow"').drop_duplicates(subset=['Velocity[j] (m/s)'])
        unique_inflow_frac = data.query('node_type=="inflow"').drop_duplicates(subset=['Volume Fraction of Liq'])
        unique_inflow_pres = data.query('node_type=="inflow"').drop_duplicates(subset=['Pressure (Pa)'])

        # at inflow, volume fraction and velocity are constant, pressure is not
        assert len(unique_inflow_vel)== 1
        assert len(unique_inflow_frac)==1
        assert len(unique_inflow_pres)>1

        # at inflow, volume fraction is always 1 
        assert unique_inflow_vel['Volume Fraction of Liq'].values[0] == 1

        # at inflow, y velocity is equal to the negative of the sim's inlet velocity     
        assert unique_inflow_vel['Velocity[j] (m/s)'].values[0] == -self.inletVelocity[sim-1]

        # at outflow top, y velocity is always 0.019639
        unique_outflow_top_vel = data.query('node_type=="outflow_top"').drop_duplicates(subset=['Velocity[j] (m/s)'])
        assert len(unique_outflow_top_vel) == 1
        assert unique_outflow_top_vel['Velocity[j] (m/s)'].values[0] == 0.01963862, (unique_outflow_top_vel['Velocity[j] (m/s)'].values[0])

        # track maximum and minimum for each channel
        for node_type in ['fluid', 'side_wall', 'outflow_bottom', 'outflow_top', 'inflow', 'packing']:
            mn = data.query('node_type==@node_type')['Velocity[j] (m/s)'].min()
            mx = data.query('node_type==@node_type')['Velocity[j] (m/s)'].max()
            if self.min_yvel_by_node_type[node_type] > mn:
                self.min_yvel_by_node_type[node_type] = mn
            if self.max_yvel_by_node_type[node_type] < mx:
                self.max_yvel_by_node_type[node_type] = mx

            mn = data.query('node_type==@node_type')['Velocity[i] (m/s)'].min()
            mx = data.query('node_type==@node_type')['Velocity[i] (m/s)'].max()
            if self.min_xvel_by_node_type[node_type] > mn:
                self.min_xvel_by_node_type[node_type] = mn
            if self.max_xvel_by_node_type[node_type] < mx:
                self.max_xvel_by_node_type[node_type] = mx

            mn = data.query('node_type==@node_type')['Pressure (Pa)'].min()
            mx = data.query('node_type==@node_type')['Pressure (Pa)'].max()
            if self.min_pres_by_node_type[node_type] > mn:
                self.min_pres_by_node_type[node_type] = mn
            if self.max_pres_by_node_type[node_type] < mx:
                self.max_pres_by_node_type[node_type] = mx

            mx = data.query('node_type==@node_type')['Volume Fraction of Liq'].max()
            if self.max_frac_by_node_type[node_type] < mx:
                self.max_frac_by_node_type[node_type] = mx
                

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        sim = idx // 500 + 1
        time = idx % 500 + 1
        data = torch.load(osp.join(self.processed_dir, 'PNNL_{}_{}.pt'.format(sim, time)))
        return data
    

def pre_transform(self, data, sim, time):
    # sort data by y then x
    data = data.sort_values(by=['Y (m)','X (m)'])
    data['sim'] = [sim]*data['X (m)'].size
    data['time'] = [time]*data['X (m)'].size
    data['node'] = range(data['X (m)'].size)
    if self.n_nodes==None:
        self.n_nodes = data['X (m)'].size
    else:
        # confirm that all frames have the same number of nodes
        # note that these nodes are not always in the same place
        assert data['X (m)'].size == self.n_nodes

    ###########################################################################
    # define node types 
    data['node_type'] = 'fluid'
    data.loc[np.isin(np.array(data.node), list(self.const_nodes_by_sim)),'node_type'] = 'side_wall'
    data.loc[data['Y (m)'] == 0.09,'node_type'] = 'outflow_bottom'
    data.loc[data['Y (m)'] == 0.21,'node_type'] = 'outflow_top'
    data.loc[data['Y (m)'] == 0.2075,'node_type'] = 'inflow'        

    # note that packing and side wall always have zero velocity but can have non-zero volume fraction. 
    # there may not be a great justification for splitting these two node types. 
    # ***to-do*** consider merging these two node types under the "obstacle" name
    data.loc[np.all((data['node_type'] == 'side_wall',
                        data['X (m)'].abs()<0.0505,
                        data['Y (m)']<0.2075),axis=0), 'node_type'] = 'packing'
    # end define node types
    ###########################################################################

    # check node values in a frame
    self.check_vals_in_frame(data, sim)
    
    return data