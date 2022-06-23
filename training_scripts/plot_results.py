import imp
import matplotlib.pyplot as plt 
# See CylinderFlowDataset2.py
# Original Dataset
#   self.data (601,2520,3) 601 = time, 2520 = number of points, 3 = u,v,P
# 
#   Features: 8 (u,v, x, y, 0, 0, 0, 1), last 4 are node_type
#       node_type = 1 - source This is [0 0 1 0] because of oneshot
#       node_type = 2 - top, bottom, interior. This is [0 0 1 0] because of oneshot
#       node_type = 3 - right/outlet 
#       onehot num_classes = 3 which means [3] becomes [0 0 0 1]. [1] becomes [0 1 0 0]
#           so [0, 0, 0, 1] is 3 (an outlet) see process_node_window


# Outputs mgn_utils.py : get_sample
#   This depends on the output_type:
#   if "state" then outputs = u,v for a single time instance 
#   if "acceleration" outputs = u,v for a 3 time instances
#   if "velocity" outputs = u,v for a 2 time instances

#########
# local application imports
# get path to root of the project
import os
from sys import path
mgn_code_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
path.append(mgn_code_dir)
from importlib import import_module
import numpy as np 
import torch 
from GNN.ModelConfig.ConfigMGN import ModelConfig
from GNN.MeshGraphNets import MeshGraphNets
from GNN.utils.train_utils import train_epoch, val_epoch, get_free_gpus, save_model, load_model
from GNN.utils.utils import get_delta_minute
from GNN.utils.ExpLR import ExpLR
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.tri as mtri
import matplotlib.pyplot as plt 
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader

# Lets get the model and dataset 
m = ModelConfig('../configfiles/config_cylinderflow.ini'); free_gpus = [0]
batch_size = m.get_batch_size()

device = torch.device("cuda:" + str(free_gpus[0]) if torch.cuda.is_available() and len(free_gpus) > 0 else "cpu")

#########
# prepare train / test sets
train_data_params = m.get_train_data_params()
test_data_params = m.get_test_data_params()
ds_class_name = m.get_class_name()
DS = getattr(import_module('GNN.DatasetClasses.' + ds_class_name), ds_class_name)
train_dataset = DS(**train_data_params)
test_dataset = DS(**test_data_params)

levels= np.linspace(0,1.5,10) # color bar scale

#########
# build model
mgn_dim = m.get_mgn_dim()
model = MeshGraphNets(train_dataset.num_node_features, train_dataset.num_edge_features, train_dataset.output_dim, 
    out_dim_node=mgn_dim, 
    out_dim_edge=mgn_dim, 
    hidden_dim_node=mgn_dim, 
    hidden_dim_edge=mgn_dim, 
    hidden_dim_processor_node=mgn_dim, 
    hidden_dim_processor_edge=mgn_dim, 
    hidden_dim_decoder=mgn_dim,
    mp_iterations=m.get_mp_iterations(),
    mlp_norm_type=m.get_mlp_norm_type(),
    output_type=m.get_output_type())


#########
# reload model, if available
checkpoint_dir = m.get_checkpoint_dir()
train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=True)
model = DataParallel(model, device_ids=free_gpus)

#########
# optimizer settings
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = m.get_scheduler()
if scheduler == 'ExpLR':
    lr_scheduler = ExpLR(optimizer, decay_steps=4e4, min_lr=1e-8) 
elif scheduler == 'OneCycleLR':
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                                                           steps_per_epoch=len(train_loader), epochs=m.get_epochs())
elif scheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, min_lr=1e-8)
else:
    raise Exception('LR scheduler not recognized')

model, lr_scheduler, last_epoch = load_model(checkpoint_dir, 
    'best_val.pt', model, 
    'best_val_scheduler.pt', lr_scheduler,  
    'last_epoch.pt')

model = model.to(device)



os.makedirs('results/actual',exist_ok=True)

for i in range(len(test_dataset)):
    # Data to Predict 
    u = test_dataset[i].x[:,0].numpy()
    v = test_dataset[i].x[:,1].numpy()
    x = test_dataset[i].x[:,2].numpy()
    y = test_dataset[i].x[:,3].numpy()
    vmag = np.sqrt(u*u + v*v)
    point_indx = np.arange(len(y))

    # Grab the cylinder nodes 
    cylinder_nodes = test_dataset[i].x[:,6] == 1 
    x_cylinder = x[cylinder_nodes]
    y_cylinder = y[cylinder_nodes]
    point_indx = point_indx[cylinder_nodes]
    ind = (y_cylinder > 0.05) & (y_cylinder <0.35) 
    x_cylinder = x_cylinder[ind]
    y_cylinder = y_cylinder[ind]   
    point_indx = point_indx[ind]
  
    # Plot the cylinder (no need in a loop)
    # plt.figure(clear=True)
    # plt.plot(x_cylinder,y_cylinder,'.')
    # plt.axis('equal')
    # plt.show()
    
    # Lets mask the cylinder
    #   Hide any triangle with two or more verticies that match x_cylinder,y_cylinder 
    tri = Delaunay(test_dataset[i].x[:,2:4]) # Create triangle from x,y data
    mask = [True if (np.sum(np.isin(tri.simplices[i], point_indx)) >= 1) else False for i in range(len(tri.simplices)) ] 
    # Use the model to predict data
    # TODO: Write code to predict the what the data should be 



    # TODO: Make 3 subplots 
    # Plot the actual results 
    fig, ax = plt.subplot(311,clear=True)
    triang  = mtri.Triangulation(x, y, tri.simplices)
    triang.set_mask(mask)
    cbar = ax[0].tricontourf(triang, u, cmap='rainbow', levels=levels)
    ax[0].triplot(x, y, tri.simplices,linewidth=0.2)    
    ax[0].plot(x, y, '.',markersize=0.2)
    cbar2 = ax[0].colorbar(cbar, location='bottom')
    ax[0].set_axis('equal')
    ax[0].title('Velocity Magnitude')
    plt.show()
    print('check')

    # Plot the predicted results (Keep same scale as actual results)

    # Plot the error 

    