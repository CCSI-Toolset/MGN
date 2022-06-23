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

from GNN.ModelConfig.ConfigMGN import ModelConfig
from GNN.MeshGraphNets import MeshGraphNets
from GNN.utils.train_utils import train_epoch, val_epoch, get_free_gpus, save_model, load_model
from GNN.utils.utils import get_delta_minute
from GNN.utils.ExpLR import ExpLR
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.tri as mtri
import matplotlib.pyplot as plt 

m = ModelConfig('../configfiles/config_cylinderflow.ini')

#########
# prepare train / test sets
train_data_params = m.get_train_data_params()
test_data_params = m.get_test_data_params()

ds_class_name = m.get_class_name()
DS = getattr(import_module('GNN.DatasetClasses.' + ds_class_name), ds_class_name)
train_dataset = DS(**train_data_params)
test_dataset = DS(**test_data_params)


os.makedirs('results/actual',exist_ok=True)
# Plot the actual results 
for i in range(len(train_dataset)):
    u = train_dataset[i].x[:,0]
    v = train_dataset[i].x[:,1]
    x = train_dataset[i].x[:,2]
    y = train_dataset[i].x[:,3]

    plt.figure(clear=True)
    tri = Delaunay(train_dataset[i].x[:,2:4])
    triang  = mtri.Triangulation(x, y, tri.simplices)
    cbar = plt.tricontourf(triang, u, cmap='rainbow')
    plt.triplot(x, y, tri.simplices,linewidth=0.2)    
    plt.plot(x, y, '.',markersize=0.2)
    cbar2 = plt.colorbar(cbar, location='bottom')
    plt.axis('equal')
    plt.title('X-Velocity')
    plt.show()
    print('check')