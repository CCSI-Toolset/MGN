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
#   if "state" then outputs = u,v,p for a single time instance 
#   if "acceleration" outputs = u,v,p for a 3 time instances
#   if "velocity" outputs = u,v,p for a 2 time instances
# *Note Data is read and normalized and stored in the dataset. 

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
from torch_geometric.data import DataLoader


def preprocess():
    # parser = build_parser()
    # args = parser.parse_args()
    m = ModelConfig('../configfiles/config_cylinderflow.ini')

    #########
    # prepare train / test sets
    train_data_params = m.get_train_data_params()
    test_data_params = m.get_test_data_params()

    ds_class_name = m.get_class_name()
    DS = getattr(import_module('GNN.DatasetClasses.' + ds_class_name), ds_class_name)
    train_dataset = DS(**train_data_params)
    test_dataset = DS(**test_data_params)

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
    # device settings
    batch_size = m.get_batch_size()
    free_gpus = [0] #$ get_free_gpus()
    # if 'CUDA_VISIBLE_DEVICES' in os.environ:
    #     free_gpus = list(set(free_gpus).intersection(list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))))

    device = torch.device("cuda:" + str(free_gpus[0]) if torch.cuda.is_available() and len(free_gpus) > 0 else "cpu")
    # if len(free_gpus) == 0 and device != 'cpu':
    #     raise Exception('No free GPUs')
    use_parallel = m.get_use_parallel() and len(free_gpus) > 1 and device != 'cpu'

    if use_parallel:
        train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=True)
        model = DataParallel(model, device_ids=free_gpus)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


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

    loss_function = torch.nn.MSELoss()


    #########
    # reload model, if available
    checkpoint_dir = m.get_checkpoint_dir()

    model, lr_scheduler, last_epoch = load_model(checkpoint_dir, 
        'best_val.pt', model, 
        'best_val_scheduler.pt', lr_scheduler,  
        'last_epoch.pt')

    
    return test_dataset, model 



def CreatePlots():
    test_dataset,model = preprocess()
    model.eval()
    levels= np.linspace(0,1.5,10) # color bar scale

    #########
    # Start the prediction
    os.makedirs('results/actual',exist_ok=True)
    loss_fn = torch.nn.MSELoss()

    for i in range(len(test_dataset)):
        ########
        # Data to Predict 
        u = test_dataset[i].x[:,0].numpy()
        v = test_dataset[i].x[:,1].numpy()
        x = test_dataset[i].x[:,2].numpy()
        y = test_dataset[i].x[:,3].numpy()
        p = test_dataset[i].y[:,-1].numpy()
        vmag = np.sqrt(u*u + v*v)
        point_indx = np.arange(len(y))

        ##########
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
        
        #############
        # Use the model to predict data
        # TODO: Write code to predict the what the data should be 
        predicted = model(test_dataset[i])
        predicted = predicted.detach().numpy()
        u_loss = np.abs(predicted[:,0] - u)
        v_loss = np.abs(predicted[:,1] - v)
        p_loss = np.abs(predicted[:,2] - p)
        print('check')
        
        # TODO: Make 3 subplots 
        # Plot the actual results U-Velocity
        fig, ax = plt.subplots(333,sharey=True)
        triang  = mtri.Triangulation(x, y, tri.simplices)
        triang.set_mask(mask)
        cbar = ax[0,0].tricontourf(triang, u, cmap='rainbow', levels=levels)        # Plot U
        cbar2 = ax[0,0].colorbar(cbar, location='bottom')
        ax[0,0].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[0,0].plot(x, y, '.',markersize=0.2)
        ax[0,0].set_axis('scaled')
        ax[0,0].title('Normalized U - Velocity')

        cbar = ax[1,0].tricontourf(triang, predicted[:,0], cmap='rainbow', levels=levels)
        cbar2 = ax[1,0].colorbar(cbar, location='bottom')
        ax[1,0].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[1,0].plot(x, y, '.',markersize=0.2)
        ax[1,0].set_axis('scaled')
        ax[1,0].title('Normalized U - Velocity')

        cbar = ax[2,0].tricontourf(triang, u_loss, cmap='rainbow', levels=np.linspace(0,u_loss.max(),10))
        cbar2 = ax[2,0].colorbar(cbar, location='bottom')
        ax[2,0].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[2,0].plot(x, y, '.',markersize=0.2)
        ax[2,0].set_axis('scaled')
        ax[2,0].title('Elementwise Loss |u-u_actual|')

        cbar = ax[0,1].tricontourf(triang, v, cmap='rainbow', levels=levels)        # Plot V
        cbar2 = ax[0,1].colorbar(cbar, location='bottom')
        ax[0,1].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[0,1].plot(x, y, '.',markersize=0.2)
        ax[0,1].set_axis('scaled')
        ax[0,1].title('Normalized V - Velocity')

        cbar = ax[1,1].tricontourf(triang, predicted[:,1], cmap='rainbow', levels=levels)
        cbar2 = ax[1,1].colorbar(cbar, location='bottom')
        ax[1,1].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[1,1].plot(x, y, '.',markersize=0.2)
        ax[1,1].set_axis('scaled')
        ax[1,1].title('Normalized V - Velocity')

        cbar = ax[2,1].tricontourf(triang, v_loss, cmap='rainbow', levels=np.linspace(0,v_loss.max(),10))
        cbar2 = ax[2,1].colorbar(cbar, location='bottom')
        ax[2,1].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[2,1].plot(x, y, '.',markersize=0.2)
        ax[2,1].set_axis('scaled')
        ax[2,1].title('Elementwise Loss |v-v_actual|')

        cbar = ax[0,2].tricontourf(triang, v, cmap='rainbow', levels=levels)        # Plot P
        cbar2 = ax[0,2].colorbar(cbar, location='bottom')
        ax[0,2].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[0,2].plot(x, y, '.',markersize=0.2)
        ax[0,2].set_axis('scaled')
        ax[0,2].title('Normalized P')

        cbar = ax[1,2].tricontourf(triang, predicted[:,1], cmap='rainbow', levels=levels)
        cbar2 = ax[1,2].colorbar(cbar, location='bottom')
        ax[1,2].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[1,2].plot(x, y, '.',markersize=0.2)
        ax[1,2].set_axis('scaled')
        ax[1,2].title('Normalized P')

        cbar = ax[2,2].tricontourf(triang, p_loss, cmap='rainbow', levels=np.linspace(0,v_loss.max(),10))
        cbar2 = ax[2,2].colorbar(cbar, location='bottom')
        ax[2,2].triplot(x, y, tri.simplices,linewidth=0.2)    
        ax[2,2].plot(x, y, '.',markersize=0.2)
        ax[2,2].set_axis('scaled')
        ax[2,2].title('Elementwise Loss |p-p_actual|')
        plt.show()
        

        # Plot the predicted results (Keep same scale as actual results)

        # Plot the error 

if __name__ =="__main__":
    CreatePlots()