from importlib import import_module
import os
import pickle
from sys import path
import time

import argparse
import numpy as np
import tqdm

import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# for parallel: 
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel


#########
# local application imports
# get path to root of the project
mgn_code_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
path.append(mgn_code_dir)

from GNN.ModelConfig.ConfigMGN import ModelConfig
from GNN.MeshGraphNets import MeshGraphNets
from GNN.utils.train_utils import train_epoch, val_epoch, get_free_gpus, save_model, load_model
from GNN.utils.utils import get_delta_minute
from GNN.utils.ExpLR import ExpLR


#########
# load config file
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default='../configfiles/config_cylinderflow_np.ini',
                        help="Config file to train model. See ../configfiles for examples")
    return parser

# parser = build_parser()
# args = parser.parse_args()
m = ModelConfig('../configfiles/config_cylinderflow_np.ini')


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
free_gpus = get_free_gpus()
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    free_gpus = list(set(free_gpus).intersection(list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))))

device = torch.device("cuda:" + str(free_gpus[0]) if torch.cuda.is_available() and len(free_gpus) > 0 else "cpu")
if len(free_gpus) == 0 and device != 'cpu':
    raise Exception('No free GPUs')
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

model = model.to(device)


#########
# tensorboard settings
tb_dir = m.get_tensorboard_dir()
tb_rate = m.get_tb_rate()
log_rate = m.get_log_rate()


# create a summary writer.
tb_writer = None if not m.get_use_tensorboard() else SummaryWriter(tb_dir)
total_steps = (last_epoch + 1) * len(train_loader) #assumes batch size does not change on continuation


########
# training loop
best_loss = val_epoch(test_loader, model, loss_function, device, use_parallel)
epochs = m.get_epochs()

start_time = time.time()
tqdm_itr = tqdm.trange(last_epoch + 1, epochs, position=0, leave=True)
for i in tqdm_itr:
    tqdm_itr.set_description('Epoch')
    train_loss, total_steps = train_epoch(train_loader, model, loss_function, 
        optimizer, lr_scheduler, device, 
        use_parallel, tb_writer, tb_rate, total_steps)
    tqdm_itr.refresh()
    val_loss = val_epoch(test_loader, model, loss_function, device, use_parallel,
        tb_writer, total_steps, i)
    tqdm_itr.refresh()
   
    if val_loss < best_loss:
        best_loss = val_loss
        save_model(checkpoint_dir, 
            'best_val.pt', model, 
            'best_val_scheduler.pt', lr_scheduler,  
            'last_epoch.pt', i)

    if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
        lr_scheduler.step(val_loss)

    if tb_writer is not None:
        tb_writer.add_scalar("Opt/lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=total_steps)
        tb_writer.add_scalar("Profile/epoch_time", scalar_value=get_delta_minute(start_time), global_step=i)


#########
# rollout test
do_rollout_test = m.get_do_rollout_test()
if do_rollout_test:
    idx = m.get_rollout_start_idx()
    data = train_dataset.data[idx:]
    feature_size = data.shape[-1]
    graph = train_dataset.graph
    window_length = train_data_params['window_length']
    
    input_len = (train_dataset[0].x.shape[-1] - train_dataset.onehot_dim - graph.pos.shape[-1]) // window_length 
    initial_window_data = data[:window_length,:,:input_len]
    node_types = train_dataset.node_types
    num_nodes = len(node_types)
    node_coordinates = graph.pos.numpy()
    onehot_info = (train_dataset.apply_onehot, train_dataset.onehot_dim)
    rollout_iterations = len(data) - window_length
    sources = train_dataset.source_nodes
    source_data = (sources, data[window_length:, sources, :input_len]) 
    # script boundaries as well?
    boundaries = train_dataset.boundary_nodes
    fixed = np.concatenate([sources, boundaries])
    source_data = (fixed, data[window_length:, fixed, :input_len])
    update_function = None if not hasattr(train_dataset, 'update_function') else train_dataset.update_function

    if use_parallel:
        model = model.module
    
    # rollout_output is a numpy array of shape (rollout_iterations, num_nodes, feature_size)
    # for PNNL data: num_nodes=[variable depending on the data]                        
    #                feature_size=3 (x_velocity, y_velocity, pressure)
    rollout_output = {
        'pred': model.rollout(device,
                              initial_window_data,
                              graph,
                              node_types,
                              node_coordinates,
                              onehot_info=onehot_info,
                              rollout_iterations=rollout_iterations,
                              source_data=source_data,
                              update_function=update_function),
        'coords': node_coordinates
    }
    # [LOW-PRIORITY] TO DO: if we remesh, then we will need (rollout_iterations, num_nodes, new_feature_size) where
    #                       features are (x_position, y_position, x_velocity, y_velocity, pressure)

    # save rollout output to test_dir
    outfile = m.get_outfile()
    with open(outfile, 'wb') as f:
        assert(rollout_output['pred'].shape == (rollout_iterations, num_nodes, feature_size))
        pickle.dump(rollout_output, f)
