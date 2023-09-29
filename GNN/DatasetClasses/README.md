# Dataset classes

These classes ultimately subclass PyTorch Geometric's `Dataset` class, adding preprocessing necessary for specific kinds of raw data. Add a new class by subclassing `MGNDataset` from `base.py`, which declares some methods that are expected by the MGN code.

# PNNL-specific notes

For the PNNL data, we preprocess the raw data with either `PNNL_With_Node_Type.py` or `PNNL_Raw_Dynamic_Vars_and_Meshes.py`; the latter saves out minimal data and is useful for dealing with very large graphs (like 3D data). These files perform checks to confirm that the data conforms to expectations (e.g., the liquid inlet nodes are source nodes and should have certain volume fractions and velocities) and add certain necessary variables (like node type) as needed.

Whether you're using 2D or 3D data, the preprocessed graphs can be too large to fit into GPU memory, so we further preprocess the data using either `PNNL_Subdomain.py` or `PNNL_Dynamic.py`; the latter is the relevant file when seeking to create patches of data that was initially preprocessed with `PNNL_Raw_Dynamic_Vars_and_Meshes.py`. Unlike its counterpart, `PNNL_Dynamic.py` does not save out the data for each simulation, timestep, and patch (due to the potential memory demands); instead, it loads the full domain data---which was initially preprocessed by `PNNL_Raw_Dynamic_Vars_and_Meshes.py`---for a given timestep *during training* and subsets it to the patch requested by the dataloader (i.e., preprocessing is done dynamically at training time).

An example call to an initial preprocessing code is given below, using multiple nodes managed with Slurm:

```
from PNNL_Raw_Dynamic_Vars_and_Meshes import PNNL_Raw_Dynamic_Vars_and_Meshes, pre_transform
import os
# raw data loc
root = '/global/cfs/cdirs/m4313/pnnl_globus/3D/'
# processed data loc
destination = '/pscratch/sd/b/bartolds/pnnl_data/'
# make data
PNNL_Raw_Dynamic_Vars_and_Meshes(
                 root,
                 transform= None, 
                 pre_transform= pre_transform,
                 node_type_file= None,
                 velocity_file= f'{destination}vels.txt',
                 sim_file_form= root[:-1],
                 force_check= False, # will overwrite stats created during processing
                 processed_dir = f'{destination}processed',
                 sim_range= None,
                 proc = int(os.environ['SLURM_PROCID']),
                 n_proc = int(os.environ['SLURM_NNODES'])
                 )

```