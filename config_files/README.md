


# Configuration Parameters

Below are the various configuration parameters currently supported in our codebase.
> Note: If any of the following configuration parameters are not specified, they're *Default* values are used automatically

## [DATA] Parameters

### Dataset Class

**class_name** - Dataset Class name
**name** - Current run name, for logging purposes

### Graph Construction

**normal_node_types** - A list of normal node indices (Default: `[0]`)
**boundary_node_types** - A list of boundary node indexes (Default: `[1]`)
**source_node_types** - A list of source node indexes (Default: `[2]`)
**num_node_types** - Number of unique node types (Defaults to the sum of the length of each node type list. e.g. `sum(len(*_node_types)))`)
**graph_type** - Graph type, must be one of: `[radius, ??]`, (Default: `radius`)
**k** - Description. (Default: `10`)
**radius** - Radius of graph. (Default: `0.01`)

### Data Processing

**output_type** - Prediction target, must be one of: `[velocity, state]`, (Default: **velocity**)
**window_length** - Length of window, (Default : `5`)
**apply_onehot** - Apply OneHot transformation to input indices, must be one of: `[True, False]`, (Default: `False`)
**apply_partitioning** - Whether to partition the data, must be one of: `[True, False]`, (Default: `False`)
**noise** - Min/Max range representing the amount of random noise to inject into inputs, (Default: `None`)
**noise_gamma** - Gamma to use when sampling noise, (Default: `0.1`)
**normalize** - Whether to normalize model inputs, must be one of: `[True, False]`, (Default: `True`)

## [MODEL] Parameters

### Architecture

**model_arch** - Model architectures, must be one of `[MeshGraphNets, MultiMeshGraphNets]`, (Default: `MeshGraphNets`)
**mgn_dim** - Layer dimensions of the MLPs, (Default: `128`)
**mp_iterations** - Number of Message Passing Iterations and MLP Blocks, (Default: `15`)
**mlp_norm_type** - Normalization Layer to use, must be one of `[LayerNorm, GraphNorm, InstanceNorm, BatchNorm, MessageNorm]`, (Default: `LayerNorm`)
**normalize_output** - Whether to normalize model outputs, must be one of: `[True, False]`, (Default: `True`)
**integrator** - Which integrator to use, must be one of `[euler, heun, midpoint, heun_third, ssprk3]`, (Default: `euler`)
**connection_type** - *Deprecated, will use `FullResidualConnection` by default in future releases.* Type of connection to use between MP's. Must be one of: `[FullResidualConnection, ResidualConnection, InitialConnection, DenseConnection]`, (Default: `FullResidualConnection`)
**connection_alpha** -  *Deprecated, will use `0.5` by default in future releases.* Value within range [0,1] representing the linear interpolation amount for MP features. `ResidualConnection` & `InitialConnection` connection types only, (Default: `0.5`)
**connection_aggregation** - *Deprecated, will be removed in future releases.* The method of MP feature aggregation, for `DenseConnection` connection type only. Must be one of: `[concat, maxpool, attention]`, (Default: `concat`)

## [TRAINING] Parameters

### Hyperparameters

**epochs** - Number of epochs to train for, (Default: `100`)
**batch_size** - Number of samples within a batch, (Default: `4`)
**grad_accum_steps** - Number of batches to predict before computing gradients, (Default: `1`)
**lr** - Learning rate, (Default: `1e-3`)
**scheduler** - Learning rate scheduler, must be one of `[ExpLR, OneCycleLR, ReduceLROnPlateau]`, (Default: `ExpLR`)
**use_parallel** - Whether to use DataParallel (DP). Allows the use of larger batch sizes via multiple GPUs, but the model will still run in a single-process setting. For true parallelism, set to `False` and use DistributedDataParallel (DDP) training scripts. (Default: `False`)
**load_prefix** = Prefix string to load specific checkpoint files, must be one of `[last_epoch, best_val]`, (Default: `last_epoch`)

### Logging

**log_rate** - Number of steps to log metrics, (Default: `100`)
**use_tensorboard** - Whether to record metrics with Tensorboard, must be one of `[True, False]`, (Default: `True`)
**tb_rate** - Number of steps to log metrics with Tensorboard, (Default: `10`)
**expt_name** - Experiment name/folder name for saving any outputs, such as checkpoints, metrics, etc, (Default: `Experiment_1`)
**train_output_dir** - Absolute path of where to save any training outputs, a new folder will be created using the value in `expt_name`, (Default: `~`)
**checkpoint_dir** - Checkpoint write-out directory. Automatically set, would not recommend changing, (Default: `\${TRAINING:train_output_dir}/checkpoints/\${TRAINING:expt_name}`)
**log_dir** - Log directory. Automatically set, would not recommend changing, (Default: `\${TRAINING:train_output_dir}/logs/\${TRAINING:expt_name}`)
**tensorboard_dir** - Tensorboard directory. Automatically set, would not recommend changing, (Default: `\${TRAINING:train_output_dir}/tensorboard/\${TRAINING:expt_name}`)

### Parallelism

**pin_memory** - Sets `pin_memory` in PyTorch's DataLoaders, must be one of `[True, False]`, (Default: `True`)
**ddp_type** - Determines DDP process init strategy. Set to `srun` if using a job scheduler like LSF or SLURM to run DDP across multiple nodes. Set to `manual` if using an interactive/local session to run DDP via a single python command, (Default: `srun`)
**world_size** - The amount of processes to run. If using a normal training script, set to `1` (single process). If using distributed training scripts, set to `number_of_gpus`. See `DistributedDataParallel` training scripts for more details, (Default: `1`)
**data_num_workers** - Sets `num_works` in PyTorch's DataLoaders, (Default: `0`)

## [SCHEDULER] Parameters

### ExpLR Settings

**scheduler_decay_interval** = Used to compute number of `decay_steps`, (Default: `0.5`)
**gamma** = Multiplicative factor of learning rate decay, (Default: `1e-2`)
**min_lr** = Minimum learning rate to not decay past, (Default: `1e-8`)

**decay_steps** = Manually override decay steps, otherwise it'll automatically compute it using `tot_steps * scheduler_decay_interval`, (Default: *Don't list in config*)

### OneCycle Settings
**max_lr** = Max learning rate for PyTorch's `OneCycleLR`, (Default: `3e-2`)

### ReduceLROPlateau Settings
**patience** - Patience for PyTorch's `ReduceLROnPlateau`, (Default: `50`)

## [PARTITIONING] Settings

**partitioning_method** - Method to partition a domain, must be one of `["null", "grid", "modularity", "range"]`, (Default: `"null"`)
**padding** - Amount to pad each patch (e.g. `0`)
**nrows** - Number of rows in the grid (e.g. `8`)
**ncols** - Number of columns in the grid (e.g. `8`)
**x_range** - Lists of 2-tuples defining the patches (e.g. `[(0, 0.8), (0.8, 1.6), (0, 0.8), (0.8, 1.6)]`)
**y_range** - Lists of 2-tuples defining the patches (e.g. `[(0, 0.205), (0, 0.205), (0.205, 0.41), (0.205, 0.41)]`)
Note regarding `x_range` and `y_range`:

> `x_range` and `y_range` must have the same length. graph will be partitioned into N patches, where N is the size of `x_range` and `y_range`
> The *ith* patch consists of the nodes in the rectangle with corners:
> - `(x_range[i][0], y_range[i][0])`, `(x_range[i][0], y_range[i][1])`, `(x_range[i][1], y_range[i][0])`, `(x_range[i][1], y_range[i][1])`

## [TESTING] Settings

**do_rollout_test** = If `True`, the test dataset will be used for rollout predictions. If `False`, the train dataset will be used for rollout predictions. (Default: `True`)
**rollout_start_idx** = Jumps to a specific index in the dataset to start rollouts, (Default: `0`)
**test_output_dir** = Absolute path of where to save any testing outputs, a new folder will be created using the value in `expt_name`, (Default: `~`)
**rollout_dir** =Rollout write-out directory. Automatically set, would not recommend changing, (Default: `\${TESTING:test_output_dir}/rollout/\${TRAINING:expt_name}`)
**outfile** = Outfile for rollouts. Automatically set, would not recommend changing, (Default: `${TESTING:test_output_dir}/rollout.pk`)
**batch_size** - Batch size for test dataloaders. (Default: `${TRAINING:batch_size}`)
