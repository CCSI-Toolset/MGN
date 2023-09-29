# Distributed Data Parallel Patch-Based Rollouts
*Note: This code is generalized to work with any dataset/config, but the READMEs will use the DM Cylinder Flow dataset/config as examples*

## Usage

In a **login node**,
- Activate your environment, `conda activate [ENV_NAME]`
- Then run: `bash job_submit_patch_rollout_ddp.sh [CONFIG_FILE]` to run a ddp patch-based rollout job
  - `[CONFIG_FILE]` - Must be an **absolute path** to a config file

## Setup

- Be sure to update `NUM_PROCESS_PER_NODE` in `job_run_patch_rollout_ddp.sh` to reflect the number of GPUs per node. (Set by default to 4)
- Be sure to update `job_submit_patch_rollout_ddp.sh` to use the desired settings:
  - `NNODES` - Number of nodes to use
  - `ACCOUNT` - Account to charge compute hours
  - `TIME` - Time limite for each job, in HH:MM:SS format
  - `GPU_CONSTRAINT` - Set to either `gpu` or `gpu_hbm80g` (NERSC specific)
-  In your config file, be sure to set `world_size` to be equal to: `NNODES * NUM_PROCESS_PER_NODE`,
    - As an example, if each node has 4 GPUs, and we're using 2 nodes, `world_size=8`
-  In your config file, be sure to set `ddp_type` to `srun`
