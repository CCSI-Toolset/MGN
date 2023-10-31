#!/bin/bash

# number of nodes, we need to specify here instead of in the submit.sh job
export NNODES=8
# rollout script, should just be a filename.
export ROLLOUTSCRIPT=rollout_patches_with_updates_ddp.py
# config, should be path to config file, as used in the python cmd line argument.
export CONFIG=$1
# rollout scripts directory
export ROLLOUTDIR=../../../rollout_scripts
# script with specific environment settings for the job
export STARTSCRIPT=job_run_patch_rollout_ddp.sh
# bank to use for the job allocation
export ACCOUNT=m4313_g
# time limit for the job, HH:MM:SS format
export TIME=00:05:00
# GPU Constraint. Can be one of [gpu, gpu_hbm80g]
export GPU_CONSTRAINT=gpu

name="${RUNNAME}"

echo "sbatch --nodes $NNODES --time $TIME --constraint $GPU_CONSTRAINT --gpus-per-node 4 --account $ACCOUNT -J $name --wrap=\"sh $STARTSCRIPT\""
sbatch --nodes $NNODES --time $TIME --constraint $GPU_CONSTRAINT --gpus-per-node 4 --account $ACCOUNT -J $name --wrap="sh $STARTSCRIPT"