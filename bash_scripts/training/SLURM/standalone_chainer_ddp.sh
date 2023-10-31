#!/bin/bash

# Arguments: sh chainer.sh TRAINCONFIG
# or Arguments: sh chainer.sh TRAINCONFIG repeat

# max number of jobs to chain. going over 5 may cause problems on nersc/perlmutter
export MAX=5
# number of nodes, we need to specify here instead of in the submit.sh job
export NNODES=2
# train script, should just be a filename. `python $TRAINSCRIPT`
export TRAINSCRIPT=train_mgn_generalized_ddp.py
# train config, should be path to config file, as used in the python cmd line argument.
export TRAINCONFIG=$1
# name of the chain: for logging files and job names
export CHAINNAME="$(basename $TRAINCONFIG)"
# training scripts directory
export TRAINDIR=../../../training_scripts
# script with specific environment settings for the job
export STARTSCRIPT=submit_ddp.sh
# bank to use for the job allocation
export ACCOUNT=m4313_g
# time limit for the job, HH:MM:SS format
export TIME=00:02:00
# GPU Constraint. Can be one of [gpu, gpu_hbm80g]
export GPU_CONSTRAINT=gpu

if [ $# -ne 1 ]; then
  echo "Usage: bash standalone_chainer_ddp.sh [ABS_PATH_TO_CONFIG]"
  exit 0
fi

# Create a normal job without a dependency
export currcount=0
name="${CHAINNAME}_${currcount}"
echo "sbatch --nodes $NNODES --time $TIME --constraint $GPU_CONSTRAINT --gpus-per-node 4 --account $ACCOUNT -o $name --wrap=\"sh $STARTSCRIPT\""
submit_text=$(sbatch --nodes $NNODES --time $TIME --constraint $GPU_CONSTRAINT --gpus-per-node 4 --account $ACCOUNT -o $name --wrap="sh $STARTSCRIPT")
job_id=$(echo $submit_text | grep -o -E [0-9]+)

# create MAX-1 more jobs that depend on the previous job
export currcount=$(expr $currcount + 1)
while [ $currcount -lt $MAX ]
do
    name="${CHAINNAME}_${currcount}"
    echo "sbatch --nodes $NNODES --time $TIME --constraint $GPU_CONSTRAINT --gpus-per-node 4 --account $ACCOUNT -o $name -d $job_id --wrap=\"sh $STARTSCRIPT\""
    submit_text=$(sbatch --nodes $NNODES --time $TIME --constraint $GPU_CONSTRAINT --gpus-per-node 4 --account $ACCOUNT -o $name -d $job_id --wrap="sh $STARTSCRIPT")
    job_id=$(echo $submit_text | grep -o -E [0-9]+)
    export currcount=$(expr $currcount + 1)
done