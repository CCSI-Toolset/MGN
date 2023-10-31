#!/bin/sh
# echo list of nodes to job output
# useful to rsh into nodes while job is running to run top/nvidia-smi
srun -r 0 /bin/hostname

echo "START"

# ----- DDP Setup ----- 

export NUM_PROCESS_PER_NODE=4
export WORLD_SIZE=$(($NUM_PROCESS_PER_NODE*$SLURM_NNODES))

echo "NUM_PROCESS_PER_NODE="${NUM_PROCESS_PER_NODE}
echo "WORLD_SIZE="${WORLD_SIZE}

### source: https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
echo "NODELIST="${SLURM_NODELIST}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT

nvidia-smi
echo "SLURM_GPUS="$SLURM_GPUS
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE
echo "SLURM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE

echo `which python`

# ----- Run ----- 

# Go to rollout dir (specified by chainer.sh)
echo "ROLLOUTDIR="$ROLLOUTDIR
echo "pwd="$pwd

cd $ROLLOUTDIR

PYTHON_CMD="srun --ntasks-per-node $NUM_PROCESS_PER_NODE python $ROLLOUTSCRIPT --config_file $CONFIG"
echo $PYTHON_CMD
$PYTHON_CMD

echo "END"