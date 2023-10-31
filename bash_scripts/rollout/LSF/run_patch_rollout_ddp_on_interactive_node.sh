# Edit ROLLOUTDIR accordingly

echo "START"

export ROLLOUTDIR=/g/g12/saini5/CCSI/general_rollout2/fluidgnn/rollout_scripts
export ROLLOUTSCRIPT=rollout_patches_with_updates_ddp.py
export CONFIG=/g/g12/saini5/CCSI/general_rollout2/fluidgnn/config_files/delaunay_fixed/patch_rollout_pnnl_3D.ini

export OMPI_COMM_WORLD_SIZE=1
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_LOCAL_RANK=0
export MASTER_ADDR=`jsrun --nrs 1 -r 1 /bin/hostname`
export MASTER_PORT=12321
export NUM_PROCESS_PER_NODE=4

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export TORCH_CPP_LOG_LEVEL=INFO
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL

source activate /p/vast1/saini5/miniconda3/
conda activate lassen_ccsi

# # get out of all levels of conda
# conda deactivate
# conda deactivate
# source deactivate
    
# # activate shared env
# source /usr/workspace/ccsi/ml4cfd/conda_envs/torch190_cuda111/bin/activate

cd $ROLLOUTDIR
jsrun --smpiargs="-disable_gpu_hooks" -r $NUM_PROCESS_PER_NODE python $ROLLOUTSCRIPT --config_file $CONFIG

echo "END"