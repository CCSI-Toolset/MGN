#!/bin/bash

# number of nodes, we need to specify here instead of in the submit.sh job
export NNODES=8
# rollout script, should just be a filename.
export ROLLOUTSCRIPT=rollout_patches_with_updates_ddp.py
# config, should be path to config file, as used in the python cmd line argument.
export CONFIG=$1
# name of the chain: for logging files and job names
export RUNNAME="$(basename $CONFIG)"
# rollout scripts directory
export ROLLOUTDIR=../../../rollout_scripts
# script with specific environment settings for the job
export STARTSCRIPT=job_run_patch_rollout_ddp.sh
# bank to use for the job allocation
export BBANK=ccsi2
# time limit for the job
export BTIME=2:00
# outdir of the logs
export BOUTDIR=chainer_output_log

name="${RUNNAME}"
CMD="bsub -q pdebug -nnodes $NNODES -alloc_flags ipisolate -W $BTIME -G $BBANK -J $name -outdir $BOUTDIR -oo ${BOUTDIR}/${name}.out "

CMD_2="$CMD sh $STARTSCRIPT $ROLLOUTCONFIG"
echo $CMD_2

$CMD_2
