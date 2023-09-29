#!/bin/bash

# Arguments: sh chainer.sh TRAINCONFIG
# or Arguments: sh chainer.sh TRAINCONFIG repeat

# max number of jobs to chain
export MAX=4
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
export BBANK=ccsi2
# time limit for the job
export BTIME=12:00
# outdir of the logs
export BOUTDIR=chainer_output_log

repeat=$2

if [ $# -lt 1 ]; then
  echo "Number of arguments not expected. Exiting.."
  exit 0
fi

if [ $# -gt 2 ]; then
  echo "Number of arguments not expected. Exiting.."
  exit 0
fi

export FINISHED_FLAG=$(python fetch_finished_flag.py $TRAINCONFIG)
if [ "$FINISHED_FLAG" == "Finished" ]; then
	echo "Training finished, stopping chaining of jobs."
else
    if [ $# -eq 2 ]; then
      # Arguments: sh chainer.sh TRAINCONFIG repeat
      # if there's an argument to chainer.sh script: sh chainer arg0
      # meaning, there was a job running before this one
      lastcount=$repeat
      currcount=$(expr $lastcount + 1)
      if [ "$currcount" -ge "$MAX" ]; then
        echo "Chained more than $MAX jobs. Stop chaining..."
        exit 0
      fi

      name="${CHAINNAME}_${currcount}"
      dep_name="${CHAINNAME}_${lastcount}"
      outputname="${name}_%J"
      outputname="${outputname/..\/config_files\//}"
      outputname="${outputname/.ini/}"
      outputname="${outputname//\//_}"
      CMD="bsub -nnodes $NNODES -alloc_flags ipisolate -W $BTIME -G $BBANK -J $name -outdir $BOUTDIR -oo ${BOUTDIR}/${outputname}.out -w ended(${dep_name})"
    fi

    if [ $# -eq 1 ]; then
      # Arguments: sh chainer.sh TRAINCONFIG
      # if there's not an argument to chainer.sh script: sh chainer
      currcount=0
      name="${CHAINNAME}_${currcount}"
      outputname="${name}_%J"
      outputname="${outputname/..\/config_files\//}"
      outputname="${outputname/.ini/}"
      outputname="${outputname//\//_}"
      CMD="bsub -nnodes $NNODES -alloc_flags ipisolate -W $BTIME -G $BBANK -J $name -outdir $BOUTDIR -oo ${BOUTDIR}/${outputname}.out"
    fi

    CMD_2="$CMD sh $STARTSCRIPT $TRAINCONFIG $currcount"
    echo $CMD_2

    $CMD_2
fi
