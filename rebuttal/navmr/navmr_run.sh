#!/bin/bash
: "${WORK_ROOT:?set WORK_ROOT (working directory on the GPU machine (models/, results/, the scripts))}"
# usage: bash navmr_run.sh <logname> [extra args to navmr_audit.py]
cd ${WORK_ROOT}/navmr || exit 1
LOG=$1; shift
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
GI=$(echo "$G" | cut -d, -f1 | tr -d ' '); GF=$(echo "$G" | cut -d, -f2 | tr -d ' ')
if [ "$GF" -ge 6144 ]; then export CUDA_VISIBLE_DEVICES=$GI; DEVARG="--device cuda"; else export CUDA_VISIBLE_DEVICES=; DEVARG="--device cpu"; fi
echo "gpu=$GI free=${GF}MiB CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $DEVARG" > $LOG
${PYTHON:-python} -I navmr_audit.py $DEVARG "$@" >> $LOG 2>&1
echo "EXIT $?" >> $LOG
