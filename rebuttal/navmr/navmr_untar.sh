#!/bin/bash
: "${WORK_ROOT:?set WORK_ROOT (working directory on the GPU machine (models/, results/, the scripts))}"
# extract Charades features for the navmr audit
cd ${WORK_ROOT}/navmr || exit 1
mkdir -p feat/charades
for f in txt_clip vid_clip vid_slowfast; do
  tar xzf dl/feat/charades/$f.tar.gz -C feat/charades && echo "done $f $(ls feat/charades/$f | wc -l)"
done
echo ALLDONE
