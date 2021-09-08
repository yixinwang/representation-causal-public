#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S%N)

PYCODE_SWEEP="preproc_text"
DATASET_SWEEP="imdb imdb_sents kindle"
AGGRESSIVE_SWEEP="0"
TASKID_SWEEP="3"
LR_SWEEP="5e-2"
L2REG_SWEEP="5e-6"
OPTIM_SWEEP="adam"

CODE_SUFFIX=".py"
OUT_SUFFIX=".out"
PRT_SUFFIX=".txt"

RUN_SCRIPT="run_preproc_causaltext_base.sh"

for OPTIMi in ${OPTIM_SWEEP}; do
    export OPTIM=${OPTIMi}
    for L2REGi in ${L2REG_SWEEP}; do
        export L2REG=${L2REGi}
        for LRi in ${LR_SWEEP}; do
            export LR=${LRi}
            for TASKIDi in ${TASKID_SWEEP}; do
                export TASKID=${TASKIDi}
                for PYCODE_SWEEPi in ${PYCODE_SWEEP}; do
                    NAME=${PYCODE_SWEEPi}
                    for DATASETi in ${DATASET_SWEEP}; do
                        export DATASET=${DATASETi}
                        for AGGRESSIVEi in ${AGGRESSIVE_SWEEP}; do
                            export AGGRESSIVE=${AGGRESSIVEi}
                            export FILENAME=${PYCODE_SWEEPi}${CODE_SUFFIX}
                            export OUTNAME=${PYCODE_SWEEPi}_data${DATASETi}_agg${AGGRESSIVEi}_taskid${TASKIDi}_lr${LRi}_l2reg${L2REGi}_optim${OPTIMi}_${TIMESTAMP}${OUT_SUFFIX}
                            export PRTOUT=${PYCODE_SWEEPi}_data${DATASETi}_agg${AGGRESSIVEi}_taskid${TASKIDi}_lr${LRi}_l2reg${L2REGi}_optim${OPTIMi}_${TIMESTAMP}${PRT_SUFFIX}
                            echo ${NAME}
                            sbatch --job-name=${NAME} --output=${OUTNAME} ${RUN_SCRIPT}
                        done
                    done
                done
            done
        done
    done
done
