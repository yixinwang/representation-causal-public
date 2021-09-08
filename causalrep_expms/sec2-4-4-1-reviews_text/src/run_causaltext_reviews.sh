#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S%N)

PYCODE_SWEEP="causaltext_reviews"
DATASET_SWEEP="amazon tripadvisor yelp"
HDIM_SWEEP="128"
LR_SWEEP="1e-2"
L2REG_SWEEP="1e-3"
MODE_SWEEP="linear"
ZDIM_SWEEP="1000"
NUMFEA_SWEEP="5"

CODE_SUFFIX=".py"
OUT_SUFFIX=".out"
PRT_SUFFIX=".txt"

RUN_SCRIPT="run_causaltext_reviews_base.sh"

for i in {1..100}; do
    for ZDIMi in ${ZDIM_SWEEP}; do
        export ZDIM=${ZDIMi}
        for L2REGi in ${L2REG_SWEEP}; do
            export L2REG=${L2REGi}
            for LRi in ${LR_SWEEP}; do
                export LR=${LRi}
                for MODEi in ${MODE_SWEEP}; do
                    export MODE=${MODEi}
                    for NUMFEAi in ${NUMFEA_SWEEP}; do
                        export NUMFEA=${NUMFEAi}
                        for PYCODE_SWEEPi in ${PYCODE_SWEEP}; do
                            NAME=bash_${PYCODE_SWEEPi}
                            for DATASETi in ${DATASET_SWEEP}; do
                                export DATASET=${DATASETi}
                                for HDIMi in ${HDIM_SWEEP}; do
                                    export HDIM=${HDIMi}
                                    export FILENAME=${PYCODE_SWEEPi}${CODE_SUFFIX}
                                    export OUTNAME=${PYCODE_SWEEPi}_data${DATASETi}_HDIM${HDIMi}_MODE${MODEi}_lr${LRi}_L2REG${L2REGi}_ZDIM${ZDIMi}_NUMFEA${NUMFEAi}_${TIMESTAMP}${OUT_SUFFIX}
                                    export PRTOUT=${PYCODE_SWEEPi}_data${DATASETi}_HDIM${HDIMi}_MODE${MODEi}_lr${LRi}_L2REG${L2REGi}_ZDIM${ZDIMi}_NUMFEA${NUMFEAi}_${TIMESTAMP}${PRT_SUFFIX}
                                    echo ${NAME}
                                    sbatch --job-name=${NAME} --output=${OUTNAME} ${RUN_SCRIPT}
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
