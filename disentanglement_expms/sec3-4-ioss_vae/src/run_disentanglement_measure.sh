#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S%N)

PYCODE_SWEEP="disentangle_measure"
DATASET_SWEEP="dsprites cars3d mpi3d smallnorb "
HDIM_SWEEP="1024"
LR_SWEEP="1e-2"
L2REG_SWEEP="1e-1"
MODE_SWEEP="logistic"
ZDIM_SWEEP="5"
NUMFEA_SWEEP="20"

CODE_SUFFIX=".py"
OUT_SUFFIX=".out"
PRT_SUFFIX=".txt"

RUN_SCRIPT="run_disentanglement_measure_base.sh"

for i in {1..500}; do
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
