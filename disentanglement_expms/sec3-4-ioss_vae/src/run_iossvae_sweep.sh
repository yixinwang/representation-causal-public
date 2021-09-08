#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S%N)

PYCODE_SWEEP="disentangle_learn"
DATASET_SWEEP="dsprites cars3d mpi3d smallnorb"
HDIM_SWEEP="512"
LR_SWEEP="1e-3"
BATCHSIZE_SWEEP="100"
IOSSWEIGHT_SWEEP="0 1e0 1e1 1e2 1e3 1e4 1e5 1e6"
ZDIM_SWEEP="10"
SPURIOUSCORR_SWEEP="0.9 0.2"

CODE_SUFFIX=".py"
OUT_SUFFIX=".out"
PRT_SUFFIX=".txt"

RUN_SCRIPT="run_iossvae_sweep_base.sh"

for i in {1..10}; do
    for ZDIMi in ${ZDIM_SWEEP}; do
        export ZDIM=${ZDIMi}
        for BATCHSIZEi in ${BATCHSIZE_SWEEP}; do
            export BATCHSIZE=${BATCHSIZEi}
            for LRi in ${LR_SWEEP}; do
                export LR=${LRi}
                for IOSSWEIGHTi in ${IOSSWEIGHT_SWEEP}; do
                    export IOSSWEIGHT=${IOSSWEIGHTi}
                    for SPURIOUSCORRi in ${SPURIOUSCORR_SWEEP}; do
                        export SPURIOUSCORR=${SPURIOUSCORRi}
                        for PYCODE_SWEEPi in ${PYCODE_SWEEP}; do
                            # NAME=${PYCODE_SWEEPi}
                            NAME="bash"
                            for DATASETi in ${DATASET_SWEEP}; do
                                export DATASET=${DATASETi}
                                for HDIMi in ${HDIM_SWEEP}; do
                                    export HDIM=${HDIMi}
                                    export FILENAME=${PYCODE_SWEEPi}${CODE_SUFFIX}
                                    export OUTNAME=${PYCODE_SWEEPi}_data${DATASETi}_HDIM${HDIMi}_IOSSWEIGHT${IOSSWEIGHTi}_lr${LRi}_BATCHSIZE${BATCHSIZEi}_ZDIM${ZDIMi}_SPURIOUSCORR${SPURIOUSCORRi}_${TIMESTAMP}${OUT_SUFFIX}
                                    export PRTOUT=${PYCODE_SWEEPi}_data${DATASETi}_HDIM${HDIMi}_IOSSWEIGHT${IOSSWEIGHTi}_lr${LRi}_BATCHSIZE${BATCHSIZEi}_ZDIM${ZDIMi}_SPURIOUSCORR${SPURIOUSCORRi}_${TIMESTAMP}${PRT_SUFFIX}
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
