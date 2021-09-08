#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S%N)

PYCODE_SWEEP="spurious_linear"
SPURIOUSCORR_SWEEP="0.95" # 0.1 0.5
HDIM_SWEEP="128"
LR_SWEEP="1e-2"
L2REG_SWEEP="1."
PMCOEF_SWEEP="0"
ZDIM_SWEEP="1"
NEGCORR_SWEEP="0"

CODE_SUFFIX=".py"
OUT_SUFFIX=".out"
PRT_SUFFIX=".txt"

RUN_SCRIPT="run_linear_synthetic_base.sh"

for i in {1..100}; do
    for ZDIMi in ${ZDIM_SWEEP}; do
        export ZDIM=${ZDIMi}
        for L2REGi in ${L2REG_SWEEP}; do
            export L2REG=${L2REGi}
            for LRi in ${LR_SWEEP}; do
                export LR=${LRi}
                for PMCOEFi in ${PMCOEF_SWEEP}; do
                    export PMCOEF=${PMCOEFi}
                    for NEGCORRi in ${NEGCORR_SWEEP}; do
                        export NEGCORR=${NEGCORRi}
                        for PYCODE_SWEEPi in ${PYCODE_SWEEP}; do
                            NAME=bash_${PYCODE_SWEEPi}
                            for SPURIOUSCORRi in ${SPURIOUSCORR_SWEEP}; do
                                export SPURIOUSCORR=${SPURIOUSCORRi}
                                for HDIMi in ${HDIM_SWEEP}; do
                                    export HDIM=${HDIMi}
                                    export FILENAME=${PYCODE_SWEEPi}${CODE_SUFFIX}
                                    export OUTNAME=${PYCODE_SWEEPi}_corr${SPURIOUSCORRi}_HDIM${HDIMi}_PMCOEF${PMCOEFi}_lr${LRi}_L2REG${L2REGi}_ZDIM${ZDIMi}_NEGCORR${NEGCORRi}_${TIMESTAMP}${OUT_SUFFIX}
                                    export PRTOUT=${PYCODE_SWEEPi}_corr${SPURIOUSCORRi}_HDIM${HDIMi}_PMCOEF${PMCOEFi}_lr${LRi}_L2REG${L2REGi}_ZDIM${ZDIMi}_NEGCORR${NEGCORRi}_${TIMESTAMP}${PRT_SUFFIX}
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
