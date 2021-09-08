#!/bin/bash

#SBATCH --account=sml
#SBATCH -c 1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=32gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=gonzo,bobo

source /proj/sml_netapp/opt/anaconda3/etc/profile.d/conda.sh

conda activate pytorch

echo "python ${FILENAME} --spurious_corr ${SPURIOUSCORR} --hidden_dim ${HDIM} --l2_reg ${L2REG} --lr ${LR} --pm_coef ${PMCOEF} --z_dim ${ZDIM} --neg_corr ${NEGCORR} "

python ${FILENAME} --spurious_corr ${SPURIOUSCORR} --hidden_dim ${HDIM} --l2_reg ${L2REG} --lr ${LR} --pm_coef ${PMCOEF} --z_dim ${ZDIM} --neg_corr ${NEGCORR}

