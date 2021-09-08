#!/bin/bash

#SBATCH --account=sml
#SBATCH -c 1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=32gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=gonzo

source /proj/sml_netapp/opt/anaconda3/etc/profile.d/conda.sh

conda activate pytorch

echo "python ${FILENAME} --dataset ${DATASET} --hidden_dim ${HDIM} --lr ${LR} --z_dim ${ZDIM} --batch_size ${BATCHSIZE} --ioss_weight ${IOSSWEIGHT} --spurious_corr ${SPURIOUSCORR} "

python ${FILENAME} --dataset ${DATASET} --hidden_dim ${HDIM} --lr ${LR} --z_dim ${ZDIM} --batch_size ${BATCHSIZE} --ioss_weight ${IOSSWEIGHT} --spurious_corr ${SPURIOUSCORR} 
