#!/bin/bash

#SBATCH --account=sml
#SBATCH -c 1
#SBATCH --time=72:59:00
#SBATCH --mem-per-cpu=32gb
#SBATCH --gres=gpu:1

source /proj/sml_netapp/opt/anaconda3/etc/profile.d/conda.sh

conda activate pytorch

echo "python ${FILENAME} "

python ${FILENAME}

