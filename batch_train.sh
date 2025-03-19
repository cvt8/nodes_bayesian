#!/bin/bash
#SBATCH --job-name=nodes_bm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/lamb10.out
#SBATCH --error=logs/lamb10.err
#SBATCH --array=1-1:1
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL

# Author : Constantin Vaillant-Tenzer
# To see all the options, type sbatch --help
# Usage sbatch batch_example.sh

cd /home/adminialab #Replace with the relevant path

source ~/.bashrc
conda activate nodes_env

cd /home/adminialab/GitFiles/nodes_bayesian #Replace with the relevant path

# Creates the logs directory if it does not exist
if [ ! -d logs ]; then
    mkdir logs
fi

python main.py

conda deactivate