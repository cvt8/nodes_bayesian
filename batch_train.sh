#!/bin/bash
#SBATCH --job-name=nodes_bm_3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/run3.out
#SBATCH --error=logs/run3.err
#SBATCH --array=1-1:1
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL

# Author : Constantin Vaillant-Tenzer
# To see all the options, type sbatch --help
# Usage sbatch batch_example.sh

cd /home/your_name #Replace with the relevant path

source ~/.bashrc
conda activate nodes_env

cd PATH_TO_FILE/nodes_bayesian #Replace with the relevant path

# Creates the logs directory if it does not exist
if [ ! -d logs ]; then
    mkdir logs
fi

python main.py
python main_vanila.py

conda deactivate