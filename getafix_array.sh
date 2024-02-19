#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-00:30:00
#SBATCH --array=1-10

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate ladder

# Change to the directory where your Python script is located
cd /data/uqbpope/Simulated-Universe

# Run your Python script with SLURM_ARRAY_TASK_ID as argument
python MainSimulationFile.py -s $SLURM_ARRAY_TASK_ID > simulation_output_$SLURM_ARRAY_TASK_ID.txt
