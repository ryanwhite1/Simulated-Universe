#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00

# Load the necessary modules
module load anaconda3/5.2.0
conda init bash
conda activate ladder

# Change to the directory where your Python script is located
cd /data/uqbpope/Simulated-Universe

# Run your Python script
python3 MainSimulationFile.py > simulation_output.txt