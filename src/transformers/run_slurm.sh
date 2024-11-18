#!/bin/bash
#SBATCH --job-name=attempt_training
#SBATCH --output=output_%j.log       # Log file for standard output, %j is the job ID
#SBATCH --error=error_%j.log         # Log file for errors
#SBATCH --partition=gpu_h100              # Specify the partition (adjust if needed for A100 GPUs)
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G                    # Memory allocation, adjusted for deep learning on A100
#SBATCH --time=1:00:00            # Maximum runtime (4 days)
#SBATCH --mail-type=ALL              # Notifications for job start, end, and failure
#SBATCH --mail-user=a.changalidi@student.maastrichtuniversity.nl

# Load required modules
module load 2022                     # Load the environment version as per the tutorial

# Activate your Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Source Conda
conda activate py37                        # Activate py37 environment

# Run the Python script
srun python /home/achangalidi/project/capacity/src/transformers/try.py  # Replace with the actual path to your Python script
