#!/usr/bin/env bash
##### Name of the job (pick a name)
#SBATCH --job-name=transf_Train
##### Partition to run job
#SBATCH --partition=talon-gpu32
##### Number of nodes
#SBATCH --nodes=1
##### Number of GPUs per node for the job
#SBATCH --gpus-per-node=1
##### Maximum runtime for job (in this case the job involves training the net) (hh:mm:ss)
#SBATCH --time=48:00:00
##### Job events (mail-type): begin, end, fail, all.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alessandro.dossantos@und.edu

# Load virtual environment
source csci-463/project/venv_transformers/bin/activate

# Set current directory
cd csci-463/project/

# Create a timestamp variable
TIMESTAMP=$(date +"%Y-%m-%d_%H:%M:%S")

# Create a job batch id variable
JOB_BATCH_ID=$SLURM_JOB_ID

# Specify your model name for the output directory
$1="bart-large-cnn"

# Run training script
GPUS=1
NNODES=1
NODE_RANK=0
PORT=29500
MASTER_ADDR=127.0.0.1


current_date_time=$(date '+%Y\%m\%d___%H-%M-%S')
mkdir -p output/$1/$current_date_time
python3 huggingFace/finetuneTransformer_main.py --outputDir output/$1 --modelAndTokenizerName facebook/bart-large-cnn --task summarization >> output/$1/$current_date_time/output.log 2>&1