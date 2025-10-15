#!/bin/bash
#SBATCH --job-name="openwebtext"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuH200x8
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=8   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdxi-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 24:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

#SBATCH --mail-user=eshen6959@gmail.com
#SBATCH --mail-type="BEGIN,END"

export NCCL_DEBUG=INFO
export MASTER_PORT=$((SLURM_JOB_ID % 10000 + 10000))

cd /projects/bdxi/yshen10/llm_from_scratch/assignment1-basics

module load anaconda3_gpu/23.9.0
module load cuda/12.4

source activate cs336

echo "Node num: "$SLURM_NNODES
echo "GPU num: "$SLURM_GPUS_PER_NODE

export PYTHONPATH=$PWD

srun python -m cProfile -o adapters.prof tests/adapters.py

echo "Finished running baseline script."