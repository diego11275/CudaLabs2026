#!/bin/bash
#SBATCH -J kardash_task3
#SBATCH -p compclass
#SBATCH -o task3_filter_%j.out
#SBATCH -e task3_filter_%j.err
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH -t 20:00:00

set -e
export PATH=/opt/slurm/bin:$PATH
module load nvidia/cuda
module load mpi/openmpi

test -x task3_filter || nvcc -O3 -ccbin=mpicxx -o task3_filter task3_filter_mpi.cu -lpng -lz

HOSTS=$(mktemp)
scontrol show hostnames $SLURM_JOB_NODELIST > "$HOSTS"
echo "hostfile:"; cat "$HOSTS"

# Ключевое: OpenMPI-лончер обходит PMI/PMIx и стартует процессы сам
mpirun --hostfile "$HOSTS" -np 2 --map-by ppr:1:node \
       --mca plm_rsh_agent ssh \
       ./task3_filter input.png

rm -f "$HOSTS"