#!/bin/bash
#SBATCH -J kardash_calculation # Job name
#SBATCH -p gpuserv #(or "compclass_unstable", or "gpuserv", or "a100serv")
#SBATCH -o task2_filter_%j.out # Name of stdout output file (%j expands to %jobId)
#SBATCH -e task2_filter_%j.err # Name of stderr file (%j expands to %jobId)
#SBATCH -N 1 # Total number of nodes requested
#SBATCH -n 1 # Total number of mpi tasks requested
#SBATCH --mem=60G
#SBATCH -t 20:00:00 # Run time (hh:mm:ss)

export PATH=/opt/slurm/bin:$PATH
module load nvidia/cuda

# Компиляция на вычислительном узле (на головном нет nvcc/CUDA)
test -x task2_filter || nvcc -O3 -o task2_filter task2_filter.cu -lpng -lz
# Запуск: программа сама сгенерирует input.png, если его нет,
# прогонит blur/sobel/median и напечатает одну общую таблицу в stdout.
./task2_filter input.png
