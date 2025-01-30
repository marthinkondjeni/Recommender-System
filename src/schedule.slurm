#!/bin/bash
#SBATCH --job-name=trans
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --error=job%J.err
#SBATCH --output=job%J.out

module purge
module load julia/1.11.2

julia -p 8 model.jl