#!/bin/sh
#SBATCH --partition=c
#SBATCH --cpus-per-task=1
#SBATCH --qos=c_short
#SBATCH --job-name=hist
#SBATCH --output=msg/hist.log
#SBATCH --error=msg/hist.err

python makehisto.py --ge=0.8 --gm=0.8 --M=750 --model=VV --int 
#python makehisto.py --ge=0.8 --gm=0.8 --M=750 --model=VV

python makehisto.py --ge=0.8 --gm=0.8 --M=750 --model=LR --int 
#python makehisto.py --ge=0.8 --gm=0.8 --M=750 --model=LR
