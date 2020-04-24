#!/bin/sh
#SBATCH --partition=c
#SBATCH --cpus-per-task=1
#SBATCH --qos=c_medium
#SBATCH --job-name=stat_1000
#SBATCH --output=msg/stat_1000.log
#SBATCH --error=msg/stat_1000.err

python onemass_exclusionplot_withexpected.py --M=500 --model=VV --points=20 --int 
#python onemass_exclusionplot.py --plot_directory=firstscan_noint --M=500 
