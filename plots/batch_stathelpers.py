#!/bin/sh
#SBATCH --partition=c
#SBATCH --cpus-per-task=1
#SBATCH --qos=c_short
#SBATCH --job-name=stat_b
#SBATCH --output=msg/stat_b.log
#SBATCH --error=msg/stat_b.err

#python stathelpers.py --powN=5 --ge=1. --gm=0.2 --M=750
python stathelpers.py --powN=5 --ge=0.6 --gm=1. --M=750
#python stathelpers.py --powN=5 --ge=0.4 --gm=0.6 --M=750

#python stathelpers.py --tag=testreversecoupling --powN=4 --ge=0.1 --gm=1.
#python stathelpers.py --tag=testreversecoupling --powN=4 --ge=1. --gm=0.1
# tooo much
#python stathelpers.py --powN=6
