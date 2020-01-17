#!/bin/bash
#SBATCH --partition=c
#SBATCH --cpus-per-task=20
#SBATCH --qos=c_medium
ml --latest singularity
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC $1
