#!/bin/bash
#SBATCH -J 30_DYjets_std
#SBATCH --output=SM_DYjets_dimuon_30_std.log
#SBATCH --error=SM_DYjets_dimuon_30_std.err
#SBATCH --partition=c
#SBATCH --cpus-per-task=30
#SBATCH --qos=c_long
ml --latest singularity
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC SM_DYjets_dimuon_30
