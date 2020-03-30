#!/bin/sh
#SBATCH --partition=c
#SBATCH --cpus-per-task=1
#SBATCH --qos=c_short
#SBATCH --job-name=MG5
#SBATCH --output=MG5.log
#SBATCH --error=MG5.err
ml --latest singularity

singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/LL05_total/MG5Card_CheckZPEED_LL05_uux_tot_mm.dat
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/LL05_total/MG5Card_CheckZPEED_LL05_uux_tot_ee.dat
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/LL05_total/MG5Card_CheckZPEED_LL05_ddx_tot_mm.dat
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/LL05_total/MG5Card_CheckZPEED_LL05_ddx_tot_ee.dat

