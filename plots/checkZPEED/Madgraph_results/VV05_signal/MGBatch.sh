#!/bin/sh
#SBATCH --partition=c
#SBATCH --cpus-per-task=1
#SBATCH --qos=c_short
#SBATCH --job-name=MG5
#SBATCH --output=MG5.log
#SBATCH --error=MG5.err
ml --latest singularity

singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/VV05_signal/MG5Card_CheckZPEED_VV05_uux_zp_mm.dat
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/VV05_signal/MG5Card_CheckZPEED_VV05_uux_zp_ee.dat
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/VV05_signal/MG5Card_CheckZPEED_VV05_ddx_zp_mm.dat
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/bin/mg5_aMC /users/gerhard.ungersbaeck/masterscripts/plots/checkZPEED/Madgraph_results/VV05_signal/MG5Card_CheckZPEED_VV05_ddx_zp_ee.dat

