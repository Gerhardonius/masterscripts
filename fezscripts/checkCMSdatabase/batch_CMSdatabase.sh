#!/bin/bash
#SBATCH -J FEWZ_CMSDB
#SBATCH --output=FEWZ_CMSDB.log
#SBATCH --error=FEWZ_CMSDB.err
#SBATCH --partition=c
#SBATCH --cpus-per-task=22
#SBATCH --qos=c_long
ml --latest singularity
#singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg local_run.sh z <OUTPUTDIR> <input_z.txt in bin> <histograms_CMS.txt in bin> <outputfile extension> .. <CPUs>
#NNLO computations are split in 127 sectors, LO and NLO have only one
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg ./local_run.sh z CMSDATABASE input_z_m50_NNPDF31_nnlo_luxqed.txt histograms.txt outputfileCMSDATABASE.dat .. 22
