#!/bin/bash
#SBATCH --partition=c
#SBATCH --cpus-per-task=1
#SBATCH --qos=c_long
ml --latest singularity
# synthax for localrun per sector
#./local_run.sh z <run_dir> <usr_input_file> <usr_histo_file> <output_file_extension> <pdf_location> <num_processors> <which_sector>
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg ./local_run.sh z CMSDATABASE_LOCAL_SEP input_z_m50_NNPDF31_nnlo_luxqed.txt histograms.txt outputfileCMSDATABASE.dat .. 1 $1
