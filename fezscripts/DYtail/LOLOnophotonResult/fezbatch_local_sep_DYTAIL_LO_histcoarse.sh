#!/bin/bash
#SBATCH --job-name=DYtailLOhistcoarse
#SBATCH --output=DYtailLOhistcoarse.log
#SBATCH --error=DYtailLOhistcoarse.err
#SBATCH --partition=c
#SBATCH --cpus-per-task=20
#SBATCH --qos=c_short
ml --latest singularity
# synthax for localrun per sector
#./local_run.sh z <run_dir> <usr_input_file> <usr_histo_file> <output_file_extension> <pdf_location> <num_processors> <which_sector>
singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg ./local_run.sh z DYTAIL_LO_histcoarse input_z_m50_NNPDF31_nnlo_luxqed_DYtail_LO.txt histograms_DYtail.txt outputfileDYTAIL_LO_histcoarse.dat .. 20
