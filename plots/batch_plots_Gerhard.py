#!/bin/sh
#SBATCH --partition=c
#SBATCH --cpus-per-task=1
#SBATCH --qos=c_short
#SBATCH --job-name=pltG
#SBATCH --output=msg/pltG.log
#SBATCH --error=msg/pltG.err

#python plots_Gerhard.py --plot_directory=DY_ten_vs_singlerun --lumi=139
#python plots_Gerhard.py --plot_directory=DYjets_vs_DY --lumi=139
python plots_Gerhard.py --plot_directory=DYinallvariables --lumi=139

#python plots_Gerhard.py --plot_directory=test --small --lumi=139
#python plots_Gerhard.py --plot_directory=kfactors --lumi=139
