#!/bin/bash
# usage: ./sendtobatch_local_sep.sh

for sector in {1..127}
do
	sbatch --job-name=${sector} --output=${sector}.log --error=${sector}.err fezbatch_local_sep.sh ${sector}  
done
