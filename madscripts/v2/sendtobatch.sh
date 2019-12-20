#!/bin/bash

# use argument test if only the commands should be echoed

# SM
# SM dimuon
echo sbatch --job-name=SM_mu --output=msg/SM_mu.log --error=msg/SM_mu.err madbatch.sh singlescripts/SM_DYjets_dimuon  
if [ "$1" != echo ]
then
	sbatch --job-name=SM_mu --output=msg/SM_mu.log --error=msg/SM_mu.err madbatch.sh singlescripts/SM_DYjets_dimuon  
fi

# SM dielec
echo sbatch --job-name=SM_el --output=msg/SM_el.log --error=msg/SM_el.err madbatch.sh singlescripts/SM_DYjets_dielec 
if [ "$1" != echo ]
then
	sbatch --job-name=SM_el --output=msg/SM_el.log --error=msg/SM_el.err madbatch.sh singlescripts/SM_DYjets_dielec  
fi

# BSM
for model in VV RR LL RL LR
do
	for ratio in 01 03 05
	do
		# BSM dimuon
		echo sbatch --job-name=${model}${ratio}_mu --output=msg/${model}${ratio}_mu.log --error=msg/${model}${ratio}_mu.err madbatch.sh singlescripts/ZP${model}${ratio}_DYjets_dimuon
		if [ "$1" != echo ]
		then
			sbatch --job-name=${model}${ratio}_mu --output=msg/${model}${ratio}_mu.log --error=msg/${model}${ratio}_mu.err madbatch.sh singlescripts/ZP${model}${ratio}_DYjets_dimuon  
		fi

		# BSM dielec
		echo sbatch --job-name=${model}${ratio}_el --output=msg/${model}${ratio}_el.log --error=msg/${model}${ratio}_el.err madbatch.sh singlescripts/ZP${model}${ratio}_DYjets_dielec
		if [ "$1" != echo ]
		then
			sbatch --job-name=${model}${ratio}_el --output=msg/${model}${ratio}_el.log --error=msg/${model}${ratio}_el.err madbatch.sh singlescripts/ZP${model}${ratio}_DYjets_dielec  
		fi

	done
done
