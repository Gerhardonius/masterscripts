#!/bin/bash
# model channel Mass g Width
# ./SMsubmitSample_multirun.sh -c ee

#
# Parameter
#
MODELL="SM"

#
# read flags
#
while getopts c: option
do
	case "${option}"
	in
		c) CHANNEL=${OPTARG};;
	esac
done

#
# check flags
#
case "$CHANNEL" in 
	ee|mm)
		# submit job
		echo "sbatch --job-name=${MODELL} --output=msg/${MODELL}.log --error=msg/${MODELL}.err SMsingularitywrapper_multirun.sh -c ${CHANNEL}"
		sbatch --job-name=${MODELL} --output=msg/${MODELL}.log --error=msg/${MODELL}.err SMsingularitywrapper_multirun.sh -c ${CHANNEL} 
		;;
	*)
		echo "usage ./SMsubmitSample_multirun.sh -c ee/mm"
		;;
esac


