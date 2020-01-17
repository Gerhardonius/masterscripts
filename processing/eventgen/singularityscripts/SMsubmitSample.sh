#!/bin/bash
# model channel Mass g Width
# ./SMsubmitSample.sh -c ee

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
		echo "sbatch --job-name=${MODELL} --output=msg/${MODELL}.log --error=msg/${MODELL}.err SMsingularitywrapper.sh -c ${CHANNEL}"
		sbatch --job-name=${MODELL} --output=msg/${MODELL}.log --error=msg/${MODELL}.err SMsingularitywrapper.sh -c ${CHANNEL} 
		;;
	*)
		echo "usage ./SMsubmitSample.sh -c ee/mm"
		;;
esac


