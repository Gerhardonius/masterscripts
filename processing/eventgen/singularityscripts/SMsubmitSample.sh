#!/bin/bash
# model channel Mass g Width
# ./SMsubmitSample.sh -c ee


#
# Parameter
#
NBRUNLOW=5
NBRUNHIGH=10
MODELL="SM"

#
# read flags
#
while getopts c:r: option
do
	case "${option}"
	in
		c) CHANNEL=${OPTARG};;
		r) MLLRANGE=${OPTARG};;
	esac
done

#
# check flags
#
case "$CHANNEL" in 
	ee|mm)
	#
	# loop over runs
	#
	
	# low mll
	for run in $(seq 1 $NBRUNLOW)
	do
		# submit job
		echo "sbatch --job-name=${MODELL}_${CHANNEL}_lo_${run} --output=msg/${MODELL}_${CHANNEL}_lo_${run}.log --error=msg/${MODELL}_${CHANNEL}_lo_${run}.err SMsingularitywrapper.sh -c ${CHANNEL} -r lo -n ${run}"
		sbatch --job-name=${MODELL}_${CHANNEL}_lo_${run} --output=msg/${MODELL}_${CHANNEL}_lo_${run}.log --error=msg/${MODELL}_${CHANNEL}_lo_${run}.err SMsingularitywrapper.sh -c ${CHANNEL} -r lo -n ${run}
	done
	# high mll
	for run in $(seq 1 $NBRUNHIGH)
	do
		# submit job
		echo "sbatch --job-name=${MODELL}_${CHANNEL}_hi_${run} --output=msg/${MODELL}_${CHANNEL}_hi_${run}.log --error=msg/${MODELL}_${CHANNEL}_hi_${run}.err SMsingularitywrapper.sh -c ${CHANNEL} -r hi -n ${run}"
		sbatch --job-name=${MODELL}_${CHANNEL}_hi_${run} --output=msg/${MODELL}_${CHANNEL}_hi_${run}.log --error=msg/${MODELL}_${CHANNEL}_hi_${run}.err SMsingularitywrapper.sh -c ${CHANNEL} -r hi -n ${run}
	done
	;;
	*)
		echo "usage ./SMsubmitSample.sh -c ee/mm"
	;;
esac
