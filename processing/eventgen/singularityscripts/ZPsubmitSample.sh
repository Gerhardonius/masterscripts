#!/bin/bash
# model channel Mass g Width
# ./ZPsubmitSample.sh -m VV -c ee -M 1000 -g 0.35 -W Auto

# not finished
echo "Implement model translation to couplings"
exit 0

#
# read flags
#
while getopts m:c:M:g:W: option
do
	case "${option}"
	in
		m) MODELL=${OPTARG};;
		c) CHANNEL=${OPTARG};;
		M) MASS=${OPTARG};;
		g) COUPLING=${OPTARG};;
		W) WIDTH=${OPTARG};;
	esac
done

#
# set parameters
#
COUPLINGhund=$(bc <<< "$COUPLING*100/1")
MODELNAME="${MODELL}${CHANNEL}_${MASS}_${COUPLINGhund}_${WIDTH}"

#
# check flags
#
case "$CHANNEL" in 
	ee|mm)
		# submit job
		echo "sbatch --job-name=${MODELNAME} --output=msg/${MODELNAME}.log --error=msg/${MODELNAME}.err ZPsingularityWrapper.sh -m ${MODELL} -c ${CHANNEL} -M ${MASS} -g ${COUPLING} -W ${WIDTH}"
		sbatch --job-name=${MODELNAME} --output=msg/${MODELNAME}.log --error=msg/${MODELNAME}.err ZPsingularityWrapper.sh -m ${MODELL} -c ${CHANNEL} -M ${MASS} -g ${COUPLING} -W ${WIDTH}
		;;
	*)
		echo "usage ./ZPsubmitSample.sh -m VV -c ee -M 1000 -g 0.35 -W Auto"
		;;
esac
