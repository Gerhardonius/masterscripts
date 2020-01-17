#!/bin/bash
#SBATCH --partition=c
#SBATCH --cpus-per-task=20
#SBATCH --qos=c_medium
ml --latest singularity

#sbatch --job-name=test --output=test.log --error=test.err SMsingularitywrapper.sh -c ee/mm -r lo/hi -n 3"

#
# Directories, number of events per run
#
SOURCEELMODELDIR=/mnt/hephy/pheno/gerhard/Madresults/singularity/v1/SM/SM_DYjets_dielec
SOURCEMUMODELDIR=/mnt/hephy/pheno/gerhard/Madresults/singularity/v1/SM/SM_DYjets_dimuon
#SOURCEELMODELDIR=/mnt/hephy/pheno/gerhard/Madresults/singularity/firsttry
MODELNAME="SM"
NRE=50000

#
# read flags
#
while getopts c:r:n: option
do
	case "${option}"
	in
		c) CHANNEL=${OPTARG};;
		r) MLLRANGE=${OPTARG};;
		n) RUNNUMBER=${OPTARG};;
	esac
done

#
# choose channel
#
if [ "$CHANNEL" = "ee" ]
then
	SOURCEMODELDIR=$SOURCEELMODELDIR
elif [ "$CHANNEL" = "mm" ]
then
	SOURCEMODELDIR=$SOURCEMUMODELDIR
else
	echo "usage: sbatch --job-name=test --output=test.log --error=test.err SMsingularitywrapper.sh -c ee/mm -r lo/hi -n 3"
	exit 0
fi

#
# copy source directory 
#

# copy the compiled dir
MODELDIR=${SOURCEMODELDIR}_${MODELNAME}_${MLLRANGE}_${RUNNUMBER}
cp -r ${SOURCEMODELDIR} ${MODELDIR}

#
# write madevent commands and call singulartiy shell
#

touch tmpcmd
echo "generate_events ${MODELNAME}_${CHANNEL}_${MLLRANGE}_${RUNNUMBER}"	>> tmpcmd
#echo "launch ${MODELNAME}"			>> tmpcmd
echo "set nevents ${NRE}" 			>> tmpcmd
if [ "$MLLRANGE" = "lo" ]
then
	echo "set mmll 50" 				>> tmpcmd
	echo "set mmllmax 500" 				>> tmpcmd
elif [ "$MLLRANGE" = "hi" ]
then
	echo "set mmll 500" 				>> tmpcmd
	echo "set mmllmax -1" 				>> tmpcmd
else
	echo "usage: sbatch --job-name=test --output=test.log --error=test.err SMsingularitywrapper.sh -c ee/mm -r lo/hi -n 3"
	exit 0
fi
#echo "set gVl1x1 0.1" 				>> tmpcmd
#echo "set gAl1x1 0.2" 				>> tmpcmd
##echo "set gVl2x2 500" 			>> tmpcmd
##echo "set gAl2x2 500" 			>> tmpcmd
#echo "set MZp ${MASS}" 			>> tmpcmd
#echo "set wzp ${WIDTH}" 			>> tmpcmd

singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg ${MODELDIR}/bin/madevent tmpcmd
rm tmpcmd
