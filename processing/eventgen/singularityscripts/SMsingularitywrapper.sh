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
MODEL="SM"
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
MODELNAME=${MODEL}_${MLLRANGE}_${RUNNUMBER}
MODELDIR=${SOURCEMODELDIR}_${MODELNAME}
echo "cp -r ${SOURCEMODELDIR} ${MODELDIR}"
cp -r ${SOURCEMODELDIR} ${MODELDIR}
echo "cp completed"

#
# write madevent commands and call singulartiy shell
#

touch tmpcmd${MODELNAME}
echo "launch"					>> tmpcmd${MODELNAME}
echo "shower=Pythia8"				>> tmpcmd${MODELNAME}
echo "detector=Delphes"				>> tmpcmd${MODELNAME}
#echo "generate_events ${MODELNAME}"		>> tmpcmd${MODELNAME}
echo "done"	 				>> tmpcmd${MODELNAME}
echo "set run_tag ${MODELNAME}"			>> tmpcmd${MODELNAME}
echo "set nevents ${NRE}" 			>> tmpcmd${MODELNAME}
if [ "$MLLRANGE" = "lo" ]
then
	echo "set mmll 50" 				>> tmpcmd${MODELNAME}
	echo "set mmllmax 500" 				>> tmpcmd${MODELNAME}
elif [ "$MLLRANGE" = "hi" ]
then
	echo "set mmll 500" 				>> tmpcmd${MODELNAME}
	echo "set mmllmax -1" 				>> tmpcmd${MODELNAME}
else
	echo "usage: sbatch --job-name=test --output=test.log --error=test.err SMsingularitywrapper.sh -c ee/mm -r lo/hi -n 3"
	exit 0
fi
#echo "set gVl1x1 0.1" 				>> tmpcmd${MODELNAME}
#echo "set gAl1x1 0.2" 				>> tmpcmd${MODELNAME}
##echo "set gVl2x2 500" 			>> tmpcmd${MODELNAME}
##echo "set gAl2x2 500" 			>> tmpcmd${MODELNAME}
#echo "set MZp ${MASS}" 			>> tmpcmd${MODELNAME}
#echo "set wzp ${WIDTH}" 			>> tmpcmd${MODELNAME}
echo "done"	 				>> tmpcmd${MODELNAME}

singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg ${MODELDIR}/bin/madevent tmpcmd${MODELNAME}
rm tmpcmd${MODELNAME}
