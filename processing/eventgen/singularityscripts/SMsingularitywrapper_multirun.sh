#!/bin/bash
#SBATCH --partition=c
#SBATCH --cpus-per-task=4
#SBATCH --qos=c_short
ml --latest singularity

#sbatch --job-name=test --output=test.log --error=test.err SMsubmitSample.sh -c ee/mm"

#
# Directories, number of runs, number of events per run
#
ELMODELDIR=/mnt/hephy/pheno/gerhard/Madresults/singularity/v1/SM/SM_DYjets_dielec
MUMODELDIR=/mnt/hephy/pheno/gerhard/Madresults/singularity/v1/SM/SM_DYjets_dimuon
MODELNAME="SM"
NRE=99
NBRUNLOW=1
NBRUNHIGH=1

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
# choose channel
#
if [ "$CHANNEL" = "ee" ]
then
	MODELDIR=$ELMODELDIR
elif [ "$CHANNEL" = "mm" ]
then
	MODELDIR=$MUMODELDIR
else
	echo "usage: sbatch --job-name=test --output=test.log --error=test.err SMsubmitSample.sh -c ee/mm"
fi

#
# write madevent commands and call singulartiy shell
#

# mll 50 to 500
touch tmpcmd
echo "multi_run ${NBRUNLOW} ${MODELNAME}_lo"	>> tmpcmd
# this runs generate_events nowlaunchname_0 -f
#echo "generate_events ${MODELNAME}_lo"		>> tmpcmd
#echo "launch ${MODELNAME}"			>> tmpcmd
echo "set nevents ${NRE}" 			>> tmpcmd
echo "set mmll 50" 				>> tmpcmd
echo "set mmllmax 500" 				>> tmpcmd
#echo "set gVl1x1 0.1" 				>> tmpcmd
#echo "set gAl1x1 0.2" 				>> tmpcmd
##echo "set gVl2x2 500" 			>> tmpcmd
##echo "set gAl2x2 500" 			>> tmpcmd
#echo "set MZp ${MASS}" 			>> tmpcmd
#echo "set wzp ${WIDTH}" 			>> tmpcmd

singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg ${MODELDIR}/bin/madevent tmpcmd
rm tmpcmd

# mll 500 to -1
touch tmpcmd
echo "multi_run ${NBRUNLOW} ${MODELNAME}_hi"	>> tmpcmd
#echo "generate_events ${MODELNAME}_hi"		>> tmpcmd
#echo "launch ${MODELNAME}"			>> tmpcmd
echo "set nevents ${NRE}" 			>> tmpcmd
echo "set mmll 50" 				>> tmpcmd
echo "set mmllmax 500" 				>> tmpcmd
#echo "set gVl1x1 0.1" 				>> tmpcmd
#echo "set gAl1x1 0.2" 				>> tmpcmd
##echo "set gVl2x2 500" 			>> tmpcmd
##echo "set gAl2x2 500" 			>> tmpcmd
#echo "set MZp ${MASS}" 			>> tmpcmd
#echo "set wzp ${WIDTH}" 			>> tmpcmd

singularity exec /mnt/hephy/pheno/ubuntu1904sing34.simg ${MODELDIR}/bin/madevent tmpcmd
rm tmpcmd
