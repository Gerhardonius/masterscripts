#!/bin/bash
# ./ZPsubmitSample.sh -m VV -c ee -M 1000 -g 0.35 -W Auto

# not finished
echo "Implement model translation to couplings"
exit 0

#
# Directories, number of runs, number of events per run
#
ELMODELDIR=/mnt/hephy/pheno/gerhard/Madresults/cluster/dielec/SM/SM_DYjets_dielec
MUMODELDIR=/mnt/hephy/pheno/gerhard/Madresults/cluster/dielec/SM/SM_DYjets_dimuon
MODELNAME="SM"
NRE=50000
NBRUNLOW=3
NBRUNHIGH=10

# read flags
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
# choose channel and cd to modeldir
#
if [ "$CHANNEL" = "ee" ]
then
	MODELDIR=$ELMODELDIR
elif [ "$CHANNEL" = "mm" ]
then
	MODELDIR=$MUMODELDIR
else
	echo "usage: ./ZPsubmitSample.sh -m VV -c ee/mm -M 1000 -g 0.35 -W Auto"
fi

cd $MODELDIR

#
# write madevent commands and call madevent
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

./bin/madevent tmpcmd
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

./bin/madevent tmpcmd
rm tmpcmd
