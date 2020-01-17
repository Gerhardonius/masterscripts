#!/bin/bash
# usage .makeZpSamples.sh v1/modelparameterpoints.txt cluster/singularity
input=$1
# read lines, format: VV ee 2000 0.5 Auto asdf.root
while IFS= read -r line
do
	words=( $line )
	# skip comments starting with single #
	if [ "${words[0]}" != "#" ]
	then
		# submit batch job to produce them
		if [ "$2" = "cluster" ]
		then
			./eventgen/clusterscripts/ZPsubmitSample.sh -m ${words[0]} -c ${words[1]} -M ${words[2]} -g ${words[3]} -W ${words[4]}
		elif [ "$2" = "singularity" ]
		then
			./eventgen/singularityscripts/ZPsubmitSample.sh -m ${words[0]} -c ${words[1]} -M ${words[2]} -g ${words[3]} -W ${words[4]}
		else
			echo " usage .makeZpSamples.sh v1/modelparameterpoints.txt cluster/singularity"
		fi
	fi
done < "$input"
