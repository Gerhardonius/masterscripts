#!/bin/bash

cwd=$(pwd)
for f in ${cwd}/singlescripts/*
do
	echo "Processing $f file..."
	#tail -49 $f > tmpfile	
	#sed -i 's/done//g' $f
	#sed -i 's/set mmll 500/set mmll 50/g' tmpfile
	#sed -i 's/set mmllmax -1/set mmllmax 500/g' tmpfile
	#cat tmpfile >> $f
	#sed -i 's/^$/d' $f
done
