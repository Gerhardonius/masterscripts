#!/bin/bash
# loop over argument range: ./loopoverargument 10 20

for i in $(seq $1 $2)
do
       	echo $i;
done
