#!/bin/bash
# scancel job ids: ./loopovernumbers lower upper
for i in $(seq $1 $2)
do
       	echo $i;
       	scancel $i;
done
