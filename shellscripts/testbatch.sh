#!/bin/bash

# NOTE: Subdir msg must exist, otherwise no msgs are written
sbatch --cpus-per-task=1 --qos=c_short --job-name=test --output=msg/test.log --error=msg/test.err loopoverarguments.sh 10 20  
