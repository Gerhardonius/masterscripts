# generating events
Two implementations, with the same interface:
* clusterscripts 
* singularityscripts

With the Clusterscripts all computation is done on the cluster,
but stdout goes to the terminal. For singularity scripts (only the ZP),
the source directory is copied and on the copied version is used for the generation. 

## HOW TO:
* check settings in scripts:  
clusterscripts: info in SMsubmitSample  
singularityscripts: info in SMsingularitywrapper  
this means:  
define the compiled directory, where you want to generate events AND
define number of events and also the number of runs
* run the scripts via:  
./SMsubmitSample.sh -c ee
./ZPsubmitSample.sh -m VV -c ee -M 1000 -g 0.35 -W Auto

