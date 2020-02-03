# FEWZ
These scripts need to be run from the FEWZ/bin directory

inputfiles:
* my_input_z.txt: defines all settings
* my_histograms.txt and bins.txt: defines histograms and customary binning

pdfs can be used from FEWZ homepage (dat directory), or a external LHAPDF library can be used.

## LO
./local_run.sh <boson> <run_dir> <usr_input_file> <usr_histo_file> <output_file_extension> <pdf_location> <num_processors>  
This will create a directory called my_run where the preliminary output files named my_results.dat will be placed.  
./finish.sh <run_dir> <order_prefix>.<output_file_extension>  
This will create one final result file titled NNLO.my_results.dat  

Example:  
./local_run.sh z my_run my_input.txt my_histograms.txt my_results.dat .. 1  
./finish.sh my_run NNLO.my_results.dat  

## NNLO
NNLO calculation is splitted in 127(154) integration sectors (Vegas integration). For each sector a subir is created.   
Note: dont use the ./fewzz -i executable to run sectors separately. Use local_run.sh as follows:   
./local_run.sh <boson> <run_dir> <usr_input_file> <usr_histo_file> <output_file_extension> <pdf_location> <num_processors> <which_sector>  
./finish.sh <run_dir> <order_prefix>.<output_file_extension>    

Note: The first sector ran, until its forced termination, 14 days on slurm cluster.
But the result of the finish script gave me a correct (similar to the CMS one) result file.  

Example (of course it should be done with cluster!):  
for sector in {1..127}  
do  
./local_run.sh z CMSDATABASE_LOCAL_SEP input_z_m50_NNPDF31_nnlo_luxqed.txt histograms.txt outputfileCMSDATABASE.dat .. 1 $sector  
done  
when sectors are finished: 
./finish.sh CMSDATABASE_LOCAL_SEP NNLO.outputfileCMSDATABASE.dat

## extras:
FEWZ can also perform scale variation and pdf errors. This is done by finish.sh if outputfile name is choosen as follows:   
* preliminary merged result for central value: <order_prefix>.<output_file_extension>
* + info to calculate PDF errors: <order_prefix>.pdf.<output_file_extension>
* + positive scale variation: <order_prefix>.p_<output_file_extension>
* + negative scale variation: <order_prefix>.m_<output_file_extension>
