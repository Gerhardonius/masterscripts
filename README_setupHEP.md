#how to set up HEP tools on CLIP

all installations should be done inside the container,
since the computations are also made there.
(dont forget that: outside container madgraph works, but its LHAPDF installation does not)

## Madgraph
download tar, untar

## lhapdf6
inside madgraph `install lhapdf6` does not work since it cant download the files for some reason.  
manually download version 6.1.6 (!) to ./HEPTools, then  
+ go to `HEPTools/HEPToolsInstaller`
+ `./HEPToolInstaller.py lhapdf6 --prefix=/users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/HEPTools --lhapdf6_tarball=/users/gerhard.ungersbaeck/tarballs/LHAPDF-6.1.6.tar.gz`
+ in input/mg5_configuration.txt set
`lhapdf = /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/HEPTools/lhapdf6/bin/lhapdf-config`
+ NOT used: 
`export LD_LIBRARY_PATH=/users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/HEPTools/lhapdf6/lib:$LD_LIBRARY_PATH`

Note: From inside madgraph you could try `install lhapdf6 --source=<PATH TO TARBALL>`

## hepmc - this is a pythi8 dependency
./HEPToolsInstaller hepmc --prefix=.../HEPTools --hepmc_tarball=....pathtotarball

## boost and zlib
automatically installed when installinn pythia 8, see pythia8

## pythia8
inside madgraph 'install pythia8'
installs pythia8 and its interface to mg5

## root
export ROOTSYS=/mnt/hephy/pheno/opt/root6.16-py37
export PATH=$PATH:$ROOTSYS/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOTSYS/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$ROOTSYS/lib

## Delphes (needs root)
inside madgraph 'install Delphes'

## fastjet
in input/mg5_configuration.txt set
fastjet = /users/gerhard.ungersbaeck/MG5_aMC_v2_6_7/HEPTools/fastjet-3.3.2/fastjet-config

## FEWZ 3.1
download FEWZ and data (pdfs) separately, then extract data to FEWZ
make fewzz
example:
./local_run.sh z testrun input_z.txt histograms.txt outputfile.dat .. 1
./finish.sh testrun LO.outputfile.dat

