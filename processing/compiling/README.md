# Compile Directories
Note: The lepton masses are set to 0 in the parameters.
Also Higgs is excluded in s-channel.
If lepton masses are no-zero, than Higgs introduces a flavor violation in this process
because of the different lepton masses.
HOW TO: 
* run proc_cards: ./bin/mg5_aMC SM_dielec_proc_card.dat
* then replace the run_card.dat and parameter_card.dat of the compiled dir, with the appropriate ones.
* then submit jobs, following the instructions in eventgen

### Cards_CMS
contains run and proc card from CMS repositrory

### Cards_SM
* proc_card (to generate Madgraph process and compile dir) for every channel:  
model sm-no_b_mass, include photon in proton, DY+4jets without higgs output directory  

* run_card (copy from CMS, with minor changes):
copy this in compiled directory
changes i have made compared to CMS:  
.true.     = gridpack  !True = setting up the grid pack
$DEFAULT_PDF_SETS = lhaid
$DEFAULT_PDF_MEMBERS = reweight_PDF     ! if pdlabel=lhapdf, this is the lhapdf number 
added:  
325100    = lhaid

* param_card (default from SM UFO model with no_b_mass restriction):
nothing to be done

### Cards_BSM
* proc_card (to generate Madgraph process and compile dir) for every channel:  
model zPrime_UFO_LO_mod, include photon in proton, DY+4jets without higgs output directory  

* run_card (copy from CMS, with minor changes):
as in Cards_SM

* param_card (copy from SM compiled dir, added the additional Zp parameter)
to ensure we have the same parameters than in the SM.
restriction card no_b_mass is already included
