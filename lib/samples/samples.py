# samples 
from RootTools.core.Sample import Sample

# Syntax: fromFiles(name, files, treeName = "Events", normalization = None, xSection = -1, selectionString = None, weightString = None, isData = False, color = 0, texName = None, maxN = None):

# SM samples
# DYjets + photon induced channel
SM_DYjets_dielec = Sample.fromFiles("e/SM-DYjets", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec/PP1/SM_DYjets_dielec_hi.root", selectionString = 'gen_dl_mass>602.55')
SM_DYjets_dimuon = Sample.fromFiles("m/SM-DYjets", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dimuon/PP1/SM_DYjets_dimuon_hi.root", selectionString = 'gen_dl_mass>602.55')

CMS_sampleforlegend_e = Sample.fromFiles("e/CMS-DY", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dimuon_singlerun/PP2/SM_DY_dimuon_singlerun_hi.root")#, selectionString = 'gen_dl_mass<0')
CMS_sampleforlegend_m = Sample.fromFiles("m/CMS-DY", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dimuon_singlerun/PP2/SM_DY_dimuon_singlerun_hi.root")#, selectionString = 'gen_dl_mass<0')

## DY + photon induced channel
#SM_DY_dielec = Sample.fromFiles("SM_DY_dielec", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dielec/PP1/SM_DY_dielec_hi.root", selectionString = 'gen_dl_mass>602.55')
#SM_DY_dimuon = Sample.fromFiles("SM_DY_dimuon", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dimuon/PP1/SM_DY_dimuon_hi.root", selectionString = 'gen_dl_mass>602.55')
#SM_DY_dielec_singlerun = Sample.fromFiles("SM_DY_dielec_singlerun", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dielec_singlerun/PP2/SM_DY_dielec_singlerun_hi.root", selectionString = 'gen_dl_mass>602.55')
#SM_DY_dimuon_singlerun = Sample.fromFiles("SM_DY_dimuon_singlerun", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dimuon_singlerun/PP2/SM_DY_dimuon_singlerun_hi.root", selectionString = 'gen_dl_mass>602.55')
#
#SM_DY_dielec_singlerun_nocut = Sample.fromFiles("SM_DY_dielec_singlerun_nocut", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dielec_singlerun/PP2/SM_DY_dielec_singlerun_hi.root")
#SM_DY_dimuon_singlerun_nocut = Sample.fromFiles("SM_DY_dimuon_singlerun_nocut", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DY_dimuon_singlerun/PP2/SM_DY_dimuon_singlerun_hi.root")
#
#SM_DYjets_dielec_nocut = Sample.fromFiles("SM_DYjets_dielec", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec/PP1/SM_DYjets_dielec_hi.root")
#SM_DYjets_dimuon_nocut = Sample.fromFiles("SM_DYjets_dimuon", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dimuon/PP1/SM_DYjets_dimuon_hi.root")

#SM_DYjets_dielec_singlerun_nocut = Sample.fromFiles("SM_DYjets_dielec_singlerun_nocut", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec_singlerun/PP1/SM_DYjets_dielec_singlerun_hi.root")
# BSM samples
#BSM_DYJets_dielec = Sample.fromFiles("BSM_DYJets_dimuon", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/BSM_DYjets_dimuon/PP1/BSM_DYjets_dimuon.root" ) 
#BSM_DYJets_dimuon = Sample.fromFiles("BSM_DYJets_dimuon", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/BSM_DYjets_dimuon/PP1/BSM_DYjets_dimuon.root" ) 
