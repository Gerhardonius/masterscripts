# compare getZpmodel_sep with ZPEED
# Note: to get counts from ZPEED: uncomment print statement in ATLAS_13TeV.py

# Successfully testet for M =500 and 1000, and g = 0.5 and 1.0

import numpy as np

# my stuff
from ZPEEDmod.Zpeedcounts import getZpmodel_sep, myDecayWidth, getBSMcounts

# ZPEED stuff
import ZPEED.dileptons_functions as df 
from ZPEED.ATLAS_13TeV import calculate_chi2
from ZPEED.ATLAS_13TeV_calibration import xi_function

ge= 0.5
gm= ge # need to be the same to compare to ZPEED
MZp = 500.
width = 'auto'
#width = 0.03
withint = False
sw = 3.

#
#
#
model = 'VV' # only VV model!
Zp_model_sep = getZpmodel_sep(ge ,gm , MZp, model = model,  WZp = width)
# just for comparison purpuse - give same couplings to tau leptons
Zp_model_sep['gtv'] =Zp_model_sep['gev'] 
Zp_model_sep['gta'] =Zp_model_sep['gea'] 
Zp_model_sep['gntv']=Zp_model_sep['gnev'] 
Zp_model_sep['gnta']=Zp_model_sep['gnea'] 
# recalculate the Width, since couplings did change
if width=='auto':
	Zp_model_sep['Gamma']=myDecayWidth(Zp_model_sep) 
print Zp_model_sep

# define search window
mllrange=[ MZp - sw* Zp_model_sep['Gamma'], MZp + sw* Zp_model_sep['Gamma'] ]
ee_bsm   = getBSMcounts( 'ee', Zp_model_sep, lumi =139., mllrange =  mllrange, withinterference = withint )
ee_bsm_bin = [ x[2] for x in ee_bsm ]
mm_bsm   = getBSMcounts( 'mm', Zp_model_sep, lumi =139., mllrange =  mllrange, withinterference = withint )
mm_bsm_bin = [ x[2] for x in mm_bsm ]

print 'BSM counts ZPEED (with weights at end of SR!)'
for c in ee_bsm_bin:
	print 'ee counts: ', c
for c in mm_bsm_bin:
	print 'mm counts: ', c

#
# ZPEED
#
# same thing as in Zp_model
MZp = MZp
sw2 = 0.23126# PDG (Olive et al 2014) value at Q^2 = m_Z^2
alpha_e = 1/128.0# PDG (Olive et al 2014) value at Q^2 = m_W^2
Lambda = 10000.
muscale = MZp
gV1x1 = ge
gA1x1 = 0.
gV2x2 = gm
gA2x2 = 0.
inducedcouplingfactor = (alpha_e/((1-sw2)*np.pi))*(gV1x1 + gV2x2)*np.log(Lambda/muscale)

Zp_model = {
  'MZp': MZp,  #Zp mass
  'mDM': 0.,   #Dark matter mass

  'gxv': 0.,     #Zp-DM vector coupling
  'guv': ( 5./18.)*inducedcouplingfactor,    #Zp-up-type-quark vector coupling
  'gdv': (-1./18.)* inducedcouplingfactor,    #Zp-down-type-quark vector coupling
  'glv': gV1x1 + (-1./2.)*inducedcouplingfactor,   #Zp-lepton vector coupling

  'gxa': 0.,     #Zp-DM axial coupling
  'gua': ( 1./ 6.)*inducedcouplingfactor,     #Zp-up-type-quark axial coupling
  'gda': (-1./ 6.)*inducedcouplingfactor,     #Zp-down-type-quark axial coupling
  'gla': gA1x1 + (-1./6.)*inducedcouplingfactor,     #Zp-lepton axial coupling
}

# The couplings to neutrinos follow from SM gauge invariance and the fact that right-handed neutrinos do not exist
Zp_model['gnv'] = 0.5 * (Zp_model['glv'] - Zp_model['gla'])
Zp_model['gna'] = 0.5 * (Zp_model['gla'] - Zp_model['glv'])

Zp_model['Gamma'] = df.DecayWidth(Zp_model)

# Compare to counts used in ZPEED 
# Get counts from ZPEED (in ATLAS_13TeV.py uncomment print statements)
if not withint:
	ee_signal = lambda x : xi_function(x, "ee") * df.dsigmadmll(x, Zp_model, "ee")
	mm_signal = lambda x : xi_function(x, "mm") * df.dsigmadmll(x, Zp_model, "mm")	
else:
	ee_signal = lambda x : xi_function(x, "ee") * df.dsigmadmll_wint(x, Zp_model, "ee")
	mm_signal = lambda x : xi_function(x, "mm") * df.dsigmadmll_wint(x, Zp_model, "mm")	

#if not withint:
#	ee_signal = lambda x : xi_function(x, "ee") * mydsigmadmll(x, Zp_model, "ee")
#	mm_signal = lambda x : xi_function(x, "mm") * mydsigmadmll(x, Zp_model, "mm")	
#else:
#	ee_signal = lambda x : xi_function(x, "ee") * mydsigmadmll_wint(x, Zp_model, "ee")
#	mm_signal = lambda x : xi_function(x, "mm") * mydsigmadmll_wint(x, Zp_model, "mm")	

# prints BSM counts (SM+1 * signal)
print 'BSM counts ZPEED (with weights at end of SR!)'
chi2, chi2_Asimov = calculate_chi2(ee_signal, mm_signal, signal_range= mllrange)


#
# checks
#
print 'compare Gamma:'
print 'Zpmodel_sep, ZPEED: ', Zp_model['Gamma'], Zp_model_sep['Gamma']


#################################
#	# Get counts from functions, defined above
#	sm_counts = getSMcounts( 'mm', counttype='expected', mllrange = mllrange , lumi =139.)
#	bsm_counts = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange = mllrange , withinterference = wint )
#
#	counts, weightA, weightB = getcounts_forstatistics('BSM' ,'mm', Zp_model, lumi = 139., mllrange = mllrange, withinterference = wint)
#
#	print 'SM values'
#	for i in range(len(sm_counts)):
#		print sm_counts[i][2]
#
#	print 'BSM values'
#	for i in range(len(sm_counts)):
#		print bsm_counts[i][2]
#	
#	print 'BSM values from forstatistics'
#	print 'weightA: ', weightA
#	print 'weightB: ', weightB
#	for i in range(len(sm_counts)):
#		print counts[i][2]
