import numpy as np
import scipy.integrate as integrate

import ZPEED.dileptons_functions as df #dsigmadmll_full, dsigmadmll_wint_full, DecayWidth_full
from ZPEED.ATLAS_13TeV_calibration import xi_function
from ZPEED.ATLAS_13TeV import ee_data, mm_data, ee_response_function, mm_response_function, ee_upward_fluctuation, mm_upward_fluctuation, ee_downward_fluctuation, mm_downward_fluctuation, calculate_chi2

# adaptions to Zpeed code to include flavor non universality
def mydsigmadmll(mll, Zp_model, species = "ee"):
	''' wrapper fro dsigmadmll_full
	added lepton flavor dependend coupling
	'''
  	if species == "ee":
  		sp = 0
  		glv = Zp_model['gev']
  		gla = Zp_model['gea']
  	elif species == "mm":
  		sp = 1
  		glv = Zp_model['gmv']
  		gla = Zp_model['gma']
  	else:
  	  	print("Unknown species requested")
  	  	return 0
  	return df.dsigmadmll_full(mll, Zp_model['MZp'], Zp_model['Gamma'], np.array([Zp_model['guv'],Zp_model['gdv']]), glv, np.array([Zp_model['gua'],Zp_model['gda']]), gla, sp)

def mydsigmadmll_wint(mll, Zp_model, species = "ee"):
	''' wrapper fro dsigmadmll_wint_full
	added lepton flavor dependend coupling
	'''
  	if species == "ee":
    		sp = 0
  		glv = Zp_model['gev']
  		gla = Zp_model['gea']
  	elif species == "mm":
    		sp = 1
  		glv = Zp_model['gmv']
  		gla = Zp_model['gma']
  	else:
    		print("Unknown species requested")
    		return 0
  	return df.dsigmadmll_wint_full(mll, Zp_model['MZp'], Zp_model['Gamma'], np.array([Zp_model['guv'],Zp_model['gdv']]), glv, np.array([Zp_model['gua'],Zp_model['gda']]), gla, sp)
 
def myDecayWidth(Zp_model):
	''' wrapper for DecayWidth_full(mll, mz, gfv, gfa, mf):
	added lepton flavor dependend coupling
	'''
	# vector couplings
  	gfv = np.array([Zp_model['gxv'], # DM
			Zp_model['guv'], # top
			Zp_model['gdv'], # bottom
			Zp_model['guv'], # charm
			Zp_model['gdv'], # strange
			Zp_model['gdv'], # down 
			Zp_model['guv'], # up
			Zp_model['gtv'], # tau
			Zp_model['gmv'], # mu
			Zp_model['gev'], # el
			Zp_model['gntv'], # tau neutrino
			Zp_model['gnmv'], # mu neutrino
			Zp_model['gnev']])# el neutrino
	# axial vector couplings
  	gfa = np.array([Zp_model['gxa'], # DM
			Zp_model['gua'], # top
			Zp_model['gda'], # bottom
			Zp_model['gua'], # charm
			Zp_model['gda'], # strange
			Zp_model['gda'], # down 
			Zp_model['gua'], # up
			Zp_model['gta'], # tau
			Zp_model['gma'], # mu
			Zp_model['gea'], # el
			Zp_model['gnta'], # tau neutrino
			Zp_model['gnma'], # mu neutrino
			Zp_model['gnea']])# el neutrino
	
  	return df.DecayWidth_full(Zp_model['MZp'], Zp_model['MZp'], gfv, gfa, np.array([Zp_model['mDM'], 173.0, 4.18, 1.275, 0.095, 0.0047, 0.0022, 1.77686, 0.1056583745, 0.0005109989461, 0., 0., 0.]))

#
# main functions here
#
def getSMcounts( channel, counttype='expected', lumi =139., mllrange = None):
	''' geturns bin counts as list
	mllrange = [ mll value to be included in first bin, mll value to be included in last bin]
	if mllrange == None: all bins
	counttype= 'expected' or 'observed'
	lumi in fb-1
	'''
	if channel not in ['ee','mm']:
		print 'Chose channel ee or mm'
		return []

	if counttype not in ['expected','observed']:
		print 'Chose counttype expected or observed'
		return []
	# ATLAS search results, counts
	analysis_name = 'ATLAS_13TeV'
	if lumi != 139.:
		print 'Warning: scaling up observed counts to higher Luminosity!'
		lumifactor = lumi/139. 
	else:
		lumifactor = 1

	if channel == 'ee':
		# import from ZPEED module
		#ee_data = np.loadtxt(analysis_name+'/ee_data.dat',delimiter='\t')
		bin_low = ee_data[:,0]
		bin_high = ee_data[:,1]
		bin_low = ee_data[:,0]
		bin_high = ee_data[:,1]
		if counttype == 'expected':
			counts = ee_data[:,3]
		if counttype == 'observed':
			counts = ee_data[:,2]
	
	if channel == 'mm':
		# import from ZPEED module
		#mm_data = np.loadtxt(analysis_name+'/mm_data.dat',delimiter='\t')
		bin_low = mm_data[:,0]
		bin_high = mm_data[:,1]
		bin_low = mm_data[:,0]
		bin_high = mm_data[:,1]
		if counttype == 'expected':
			counts = mm_data[:,3]
		if counttype == 'observed':
			counts = mm_data[:,2]

	if mllrange == None:
		mllrange = [225, 5806.19]
	[ Mlow, Mhigh] = mllrange	

	# Identify bins that cover the requested signal range
	i_low = 0 
	while bin_low[i_low+1] < Mlow and i_low < len(bin_low)-2: i_low = i_low + 1
	i_high = 0
	while bin_high[i_high] < Mhigh and i_high < len(bin_high)-1: i_high = i_high + 1

	rangecounts = []
	for i in range(i_low,i_high+1):
		rangecounts.append( [bin_low[i], bin_high[i], counts[i]] )

	return rangecounts*lumifactor

def getZpmodel(g, MZp, model = 'VV05',  WZp = 'auto'):
	# couplings to leptons, notation according to Madgraph
	modelstructure = model[:2]
	ratio = int(model[2:])
	if modelstructure == 'VV':
		gV2x2 = g*1.
		gV1x1 = (ratio/10.) * gV2x2
		gA1x1 = 0.
		gA2x2 = 0.
	if modelstructure == 'LL':
		gV2x2 = g*1.
		gV1x1 = (ratio/10.) * gV2x2
		gA2x2 = -gV2x2
		gA1x1 = -gV1x1
	if modelstructure == 'LR':
		gV2x2 = g*1.
		gV1x1 = (ratio/10.) * gV2x2
		gA2x2 = gV2x2
		gA1x1 = -gV1x1
	if modelstructure == 'RR':
		gV2x2 = g*1.
		gV1x1 = (ratio/10.) * gV2x2
		gA2x2 = gV2x2
		gA1x1 = gV1x1
	if modelstructure == 'RL':
		gV2x2 = g*1.
		gV1x1 = (ratio/10.) * gV2x2
		gA2x2 = -gV2x2
		gA1x1 = gV1x1
	gV3x3 = 0.
	gA3x3 = 0.

	# parameters taken from dilepton_functions.py for consistency
	# Weak Mixing Angle
	sw2 = 0.23126# PDG (Olive et al 2014) value at Q^2 = m_Z^2
	# Fine Structure Constant:
	alpha_e = 1/128.0# PDG (Olive et al 2014) value at Q^2 = m_W^2

	Lambda = 10000
	muscale = MZp
	inducedcouplingfactor = (alpha_e/((1-sw2)*np.pi))*(gV1x1 + gV2x2 + gV3x3)*np.log(Lambda/muscale)

	# Define Zp model parameter point
	#
	Zp_model = {
	  'MZp': MZp,  #Zp mass
	  'mDM': 0.,   #Dark matter mass
	
	  # Zp-DM coupling
	  'gxv': 0.,     #Zp-DM vector coupling
	  'gxa': 0.,     #Zp-DM axial coupling
	
	  # Zp lepton coupling
	  'gev': gV1x1 + (-1/2)*inducedcouplingfactor,   #Zp-el vector coupling
	  'gmv': gV2x2 + (-1/2)*inducedcouplingfactor,   #Zp-mu vector coupling
	  'gtv': gV3x3 + (-1/2)*inducedcouplingfactor,   #Zp-ta vector coupling
	  'gea': gA1x1 + (-1/6)*inducedcouplingfactor,   #Zp-el axial-vector coupling
	  'gma': gA2x2 + (-1/6)*inducedcouplingfactor,   #Zp-mu axial-vector coupling
	  'gta': gA3x3 + (-1/6)*inducedcouplingfactor,   #Zp-ta axial-vector coupling

	  # Zp-quark coupling
	  'guv': (5/18)*inducedcouplingfactor,	#Zp-up-type-quark vector coupling
	  'gdv': (-1/18)* inducedcouplingfactor,#Zp-down-type-quark vector coupling
	  'gua': (1/6)*inducedcouplingfactor,   #Zp-up-type-quark axial coupling
	  'gda': (-1/6)*inducedcouplingfactor,  #Zp-down-type-quark axial coupling
	}

	# The couplings to neutrinos follow from SM gauge invariance and the fact that right-handed neutrinos do not exist
	Zp_model['gnev'] = 0.5 * (Zp_model['gev'] - Zp_model['gea'])
	Zp_model['gnea'] = 0.5 * (Zp_model['gea'] - Zp_model['gev'])
	Zp_model['gnmv'] = 0.5 * (Zp_model['gmv'] - Zp_model['gma'])
	Zp_model['gnma'] = 0.5 * (Zp_model['gma'] - Zp_model['gmv'])
	Zp_model['gntv'] = 0.5 * (Zp_model['gtv'] - Zp_model['gta'])
	Zp_model['gnta'] = 0.5 * (Zp_model['gta'] - Zp_model['gtv'])

	if WZp == 'auto':
		Zp_model['Gamma'] = myDecayWidth(Zp_model)
	else:
		Zp_model['Gamma'] = WZp * Zp_model['MZp']
	
	return Zp_model

def getBSMcounts( channel, Zp_model, lumi =139., mllrange = None, withinterference = True):
	''' geturns bin counts as list
	mllrange = [ mll value to be included in first bin, mll value to be included in last bin]
	if mllrange == None: all bins
	counttype= 'expected' or 'observed'
	lumi in fb-1
	'''
	if channel not in ['ee','mm']:
		print 'Chose channel ee or mm'
		return []

	#
	# Calculate differential cross section (including detector efficiency)
	#

	# get Zprime prediction with interference effects
	
	# lambda functions for sigma(mll)
	if withinterference:
		signal_ana = lambda x : xi_function(x, channel ) * mydsigmadmll_wint(x, Zp_model, channel)
	else:
		signal_ana = lambda x : xi_function(x, channel ) * mydsigmadmll(x, Zp_model, channel)

	# get sm expected - this provides lists with the required size
	smsignal = getSMcounts( channel, counttype='expected', lumi =139., mllrange = mllrange)
	bin_low = []
	bin_high= []
	smcounts_expected = []

	for [lowedge, highedge, count] in smsignal:
		bin_low.append( lowedge )
		bin_high.append( highedge )
		# initialize counts to be 0
		smcounts_expected.append(count)
	counts = np.zeros(np.shape(smcounts_expected))

	# select response function and flucutations:
	if channel =='ee':
		response_function = ee_response_function
		upward_fluctuation = ee_upward_fluctuation
		downward_fluctuation = ee_downward_fluctuation
	if channel =='mm':
		response_function = mm_response_function
		upward_fluctuation = mm_upward_fluctuation
		downward_fluctuation = mm_downward_fluctuation

	# loop over all bins of the mll range
	for i, (mll_low, mll_high) in enumerate(zip(bin_low, bin_high)):
		integrand = lambda x, mll_low, mll_high: lumi * signal_ana(x) * response_function(x, mll_low, mll_high)
		# Calculate the counts in each bin, up/down fluctuation change the integration borders!
		counts[i] = integrate.quad(integrand, upward_fluctuation(mll_low), downward_fluctuation(mll_high), args=(mll_low, mll_high), epsabs=1e-30, epsrel = 0.01)[0]

#	# check signal for negative predictions
#	if np.any(counts<0):
#		print 'Negative signal counts'

	if lumi != 139.:
		print 'Warning: scaling up observed counts to higher Luminosity!'
		lumifactor = lumi/139. 
	else:
		lumifactor = 1

	# add sm prediction (from ATLAS search) to get (bkg + sig)
	rangecounts = []
	for i in range( len(bin_low) ):
		rangecounts.append( [bin_low[i], bin_high[i], counts[i]*lumifactor + smcounts_expected[i]] )

	return rangecounts

# use case
if __name__== "__main__":

	print 'Compare expected standard model with Zp model'
	g = 1
	MZp = 1500
	model = 'VV05'
	print 'Zprime mass: ' + str(MZp)
	print 'Zprime g: ' + str(g)
	print 'model: ' + model
	
	#sm_counts = getSMcounts( 'ee', counttype='expected', mllrange = None, lumi =139.)
	#bsm_counts = getBSMcounts( 'ee', g, MZp, lumi =139., mllrange = None, model = 'VV05',  WZp = 'auto', withinterference = True)
	#
	#print 'full range'
	#for i in range(len(sm_counts)):
	#	print sm_counts[i], bsm_counts[i]
	
	sm_counts = getSMcounts( 'ee', counttype='expected', mllrange = [ MZp - 300, MZp + 300], lumi =139.)

	Zp_model = getZpmodel(g, MZp, model = model,  WZp = 'auto')
	print 'Width'
	print Zp_model['Gamma']
	bsm_counts = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange = [ MZp - 300, MZp + 300], withinterference = True)
	
	print 'counts in range [ MZp - 300, MZp + 300]'
	for i in range(len(sm_counts)):
		print sm_counts[i], bsm_counts[i]
	
