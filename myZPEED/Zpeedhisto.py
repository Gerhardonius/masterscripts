#!/usr/bin/env python
import argparse
import numpy as np
import scipy.integrate as integrate
import ROOT
import sys
from array import array

import dileptons_functions as df 
from ATLAS_13TeV_calibration import xi_function
from ATLAS_13TeV import ee_response_function, mm_response_function, ee_upward_fluctuation, mm_upward_fluctuation, ee_downward_fluctuation, mm_downward_fluctuation, calculate_chi2

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--model', action='store', default='VV',	choices=['VV', 'RL',],	help="Zp model")
argParser.add_argument('--g',     action='store', default=1,  	type=float,	help="coupling")
argParser.add_argument('--M',     action='store', default=1000,	type=float,	help="Zp mass")
argParser.add_argument('--w',     action='store', default=0,	type=float,	help="Zp width, if 0 -> Auto")
argParser.add_argument('--SWl',   action='store', default=3,	type=float,	help="Search window lower edge: MZp - SWl * Gamma")
argParser.add_argument('--SWh',   action='store', default=3,	type=float,	help="Search window lower edge: MZp + SWl * Gamma")
argParser.add_argument('--noint', action='store_true',				help="no interference effects")
argParser.add_argument('--plot',  action='store_true',				help="save histograms")
#argParser.add_argument('--plot_directory',     action='store',      default='gen')
#argParser.add_argument('--selection',          action='store',      default='njet2p-btag1p-relIso0.12-looseLeptonVeto-mll20-met80-metSig5-dPhiJet0-dPhiJet1')
args = argParser.parse_args()

#
# plot 
#
if args.noint:
	plotFileName = 'Zpeedhistos/' + '_'.join([str(x) for x in [args.model, args.M, args.g, args.w]]) + '_noint.pdf'
else:
	plotFileName = 'Zpeedhistos/' + '_'.join([str(x) for x in [args.model, args.M, args.g, args.w]]) + '.pdf'

#
# Define Zp model parameter point
#
Zp_model = {
  'MZp': args.M,  #Zp mass
  'mDM': 0.,   #Dark matter mass

  'gxv': 0.,     #Zp-DM vector coupling
  'guv': args.g,    #Zp-up-type-quark vector coupling
  'gdv': args.g,    #Zp-down-type-quark vector coupling
  'glv': args.g,   #Zp-lepton vector coupling

  'gxa': 0.,     #Zp-DM axial coupling
  'gua': 0.,     #Zp-up-type-quark axial coupling
  'gda': 0.,     #Zp-down-type-quark axial coupling
  'gla': 0.,     #Zp-lepton axial coupling
}

# The couplings to binning_lowtrinos follow from SM gauge invariance and the fact that right-handed binning_lowtrinos do not exist
Zp_model['gnv'] = 0.5 * (Zp_model['glv'] - Zp_model['gla'])
Zp_model['gna'] = 0.5 * (Zp_model['gla'] - Zp_model['glv'])

if args.w == 0:
	Zp_model['Gamma'] = df.DecayWidth(Zp_model)
else:
	Zp_model['Gamma'] = args.w * Zp_model['M']

#
# Calculate differential cross section (including detector efficiency)
#

# lambda functions for sigma(mll)
if args.noint:
	ee_signal_ana = lambda x : xi_function(x, "ee") * df.dsigmadmll(x, Zp_model, "ee")
	mm_signal_ana = lambda x : xi_function(x, "mm") * df.dsigmadmll(x, Zp_model, "mm")	
else:
	ee_signal_ana = lambda x : xi_function(x, "ee") * df.dsigmadmll_wint(x, Zp_model, "ee")
	mm_signal_ana = lambda x : xi_function(x, "mm") * df.dsigmadmll_wint(x, Zp_model, "mm")	

#
# Define Signal region
#
Mlow  = Zp_model['MZp'] - args.SWl * Zp_model['Gamma']
Mhigh = Zp_model['MZp'] + args.SWh * Zp_model['Gamma']
signal_range = [Mlow,Mhigh]

# 
# Get histogramms
#

# ATLAS search results, counts
analysis_name = 'ATLAS_13TeV'
lumi = 139.           # in fb^1-

ee_data = np.loadtxt(analysis_name+'/ee_data.dat',delimiter='\t')
mm_data = np.loadtxt(analysis_name+'/mm_data.dat',delimiter='\t')

ee_bin_low = ee_data[:,0]
ee_bin_high = ee_data[:,1]
ee_observed = ee_data[:,2]
ee_expected = ee_data[:,3]

mm_bin_low = mm_data[:,0]
mm_bin_high = mm_data[:,1]
mm_observed = mm_data[:,2]
mm_expected = mm_data[:,3]

# Zp modell for search binning, counts
ee_signal = np.zeros(np.shape(ee_observed))
mm_signal = np.zeros(np.shape(mm_observed))

# Identify bins that cover the requested signal range
i_low = 0 
while ee_bin_low[i_low+1] < signal_range[0] and i_low < len(ee_bin_low)-2: i_low = i_low + 1
i_high = 0
while ee_bin_high[i_high] < signal_range[1] and i_high < len(ee_bin_high)-1: i_high = i_high + 1

# loop over all bins
for i, (mll_low, mll_high) in enumerate(zip(ee_bin_low, ee_bin_high)):
	ee_integrand = lambda x, mll_low, mll_high: lumi * ee_signal_ana(x) * ee_response_function(x, mll_low, mll_high)
	mm_integrand = lambda x, mll_low, mll_high: lumi * mm_signal_ana(x) * mm_response_function(x, mll_low, mll_high)

	# Calculate the counts in each bin, up/down fluctuation change the integration borders!
	#ee_signal[i] = integrate.quad(ee_integrand, mll_low, mll_high, args=(mll_low, mll_high), epsabs=1e-30, epsrel = 0.01)[0]
	#mm_signal[i] = integrate.quad(mm_integrand, mll_low, mll_high, args=(mll_low, mll_high), epsabs=1e-30, epsrel = 0.01)[0]
	ee_signal[i] = integrate.quad(ee_integrand, ee_upward_fluctuation(mll_low), ee_downward_fluctuation(mll_high), args=(mll_low, mll_high), epsabs=1e-30, epsrel = 0.01)[0]
	mm_signal[i] = integrate.quad(mm_integrand, mm_upward_fluctuation(mll_low), mm_downward_fluctuation(mll_high), args=(mll_low, mll_high), epsabs=1e-30, epsrel = 0.01)[0]

# check signal for negative predictions
if np.any(ee_signal<0) or np.any(mm_signal<0):
	print 'Negative signal counts'
	sys.exit()

# histogramm
canvas = ROOT.TCanvas('canvas','Dynamic Filling Example',200,10,700,500)
canvas.SetLogy(True)
canvas.SetLogx(True)
canvas.Print(plotFileName+"[")
#canvas.GetFrame().SetBorderSize(6)
#canvas.cd()

# binnig is based on ee_observed, but this matches mm
# ROOT expects low edges of bins + high edge of last bin -> one element longer than number of bins
binning_low = array('d')
for i in ee_bin_low:
	binning_low.append(i)
binning_low.append(ee_bin_high[-1])

histo_observed = ROOT.TH1D('histo_observed ', 'histo_observed ',len(ee_observed) , binning_low)
histo_expected = ROOT.TH1D('histo_expected ', 'histo_expected ',len(ee_observed) , binning_low)
histo_signal   = ROOT.TH1D('histo_signalcnt', 'histo_signalcnt',len(ee_observed) , binning_low)

histo_signal.SetFillColor(48)
histo_expected.SetFillColor(41)
histo_observed.SetMarkerStyle(21)
histo_observed.SetMarkerColor(ROOT.kBlue)
#histo_signal.Sumw2() # needed for stat uncertainty
#histo_expected.Sumw2() # needed for stat uncertainty

ee_info = [ee_observed, ee_expected, ee_signal]
mm_info = [mm_observed, mm_expected, mm_signal]

# loop over ee and mm
for flavor, [observed, expected, signal] in enumerate([ee_info, mm_info]):

	for i in range(len(ee_observed)):
		# ROOT bins start with 1
		histo_signal.SetBinContent(    i+1, signal[i])
		histo_expected.SetBinContent(  i+1, expected[i])
		histo_observed.SetBinContent(  i+1, observed[i])

	hs = ROOT.THStack('hs','compare histos')
	if flavor == 0:
		hs.SetTitle('ee channel')
	if flavor == 1:
		hs.SetTitle('mm channel')
	hs.Add(histo_expected)
	hs.Add(histo_signal)
	hs.Draw('hist')
	hs.GetXaxis().SetTitle("m_{ll} [MeV]")
	hs.GetYaxis().SetTitle("Number of events")
	#histo_signal.Draw('hist')
	#histo_expected.Draw('hist,Same')
	histo_observed.Draw('PE1,Same')
	boxes = []
	for i in range(i_low, i_high-1):
		boxes.append(ROOT.TBox( ee_bin_low[i], 0, ee_bin_high[i], signal[i]+expected[i]))
		#boxes.append(ROOT.TBox( ee_bin_low[i], 0, ee_bin_high[i], signal[i]))
		shadedYellow=ROOT.TColor.GetColorTransparent(ROOT.kRed,0.35) 
		boxes[-1].SetFillColor(shadedYellow)
		boxes[-1].Draw('Same')
	canvas.Print(plotFileName)

canvas.Print(plotFileName + "]")

#
# statistics
#

# calculate_ch2 involves weights for bins at the ede of the signal region 
chi2, chi2_Asimov = calculate_chi2(ee_signal_ana, mm_signal_ana, signal_range=signal_range)
result = get_likelihood(chi2, chi2_Asimov)

print("-2 log L:       ", result[0])
print("-2 Delta log L: ", result[1])
print("CLs:            ", result[2])
