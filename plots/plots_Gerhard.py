#!/usr/bin/env python

##################################################################################################################
#
# Plot script for Flattree samples (postproccessed Delphes trees)
#
# python plots_Gerhard.py --plot_directory=test --small --lumi=139
#
##################################################################################################################
import ROOT
ROOT.gROOT.SetBatch(True)
import argparse
import sys
import os

from math import sqrt, cos, sin, pi, isnan

#RootTools
from RootTools.core.Sample import Sample
from RootTools.core.TreeVariable import TreeVariable
from RootTools.plot.Stack import Stack 
from RootTools.plot.Plot import Plot
from RootTools.plot.Plot2D import Plot2D
import RootTools.plot.styles as styles
import RootTools.plot.plotting as plotting
from RootTools.plot.Binning import Binning
import RootTools.core.logger as logger_rt

#custom stuff
from directories.directories import flattreedir, plotdir, histdir
from ATLASbinning.ATLASbinning import atlasbinning
from helpers import kfactors, getsystematics, getAsymTgraphs, sigmaNNLONLO, sigmafromSample, getHisto_CMScomparison, histmod

#
# Samples
#
from samples.samples import SM_DYjets_dimuon, SM_DYjets_dielec
from samples.samples import VV05_1500_dielec, VV05_1500_dimuon
from samples.samples import VV05_1500_dielec_noPInojet, VV05_1500_dimuon_noPInojet  

from samples.samples import SM_DYjets_dimuon_ATLAS, SM_DYjets_dielec_ATLAS
from samples.samples import VV05_1500_dielec_ATLAS, VV05_1500_dimuon_ATLAS  

from samples.samples import CMS_sampleforlegend_e, CMS_sampleforlegend_m 
from samples.samples import ZPEED_sampleforlegend_e, ZPEED_sampleforlegend_m 
from samples.samples import ZPEED_VV05_1500_sampleforlegend_e, ZPEED_VV05_1500_sampleforlegend_m 

#
# argparser
# 
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',	action='store',      	default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory',     action='store',      default='plots_Gerhard_test')
argParser.add_argument('--small',       action='store_true',     help='Run only on a small subset of the data?', )
argParser.add_argument('--lumi',        type=int,		default=139,     help='Luminosity, eg 139', )
argParser.add_argument('--mode',	action='store',      	default='all', nargs='?', choices=['all','mllsamples','CMS','ZPEED','ZPEEDBSM'], help="plot all variables, or compare to CMS or ZPEED")
args = argParser.parse_args()

#
# logger
#
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

#
# directories
#
plotdirname = 'Plots_Gerhard'
plot_directory = os.path.join( plotdir, plotdirname, args.plot_directory )
if args.small: plot_directory += "_small"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory

#
# compare to ZPEED
# 

# custom binning for samples
#binningspeziale = Binning.fromThresholds([0, 500, 1000, 5000]) # gives 3 bins 
atlasbinningBinning = Binning.fromThresholds( atlasbinning )

from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel, getBSMcounts, myDecayWidth
from helpers import getHisto

mllrange=[ atlasbinning[0] , atlasbinning[-1] ]
# SM:
sty_ee_exp=	{'LineColor':ROOT.kRed,		'style': 'dashed',	'LineWidth':2, } 
sty_mm_exp=	{'LineColor':ROOT.kBlue,	'style': 'dashed',	'LineWidth':2, } 
ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
ee_histo_expected = getHisto( ee_expected, 'e/expected', **sty_ee_exp )
mm_histo_expected = getHisto( mm_expected, 'm/expected', **sty_mm_exp )

# BSM:
sty_ee_bsm=	{'LineColor':ROOT.kRed, 	'style': 'dashed', 	'LineWidth':2, }
sty_mm_bsm=	{'LineColor':ROOT.kBlue,	'style': 'dashed', 	'LineWidth':2, }
g = 1
MZp = 1500
model = 'VV05'
Zp_model =  getZpmodel( g , MZp , model = model,  WZp = 'auto')
#width = Zp_model['Gamma']
ee_signal   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
mm_signal   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
ee_histo_signal = getHisto( ee_signal  , 'e/'+ model + '/'+ str(MZp) + '/' + str(g).strip('0') , **sty_ee_bsm) 
mm_histo_signal = getHisto( mm_signal  , 'm/'+ model + '/'+ str(MZp) + '/' + str(g).strip('0') , **sty_mm_bsm) 

#
# compare to CMS 1803.06292
# 
# custom binning
#binningspeziale = Binning.fromThresholds([0, 500, 1000, 5000]) # gives 3 bins 
cmsbinning = Binning.fromThresholds([600, 900, 1300, 1800, 6000]) # to compare with CMS Resonance search 1803.06292
# note: the first bin actually starts at 600 GeV, 500 is a work around to avoid the vertical line
counts_ee = [ 	[ 600, 900,739 ],
		[ 900,1300,156 ],
		[1300,1800, 30.9],
		[1800,6000,  7],]
counts_mm = [ 	[ 600, 900,1070 ],
		[ 900,1300,220 ],
		[1300,1800,42.6],
		[1800,6000, 9.8],]

ee_histo_CMS = getHisto_CMScomparison( counts_ee, 'e/CMS', LineColor = ROOT.kRed, LineWidth = 2, style='dashed')
mm_histo_CMS = getHisto_CMScomparison( counts_mm, 'm/CMS', LineColor = ROOT.kBlue, LineWidth = 2, style='dashed')

#
# Samples
#
#name, files, treeName = "Events", normalization = None, xSection = -1, 
#selectionString = None, weightString = None, isData = False, color = 0, texName = None, maxN = None

# Set styles for samples
#SM
SM_DYjets_dielec.style = styles.lineStyle( ROOT.kRed, 		errors = False )
SM_DYjets_dimuon.style = styles.lineStyle( ROOT.kBlue, 		errors = False )

#BSM
if args.mode in ['all','mllsamples']:
	VV05_1500_dielec.style = styles.lineStyle( ROOT.kRed,  dashed=True, errors = False )
	VV05_1500_dimuon.style = styles.lineStyle( ROOT.kBlue, dashed=True, errors = False )
	VV05_1500_dielec_noPInojet.style = styles.lineStyle( ROOT.kRed,  dashed=True, errors = False )
	VV05_1500_dimuon_noPInojet.style = styles.lineStyle( ROOT.kBlue, dashed=True, errors = False )
if args.mode == 'ZPEEDBSM':
	VV05_1500_dielec.style = styles.lineStyle( ROOT.kRed,   	errors = False )
	VV05_1500_dimuon.style = styles.lineStyle( ROOT.kBlue,  	errors = False )
	VV05_1500_dielec_noPInojet.style = styles.lineStyle( ROOT.kRed, errors = False )
	VV05_1500_dimuon_noPInojet.style = styles.lineStyle( ROOT.kBlue,errors = False )

# fake styles (to get legend entry)
CMS_sampleforlegend_e.style = styles.fakelineStyle( ROOT.kRed, width=2,dashed=True,	errors = False )
CMS_sampleforlegend_m.style = styles.fakelineStyle( ROOT.kBlue,width=2,dashed=True,	errors = False )
ZPEED_sampleforlegend_e.style = styles.fakelineStyle( ROOT.kRed, width=2,dashed=True,	errors = False )
ZPEED_sampleforlegend_m.style = styles.fakelineStyle( ROOT.kBlue,width=2,dashed=True,	errors = False )
ZPEED_VV05_1500_sampleforlegend_e.style = styles.fakelineStyle( ROOT.kRed, width=2,dashed=True,	errors = False )
ZPEED_VV05_1500_sampleforlegend_m.style = styles.fakelineStyle( ROOT.kBlue,width=2,dashed=True,	errors = False )

if args.mode == 'all':
	#samples = [SM_DYjets_dielec, SM_DYjets_dimuon]
	#samples = [SM_DYjets_dielec, SM_DYjets_dimuon, VV05_2000_dielec_SR, VV05_2000_dimuon_SR]  
	samples = [VV05_1500_dielec, VV05_1500_dimuon, SM_DYjets_dielec, SM_DYjets_dimuon] 
if args.mode == 'CMS': samples = [SM_DYjets_dielec, SM_DYjets_dimuon, CMS_sampleforlegend_e, CMS_sampleforlegend_m]
if args.mode == 'ZPEED':
	#samples = [SM_DYjets_dielec, SM_DYjets_dimuon, ZPEED_sampleforlegend_e, ZPEED_sampleforlegend_m]
	samples = [SM_DYjets_dielec_ATLAS, SM_DYjets_dimuon_ATLAS, ZPEED_sampleforlegend_e, ZPEED_sampleforlegend_m]
if args.mode == 'ZPEEDBSM':
	#samples = [VV05_1500_dielec, VV05_1500_dimuon, ZPEED_VV05_1500_sampleforlegend_e, ZPEED_VV05_1500_sampleforlegend_m]
	#samples = [VV05_1500_dielec_noPInojet, VV05_1500_dimuon_noPInojet, ZPEED_VV05_1500_sampleforlegend_e, ZPEED_VV05_1500_sampleforlegend_m]
	samples = [VV05_1500_dielec_ATLAS, VV05_1500_dimuon_ATLAS, ZPEED_VV05_1500_sampleforlegend_e, ZPEED_VV05_1500_sampleforlegend_m]
if args.mode == 'mllsamples':
	#samples = [VV05_1500_dielec, VV05_1500_dimuon, SM_DYjets_dielec, SM_DYjets_dimuon] 
	samples = [VV05_1500_dielec_noPInojet, VV05_1500_dimuon_noPInojet, VV05_1500_dielec, VV05_1500_dimuon]

#
# scaling
#
sigma = sigmaNNLONLO()
# Note: Not working if more than 2 samples are scaled
#for s in samples:
#	# scales to NNLONLO cross section from sample where k-factors have already been applied
#	#if 'SM-DYjets' in s.name:
#	sigmasample = sigmafromSample(s,fromorder='LOLO')
#	s.scale = sigma/sigmasample
#if not args.small:
if False:
	# dielec
	scalefactor_dielec = sigma/sigmafromSample( SM_DYjets_dielec, fromorder='LOLO')
	SM_DYjets_dielec.scale = scalefactor_dielec 
	VV05_1500_dielec.scale = scalefactor_dielec 
	
	# dimuon
	scalefactor_dimuon = sigma/sigmafromSample( SM_DYjets_dimuon, fromorder='LOLO')
	SM_DYjets_dimuon.scale = scalefactor_dimuon 
	VV05_1500_dimuon.scale = scalefactor_dimuon 

#
# Text on the plots
#
def drawObjects( hasData = False ):
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.SetTextAlign(11) # align right
    lines = [
      (0.15, 0.95, 'Preliminary' if hasData else 'Simulation'), 
      (0.60, 0.95, 'L=%3.1f fb{}^{-1} (13 TeV)' % args.lumi)
    ]
    return [tex.DrawLatex(*l) for l in lines] 
#
# Plot functions
# takes list of Plot as argument
#
def drawPlots(plots, systematics = False):
  '''
  '''
  print 'call'
  for log in [False, True]:
    if log: subDir = "log"
    else: subDir ="linear"
    plot_directory_ = os.path.join( plot_directory, subDir)
    for plot in plots:
      #if not max(l[0].GetMaximum() for l in plot.histos): continue # Empty plot
      # get systematics and corresponding TGraph
      DrawO = drawObjects( ) 
      if systematics:
	      result = getsystematics( plot, nr_sysweight = 5, )
      	      graphs = getAsymTgraphs ( result, plots[-1])
	      DrawO += graphs
      if args.mode == 'CMS':
	      DrawO += [ ee_histo_CMS[0] , mm_histo_CMS[0] ] 
      if args.mode == 'ZPEED':
	      DrawO += [ ee_histo_expected[0] , mm_histo_expected[0] ]
      if args.mode == 'ZPEEDBSM':
	      DrawO += [ ee_histo_signal[0] , mm_histo_signal[0] ]
      plotting.draw(plot,
	    plot_directory = plot_directory_,
    	    #ratio = {'histos':[(1,0)], 'logY':True, 'style':None, 'texY': '\mu \mu / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    # logX only for mll plots (CMS, Zpeed)
	    logX = False if args.mode == 'all' else True, logY = log, sorting = True,
	    yRange = (0.2, "auto") if log else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot.histos))/2,1.,0.9), 2),
	    drawObjects = DrawO,
	    canvasModifications = [],
	    histModifications = [ histmod ], # SetMoreLogLabels
      )

#
# Read variables and sequences
#
read_variables=['l1_pt/F', 'l1_phi/F', 'l1_eta/F', #leading lepton1
          	'l2_pt/F', 'l2_phi/F', 'l2_eta/F', #leading lepton2
                'nLep/I',
                'met_pt/F','met_phi/F', #MET 
                'ht/F', 'nJet/I', # hadronic activity
                'dl_mass/F', 'dl_pt/F', 'dl_phi/F', 'dl_eta/F', 'dPhi_ll/F', #dilepton system
                'weight/F',
		'gen_dl_mass/F', 'gen_dl_pt/F', 'gen_dl_phi/F', 'gen_dl_eta/F', 'gen_dPhi_ll/F', # generator level: dilepton system
                ]

#
# sequences: [somefunction(event, sample),] , dont return anything, def new branch event.new = 
#
sequence = []
#def makestuff(event,sample):
#    event.newstuff = event.l1_pt**2 
#sequence.append(makestuff)

#
# apply selection strings to all samples
#
#for sample in samples:
#    sample.setSelectionString( "(1)" )
#    #sample.setSelectionString( "nJet==2" )
#    #sample.setSelectionString( "dl_mass>1000&&dl_mass<2000" )

# Let's use a trivial weight. All functions will
#plot_weight   = lambda event, sample : 1
plot_weight   = lambda event, sample : event.weight * 10**3 * args.lumi

#
# no idea
#
weight_         = None
#selection       = '(1)'
#selection       = 'dl_mass>600'
#selectionString = selection

stack = Stack(*[ [ sample ] for sample in samples] )

if args.small:
    for sample in stack.samples:
        sample.reduceFiles( to = 1 )

# Use some defaults
#Plot.setDefaults(stack = stack, weight = weight_, selectionString = selectionString, addOverFlowBin='upper')
Plot.setDefaults(stack = stack, weight = weight_)

#
# Plots
#
#stack = None, attribute = None, binning = None, name = None, selectionString = None, weight = None
#histo_class = None, texX = None, texY = None, addOverFlowBin = None, read_variables = []

#
# compare to CMS
#
if args.mode == 'CMS':
	print 'CMS'
	plots = []
	
	plots.append(Plot( name = "dl_mass_kfactorsLOLO",
	  texX = 'M_{ll} (GeV)', texY = 'Number of Events',
	  attribute = lambda event, sample: event.dl_mass,
	  binning= cmsbinning,
	  weight = lambda event, sample : event.weight * 10**3 * args.lumi * kfactors( event.dl_mass, fromorder='LOLO' ), 
	))
	
	plotting.fill(plots, read_variables = read_variables, sequence = sequence, max_events = 100 if args.small else -1)
	drawPlots(plots, systematics=False)

#
# compare to ZPEED
#
if args.mode in ['ZPEED','ZPEEDBSM']:
	print 'ZPEED or ZPEEDBSM'
#print getsystematics( variable='dl_mass/F', nr_sysweight = 5, lumi = 139., binning=atlasbinningBinning )

# mine: log y, ratio 
	plots = []
	
	plots.append(Plot( name = "dl_mass_kfactorsLOLO",
	  texX = 'M_{ll} (GeV)', texY = 'Number of Events',
	  attribute = lambda event, sample: event.dl_mass,
	  binning= atlasbinningBinning,
	  weight = lambda event, sample : event.weight * 10**3 * args.lumi * kfactors( event.dl_mass, fromorder='LOLO' ), 
	))
		
	# gen level
	plots.append(Plot( name = "gen_dl_mass_kfactorsLOLO",
	  texX = 'M_{ll}^{gen} (GeV)', texY = 'Number of Events / 40 GeV',
	  attribute = lambda event, sample: event.gen_dl_mass,
	  binning=[2000/40,750,2750],
	  weight = lambda event, sample : event.weight * 10**3 * args.lumi * kfactors( event.dl_mass, fromorder='LOLO' ), 
	))
	
	plotting.fill(plots, read_variables = read_variables, sequence = sequence, max_events = 100 if args.small else -1)
	drawPlots(plots, systematics=False)

#
# only mll
#
if args.mode == 'mllsamples':
	print 'mllsamples'
#print getsystematics( variable='dl_mass/F', nr_sysweight = 5, lumi = 139., binning=atlasbinningBinning )

# mine: log y, ratio 
	plots = []
	
	plots.append(Plot( name = "dl_mass_kfactorsLOLO",
	  texX = 'M_{ll} (GeV)', texY = 'Number of Events',
	  attribute = lambda event, sample: event.dl_mass,
	  binning= atlasbinningBinning,
	  weight = lambda event, sample : event.weight * 10**3 * args.lumi * kfactors( event.dl_mass, fromorder='LOLO' ), 
	))
		
	# gen level
	plots.append(Plot( name = "gen_dl_mass_kfactorsLOLO",
	  texX = 'M_{ll}^{gen} (GeV)', texY = 'Number of Events / 40 GeV',
	  attribute = lambda event, sample: event.gen_dl_mass,
	  binning=[2000/40,750,2750],
	  weight = lambda event, sample : event.weight * 10**3 * args.lumi * kfactors( event.dl_mass, fromorder='LOLO' ), 
	))
	
	plotting.fill(plots, read_variables = read_variables, sequence = sequence, max_events = 100 if args.small else -1)
	drawPlots(plots, systematics=False)


#
# draw all variables
#
if args.mode == 'all':

	plots = []
	
	#lepton 1
	plots.append(Plot( name = "l1_pt",
	  texX = 'p_{T}(l_{1}) (GeV)', texY = 'Number of Events / 40 GeV',
	  attribute = lambda event, sample: event.l1_pt,
	  binning=[2000/40,0,2000],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = "l1_eta",
	  texX = '#eta(l_{1})', texY = 'Number of Events',
	  attribute = lambda event, sample: abs(event.l1_eta),
	  binning=[15,0,3],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = "l1_phi",
	  texX = '#phi(l_{1})', texY = 'Number of Events',
	  attribute = lambda event, sample: event.l1_phi,
	  binning=[10,-pi,pi],
	  weight = plot_weight,
	))
	
	#lepton 2
	plots.append(Plot( name = "l2_pt",
	  texX = 'p_{T}(l_{2}) (GeV)', texY = 'Number of Events / 40 GeV',
	  attribute = lambda event, sample: event.l2_pt,
	  binning=[2000/40,0,2000],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = "l2_eta",
	  texX = '#eta(l_{2})', texY = 'Number of Events',
	  attribute = lambda event, sample: abs(event.l2_eta),
	  binning=[15,0,3],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = "l2_phi",
	  texX = '#phi(l_{2})', texY = 'Number of Events',
	  attribute = lambda event, sample: event.l2_phi,
	  binning=[10,-pi,pi],
	  weight = plot_weight,
	))
	
	#dilepton
	plots.append(Plot( name = "nLep",
	  texX = 'number of leptons', texY = 'Number of Events',
	  attribute = lambda event, sample: event.nLep,
	  binning=[3,0,3],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = "Mll_kfactorsLOLO",
	  texX = 'M_{ll} (GeV)', texY = 'Number of Events / 40 GeV',
	  attribute = lambda event, sample: event.dl_mass,
	  binning=[2000/40,750,2750],
	  weight = lambda event, sample : event.weight * 10**3 * args.lumi * kfactors( event.dl_mass, fromorder='LOLO' ), 
	))

	plots.append(Plot( name = 'dl_pt', 
	  texX = 'p_{T}(ll) (GeV)', texY = 'Number of Events / 20 GeV',
	  attribute = lambda event, sample: event.dl_pt,
	  binning=[400/20,0,400],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = 'dl_eta', 
	  texX = '#eta(ll)', texY = 'Number of Events',
	  attribute = lambda event, sample: abs(event.dl_eta),
	  binning=[10,0,3],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = 'dl_phi', 
	  texX = '#phi(ll)', texY = 'Number of Events',
	  attribute = lambda event, sample: event.dl_phi,
	  binning=[10,0,3],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = 'deltaPhi_ll',
	  texX = '#Delta#phi(ll)', texY = 'Number of Events',
	  attribute = lambda event, sample:event.dPhi_ll,
	  binning=[10,0,pi],
	  weight = plot_weight,
	))
	
	# MET
	plots.append(Plot( name = 'met_pt',
	  texX = 'E_{T}^{miss}', texY = 'Number of Events',
	  attribute = lambda event, sample:event.met_pt,
	  binning=[2000/40,0,2000],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = 'met_phi',
	  texX = 'phi(E_{T}^{miss})', texY = 'Number of Events',
	  attribute = lambda event, sample:event.met_phi,
	  binning = [10,0,1],
	  weight = plot_weight,
	))
	
	# hadronic activity
	plots.append(Plot( name = 'ht',
	  texX = 'H_{T} (GeV)', texY = 'Number of Events / 25 GeV',
	  attribute = lambda event, sample:event.ht,
	  binning=[500/25,0,600],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = "nJet",
	  texX = 'number of jets', texY = 'Number of Events',
	  attribute = lambda event, sample: event.nJet,
	  binning=[14,0,14],
	  weight = plot_weight,
	))
	
	# gen level
	plots.append(Plot( name = "gen_dl_mass_kfactorsLOLO",
	  texX = 'M_{ll}^{gen} (GeV)', texY = 'Number of Events / 40 GeV',
	  attribute = lambda event, sample: event.gen_dl_mass,
	  binning=[2000/40,750,2750],
	  weight = lambda event, sample : event.weight * 10**3 * args.lumi * kfactors( event.dl_mass, fromorder='LOLO' ), 
	))
	
	plots.append(Plot( name = 'gen_dl_pt', 
	  texX = 'p_{T}^{gen}(ll) (GeV)', texY = 'Number of Events / 20 GeV',
	  attribute = lambda event, sample: event.gen_dl_pt,
	  binning=[400/20,0,400],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = 'gen_dl_eta', 
	  texX = '#eta^{gen}(ll)', texY = 'Number of Events',
	  attribute = lambda event, sample: abs(event.gen_dl_eta),
	  binning=[10,0,3],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = 'gen_dl_phi', 
	  texX = '#phi^{gen}(ll)', texY = 'Number of Events',
	  attribute = lambda event, sample: event.gen_dl_phi,
	  binning=[10,0,3],
	  weight = plot_weight,
	))
	
	plots.append(Plot( name = 'gen_deltaPhi_ll',
	  texX = '#Delta#phi(ll)', texY = 'Number of Events',
	  attribute = lambda event, sample:event.gen_dPhi_ll,
	  binning=[10,0,pi],
	  weight = plot_weight,
	))
	
	def classifyweight(weight):
		if weight == 0.:
			return -0.5
		elif weight > 1.:
			return 1.5
		else:
			return weight
	
	# weight
	plots.append(Plot( name = "weight",
	  texX = 'weight', texY = 'Number of Events',
	  attribute = lambda event, sample: classifyweight( event.weight),
	  binning=[3,-1,2],
	))
	
	plotting.fill(plots, read_variables = read_variables, sequence = sequence, max_events = 1000 if args.small else -1)
	
	drawPlots(plots, systematics=False)

