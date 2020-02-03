#!/usr/bin/env python
''' Analysis script for standard plots
'''
#
# Standard imports and batch mode
#
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

#
#custom stuff
from directories.directories import flattreedir, plotdir, histdir
#from samples.samples import SM_DYjets_dielec_lo, SM_DYjets_dielec_hi

#
# argparser
# 
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',      default='INFO',          nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory',     action='store',      default='testplots_ratios')
#argParser.add_argument('--samples',            action='store',      nargs='*',               help="Which samples?")
#argParser.add_argument('--selection',          action='store',      default='njet2p-btag1p-relIso0.12-looseLeptonVeto-mll20-met80-metSig5-dPhiJet0-dPhiJet1')
#argParser.add_argument('--normalize',          action='store_true',                          help="Normalize histograms to 1?")
argParser.add_argument('--small',                                   action='store_true',     help='Run only on a small subset of the data?', )
argParser.add_argument('--lumi',      type=int,                        default=137,     help='Luminosity, eg 137', )
args = argParser.parse_args()

#
# logger
#
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

#
# directories
#
plot_directory = os.path.join( plotdir, args.plot_directory )
if args.small: plot_directory += "_small"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory

#
# Samples
#
#name, files, treeName = "Events", normalization = None, xSection = -1, 
#selectionString = None, weightString = None, isData = False, color = 0, texName = None, maxN = None

#
# Version 1
#

#SM
#dielec
SM_DYjets_dielec_hi = Sample.fromFiles("SM_DYjets_dielec_hi", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec_Bkp/PP1/SM_DYjets_dielec_hi.root")
SM_DYjets_dielec_lo = Sample.fromFiles("SM_DYjets_dielec_lo", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec_Bkp/PP1/SM_DYjets_dielec_lo.root")
samples_dielec = [ SM_DYjets_dielec_lo, SM_DYjets_dielec_hi ]
SM_DYjets_dielec = Sample.combine( 'SM_DYjets_dielec', samples_dielec) 

#dimuon
SM_DYjets_dimuon_hi = Sample.fromFiles("SM_DYjets_dimuon_hi", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dimuon/PP1/SM_DYjets_dimuon_hi.root")
SM_DYjets_dimuon_lo = Sample.fromFiles("SM_DYjets_dimuon_lo", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dimuon/PP1/SM_DYjets_dimuon_lo.root")
samples_dimuon = [ SM_DYjets_dimuon_lo, SM_DYjets_dimuon_hi ]
SM_DYjets_dimuon = Sample.combine( 'SM_DYjets_dimuon', samples_dimuon) 

# styles
SM_DYjets_dielec.style = styles.lineStyle( ROOT.kRed, errors = True )
SM_DYjets_dimuon.style = styles.lineStyle( ROOT.kBlue, errors = True )
#SM_DYjets_dielec_lo.style = styles.errorStyle( ROOT.kBlack )

#BSM
#dielec
BSM_DYjets_dielec_hi = Sample.fromFiles("BSM_DYjets_dielec_hi", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/BSM_DYjets_dielec/PP1/BSM_DYjets_dielec_hi.root")
BSM_DYjets_dielec_lo = Sample.fromFiles("BSM_DYjets_dielec_lo", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/BSM_DYjets_dielec/PP1/BSM_DYjets_dielec_lo.root")
samples_dielec = [ BSM_DYjets_dielec_lo, BSM_DYjets_dielec_hi ]
BSM_DYjets_dielec = Sample.combine( 'BSM_DYjets_dielec', samples_dielec) 

#dimuon
BSM_DYjets_dimuon_hi = Sample.fromFiles("BSM_DYjets_dimuon_hi", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/BSM_DYjets_dimuon/PP1/BSM_DYjets_dimuon_hi.root")
BSM_DYjets_dimuon_lo = Sample.fromFiles("BSM_DYjets_dimuon_lo", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/BSM_DYjets_dimuon/PP1/BSM_DYjets_dimuon_lo.root")
samples_dimuon = [ BSM_DYjets_dimuon_lo, BSM_DYjets_dimuon_hi ]
BSM_DYjets_dimuon = Sample.combine( 'BSM_DYjets_dimuon', samples_dimuon) 

# styles
SM_DYjets_dielec.style = styles.lineStyle( ROOT.kRed, errors = True )
SM_DYjets_dimuon.style = styles.lineStyle( ROOT.kBlue, errors = True )
BSM_DYjets_dielec.style = styles.lineStyle( ROOT.kMagenta, errors = True )
BSM_DYjets_dimuon.style = styles.lineStyle( ROOT.kCyan, errors = True )
#SM_DYjets_dielec_lo.style = styles.errorStyle( ROOT.kBlack )

samples = [ SM_DYjets_dimuon, SM_DYjets_dielec, BSM_DYjets_dimuon, BSM_DYjets_dielec,] 
#samples = [ SM_DYjets_dielec_lo, SM_DYjets_dielec_hi ]

#
# Version 2
#
#SM_DYjets_dielec_lo = Sample.fromFiles("SM_DYjets_dielec_lo", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec/PP1/SM_DYjets_dielec_lo.root")
#SM_DYjets_dielec_mi = Sample.fromFiles("SM_DYjets_dielec_mi", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec/PP1/SM_DYjets_dielec_mi.root")
#SM_DYjets_dielec_hi = Sample.fromFiles("SM_DYjets_dielec_hi", "/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec/PP1/SM_DYjets_dielec_hi.root")
#
#SM_DYjets_dielec_lo.style = styles.lineStyle( ROOT.kRed, errors = True )
#SM_DYjets_dielec_mi.style = styles.lineStyle( ROOT.kBlue, errors = True )
#SM_DYjets_dielec_hi.style = styles.lineStyle( ROOT.kGreen, errors = True )
#
#samples = [SM_DYjets_dielec_lo, SM_DYjets_dielec_mi, SM_DYjets_dielec_hi] 

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
      #(0.45, 0.95, 'L=%3.1f fb{}^{-1} (13 TeV) Scale %3.2f'% ( lumi_scale, dataMCScale ) ) if plotData else (0.45, 0.95, 'L=%3.1f fb{}^{-1} (13 TeV)' % lumi_scale)
    ]
    return [tex.DrawLatex(*l) for l in lines] 

# takes list of Plot as argument
def drawPlots(plots):
  for log in [False, True]:
    if log: subDir = "log"
    else: subDir ="linear"
    plot_directory_ = os.path.join(plot_directory, 'gen', plot_directory, subDir)
    for plot in plots:
      if not max(l[0].GetMaximum() for l in plot.histos): continue # Empty plot

      plotting.draw(plot,
	    plot_directory = plot_directory_,
	    #ratio = None, #{'yRange':(0.1,1.9)} if not args.noData else None,
	    #ratio = {'num':1, 'den':0, 'logY':False, 'style':None, 'texY': 'Data / MC', 'yRange': (0.5, 1.5)}
	    ratio = {'num':0, 'den':1, 'logY':False, 'style':None, 'texY': 'ee / \mu\mu', 'yRange': (0.1, 1.9)},
	    logX = False, logY = log, sorting = True,
	    yRange = (3., "auto") if log else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot.histos))/2,1.,0.9), 2),
	    drawObjects = drawObjects( ),
        #normalize = args.normalize,
      )

#
# Read variables and sequences
#

read_variables = [ "dl_mass/F", "weight/F", "l1_pt/F"]

#
# sequences: [somefunction(event, sample),] , dont return anything, def new branch event.new = 
#
sequence = []

#def makestuff(event,sample):
#    event.newstuff = event.l1_pt**2 
#
#sequence.append(makestuff)

#
# apply selection strings to all samples
#
for sample in samples:
    sample.setSelectionString( "(1)" )
    #sample.style = styles.lineStyle(  sample.color )
    #sample.style = styles.lineStyle(  color = ROOT.kRed )

# Let's use a trivial weight. All functions will
#plot_weight   = lambda event, sample : 1
plot_weight   = lambda event, sample : event.weight * 10**3 * args.lumi

#
# no idea
#
weight_         = None
selection       = '(1)'
selectionString = selection

stack = Stack(*[ [ sample ] for sample in samples] ) # not stacked
#stack = Stack(*[  sample  for sample in samples] ) # not stacked
#stack = Stack( samples ) # not working with drawplots

if args.small:
    for sample in stack.samples:
        sample.reduceFiles( to = 1 )

# Use some defaults
Plot.setDefaults(stack = stack, weight = weight_, selectionString = selectionString)#, addOverFlowBin='upper')

#
# Plots
#
#stack = None, attribute = None, binning = None, name = None, selectionString = None, weight = None
#histo_class = None, texX = None, texY = None, addOverFlowBin = None, read_variables = []
  
plots = []

plots.append(Plot( name = "l1_pt",
  texX = 'p_{T}(l1) (GeV)', texY = 'Number of Events / 20 GeV',
  attribute = lambda event, sample: event.l1_pt,
  binning=[2000/40,0,2000],
  weight = plot_weight,
))

plots.append(Plot( name = "Mll_hirange",
  texX = 'M_{ll} (GeV)', texY = 'Number of Events / 20 GeV',
  attribute = lambda event, sample: event.dl_mass,
  binning=[1000/20,1000,2000],
  weight = plot_weight,
))


plots.append(Plot( name = "Mll_all",
  texX = 'M_{ll} (GeV)', texY = 'Number of Events / 10 GeV',
  attribute = lambda event, sample: event.dl_mass,
  binning=[2000/10,0,2000],
  weight = plot_weight,
))

# create histos
plotting.fill(plots, read_variables = read_variables, sequence = sequence, max_events = 100 if args.small else -1)

# check cross section
#pthisto = plots[0].histos[0][0]
##pthisto.GetNbinsX()
#lobin = pthisto.GetXaxis().FindBin(0)
#hibin = pthisto.GetXaxis().FindBin(2000)
#integral =pthisto.Integral( lobin,hibin )
#print "PT l1 histogramm:"
#print "lo hi val: 0 2000 "
#print "lo hi bins: ", lobin, hibin
#print "Number of Events: ", integral
#print "cross section in pb: ", integral/(137*10**3)
#
#counts = []
#for i in range(pthisto.GetNbinsX()):
#    counts.append(pthisto.GetBinContent(i))
#
#print 'max counts', max(counts)
#print 'min counts', min(counts)

drawPlots(plots)
