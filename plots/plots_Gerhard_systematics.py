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

#custom stuff
from directories.directories import flattreedir, plotdir
#from samples.samples import SM_DYJets_dielec,SM_DYJets_dimuon

#
# argparser
# 
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--logLevel',           action='store',      default='INFO',          nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--plot_directory',     action='store',      default='testplots')
argParser.add_argument('--small',                                   action='store_true',     help='Run only on a small subset of the data?', )
argParser.add_argument('--sysweight', type=int, default=153,  help='Number of Weight.Weight in Madgraph root files')
args = argParser.parse_args()

#
# logger
#
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

#
# directories
#
plot_directory = os.path.join( plotdir, args.plot_directory )
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory

#
# Samples
#
#name, files, treeName = "Events", normalization = None, xSection = -1, 
#selectionString = None, weightString = None, isData = False, color = 0, texName = None, maxN = None
flattreefiles = [ '/mnt/hephy/pheno/gerhard/Flattree/singularity/MG1/SM_DYjets_dielec/PP1/SM_DYjets_dielec_lo.root' ]
print 'flatttreefiles: ',flattreefiles
samples = [ Sample.fromFiles("", x ) for x in flattreefiles ]

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
	    ratio = None, #{'yRange':(0.1,1.9)} if not args.noData else None,
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

read_variables=['l1_pt/F', 'l1_phi/F', 'l1_eta/F', #leading lepton1
          	'l2_pt/F', 'l2_phi/F', 'l2_eta/F', #leading lepton2
                'nLep/I',
                'met_pt/F','met_phi/F', #MET 
                'ht/F', 'nJet/I', # hadronic activity
                'dl_mass/F', 'dl_pt/F', 'dl_phi/F', 'dl_eta/F', 'dPhi_ll/F', #dilepton system
                'weight/F',
                ]

for i in range(1,args.sysweight + 1):
    read_variables.append( 'sysweight_' + str(i).zfill(3) + '/F')

#
# sequences: [somefunction(event, sample),] , dont return anything, def new branch event.new = 
#

sequence = []

def makestuff(event,sample):
    event.newstuff = event.l1_pt**2 

sequence.append(makestuff)

#
# apply selection strings to all samples
#
for sample in samples:
    sample.setSelectionString( "(1)" )
    #sample.style = styles.lineStyle(  sample.color )
    #sample.style = styles.lineStyle(  color = ROOT.kRed )
    sample.style = styles.errorStyle(  color = ROOT.kBlack )

# Let's use a trivial weight. All functions will
plot_weight   = lambda event, sample : 1
#plot_weight   = lambda event, sample : event.weight

#
# no idea
#
weight_         = None
selection       = '(1)'
#selection       = 'nJet<2'
selectionString = selection

stack = Stack(*[ [ sample ] for sample in samples] )

if args.small:
    for sample in stack.samples:
        sample.reduceFiles( to = 1 )

# Use some defaults
#Plot.setDefaults(stack = stack, weight = weight_, selectionString = selectionString, addOverFlowBin='upper')
Plot.setDefaults(stack = stack, weight = weight_, selectionString = selectionString,)

#
# Plots
#
#stack = None, attribute = None, binning = None, name = None, selectionString = None, weight = None
#histo_class = None, texX = None, texY = None, addOverFlowBin = None, read_variables = []

# 1) get min and max values for each bin in the plot

# create as many plots as there are sysweight 
plots = []

def make_lambda( string_attribute):
    return lambda event, sample: getattr( event, string_attribute )

for i in range(1,args.sysweight + 1):
# dilepton system (leading, subleading)
    plots.append(Plot( name = "M(ll)" + str(i).zfill(3),
      texX = 'M_{ll} (GeV)', texY = 'Number of Events / 20 GeV',
      attribute = lambda event, sample: event.dl_mass,
      binning=[400/20,0,400],
      weight = make_lambda( 'sysweight_' + str(i).zfill(3) ),
    ))

# create histos
plotting.fill(plots, read_variables = read_variables, sequence = sequence, max_events = 100 if args.small else -1)

#if necessary plot histos
#drawPlots(plots)

# loop over histos and get min and max values for each bin
minvals = [100000000] * 20
maxvals = [0] * 20
for i, plot in enumerate(plots):
    histo = plot.histos[0][0]
    for binnr in range(histo.GetNbinsX()):
        counts = histo.GetBinContent( binnr )
        if counts > maxvals[binnr]:
            maxvals[binnr] = counts
        if counts < minvals[binnr]:
            minvals[binnr] = counts

# 2) create histogram of the actual plot
# Make a new plots list with only Mll
plots2 = []
plots2.append(Plot( name = "M(ll)",
  texX = 'M_{ll} (GeV)', texY = 'Number of Events / 60 GeV',
  attribute = lambda event, sample: event.dl_mass,
  binning=[400/20,0,400],
  weight = lambda event, sample: event.weight ,
))

# create histos
plotting.fill(plots2, read_variables = read_variables, sequence = sequence, max_events = 100 if args.small else -1)

#if necessary plot histos
#drawPlots(plots2)

#
# Actual plot
#

# get histogram from mll plot
mllhisto = plots2[0].histos[0][0]
counts = [0]*20

# get the asymmetric errors and set it in TGraphAsymmErrors
graph = ROOT.TGraphAsymmErrors( mllhisto )
for i in range(mllhisto.GetNbinsX()):
    counts[i] = mllhisto.GetBinContent(i)
    center    = mllhisto.GetBinCenter(i)
    errUp   = maxvals[i] - counts[i] 
    errDown = counts[i] - minvals[i] 
    graph.SetPoint( i, center, counts[i]) # point number, centervalue and counts
    graph.SetPointError( i, 0, 0, errDown, errUp)

# plot the TGraphAsymmError
canvas = ROOT.TCanvas('canvas')
canvas.SetLogy(True)
canvas.cd()
# A (Axis), B (Bar chart), P (current marker)
# https://root.cern/doc/master/classTGraphPainter.html#GP01
graph.Draw('PA')
canvas.Print( os.path.join( plot_directory, 'errorplot_log.png'))

print 'min, expected, max count for every bin'
for i in range(20):
    print minvals[i], counts[i], maxvals[i]
