######################################################################################################
# What?
#
# 1. creates a histo from ZPEEDmod moudule
# 2. uses RootTools to plot them, with Plot.fromHisto()
#
######################################################################################################
import ROOT
from array import array
import os
import argparse
import sys
import numpy as np

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

from directories.directories import plotdir
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel_sep, getBSMcounts, myDecayWidth

from helpers import getHisto, customratiostyle, ratioStyle, canvasmod, histmod

#
# argparser
# 
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--ge',      type=float, default=0.9,  help='ge coupling', )
argParser.add_argument('--gm',      type=float, default=0.8,  help='gm coupling', )
argParser.add_argument('--M',       type=float, default=1000,  help='Resonance mass', )
argParser.add_argument('--model',   default='VV',  help='Model type: VV, RR, LL, RL, LR', )
argParser.add_argument('--int',       	action='store_true',     help='Include interference', )
argParser.add_argument('--range',   type=float, default=5.,  help='Plot range -xGamma', )
args = argParser.parse_args()

#
# Define Zp_model
#
ge= args.ge
gm= args.gm
MZp = args.M
model = args.model
Zp_model = getZpmodel_sep(ge ,gm , MZp, model = model,  WZp = 'auto')
print Zp_model
#createtag
nametags = Zp_model['name'].split('_')
gecoupling = '1.' if ge == 1 else str( args.ge ).strip('0')
gmcoupling = '1.' if gm == 1 else str( args.gm ).strip('0')
nametag = nametags[0] + nametags[1] + '/' + gecoupling + '/' + gmcoupling

#
# Plot directory
#
plotdirname = args.model + '_' + str(int(args.M)) + '_int' if args.int else args.model + '_' + str(int(args.M))
plotdirectory = os.path.join( plotdir, 'FinalPlots', plotdirname )
if not os.path.exists( plotdirectory ):
	os.makedirs(   plotdirectory )
print 'plotdirectory: ', plotdirectory

# plot_directory for files created by this very script
plot_directory = os.path.join( plotdirectory, 'Histos' )
if not os.path.exists(plot_directory):
	os.makedirs(plot_directory) 

subdir = Zp_model['name'] + '_W' + str(int(Zp_model['Gamma'])) + '_int' if args.int else Zp_model['name'] + '_W' + str(int(Zp_model['Gamma'])) 
plot_directory = os.path.join( plot_directory, subdir )
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory
plotnamebase = subdir

# old
#plotdirname = 'Zpeedhistos'
#subdir = Zp_model['name'] + '_int' if args.int else Zp_model['name']
#plot_directory = os.path.join( plotdir, plotdirname, subdir )
#if not os.path.exists(plot_directory):
#    os.makedirs(plot_directory) 
#print 'plot_directory: ', plot_directory
#plotnamebase = subdir

#
# Plotrange, for log plot
#
range_down = args.range * Zp_model['Gamma']
M_down = MZp - range_down
M_up = MZp**2/M_down 
mllrange=[ M_down , M_up]

#
# Styles
# Note: without my tweeks, ratio styles come from numerator histo
# observed styles
sty_ee_obs=	{'LineColor':ROOT.kRed,		'style': 'dotted',	'LineWidth':2, 'errors':True} 
sty_mm_obs=	{'LineColor':ROOT.kBlue,	'style': 'dotted',	'LineWidth':2, 'errors':True} 

# expected styles
sty_ee_exp=	{'LineColor':ROOT.kRed,		'style': 'solid',	'LineWidth':3, } 
sty_mm_exp=	{'LineColor':ROOT.kBlue,	'style': 'solid',	'LineWidth':3, } 

# bsm styles
sty_ee_bsm= 	{'LineColor':ROOT.kRed,		'style': 'dashed',	'LineWidth':2, } 
#		{'LineColor':ROOT.kRed, 	'style': 'dotted', 	'LineWidth':2, }
sty_mm_bsm= 	{'LineColor':ROOT.kBlue,	'style': 'dashed', 	'LineWidth':2, } 
#		{'LineColor':ROOT.kBlue,	'style': 'dotted', 	'LineWidth':2, }, ]

# ratio styles
# observed 
ratiostyles_observed = []
ratiostyles_observed.append(lambda histo: customratiostyle( histo, LineColor = ROOT.kGray, 	LineWidth = 3,	style = 'solid')) 
ratiostyles_observed.append(lambda histo: customratiostyle( histo, LineColor = ROOT.kBlack, 	LineWidth = 1, 	style = 'solid')) 
# scan
ratiostyles_scan = []
ratiostyles_scan.append(lambda histo: 	  customratiostyle( histo, LineColor = ROOT.kViolet, 	LineWidth = 1, 	style = None)) 
ratiostyles_scan.append(lambda histo: 	  customratiostyle( histo, LineColor = ROOT.kGreen, 	LineWidth = 1, 	style = None)) 
ratiostyles_scan.append(lambda histo: 	  customratiostyle( histo, LineColor = ROOT.kGreen, 	LineWidth = 1, 	style = None)) 

#
# helpers
#
def drawObjects( hasData = False ):
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextSize(0.04)
    tex.SetTextAlign(11) # align right
    lines = [
    #  (0.15, 0.95, 'Calculation'), 
      (0.60, 0.95, 'L=%3.1f fb{}^{-1} (13 TeV)' % 139)
    ]
    return [tex.DrawLatex(*l) for l in lines] 

#
# observed vs expeced plots, one plot with standard model prediction and observed events
#
# expected
ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
ee_histo_expected = getHisto( ee_expected, 'e/expected', **sty_ee_exp )
mm_histo_expected = getHisto( mm_expected, 'm/expected', **sty_mm_exp )
#ee_histo_expected[0].drawOption = "e1"
#mm_histo_expected[0].drawOption = "e1"

# observed
ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
ee_histo_observed = getHisto( ee_observed, 'e/observed', **sty_ee_obs)
mm_histo_observed = getHisto( mm_observed, 'm/observed', **sty_mm_obs)

# append them
histos = []
histos.append( ee_histo_expected )# histo 0
histos.append( mm_histo_expected )# histo 1
histos.append( ee_histo_observed )# histo 2
histos.append( mm_histo_observed )# histo 3

plot_observed = Plot.fromHisto( 'OBS' , histos, texX = 'M_{ll} (GeV)', texY = 'Number of Events') 

# Ratio
ratiodef =  {	'histos': [(0,1),(2,3)],
   		'logY':  False,
		'style': ratiostyles_observed, #my change: function or list of functions (take histo as argument)
		'texY': '(e / \mu)',
		'yRange': (0.75, 2.75),
		'drawObjects':[]}

plotting.draw( plot_observed,
	    plot_directory = os.path.join( plot_directory, plotnamebase + '_observed_ratio'),
    	    #ratio = {'histos':[(1,0),(3,2)], 'logY':False, 'style':ratiostyles, 'texY': 'mm / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    ratio =  ratiodef,
	    logX = True, logY = True, sorting = True,
	    #yRange = (3., "auto") if log else (0.001, "auto"),
	    yRange = (0.2, "auto") if True else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot_observed.histos))/2,1.,0.9), 2),
	    drawObjects = drawObjects( ),
	    histModifications = [ histmod ], # SetMoreLogLabels
      )

#RatioRatio
ratioratiodef =  {	'histos': [[(0,1),(0,1)],[(2,3),(0,1)]], # draw order is reversed
   			'logY':  False,
			'style': ratiostyles_observed, #my change: function or list of functions (take histo as argument)
			#'texY': '(\mu / e)_{O} / (\mu / e)_{E}',
			'texY': '(e / \mu)',
			'yRange': (0, 2),
			'drawObjects':[]}

plotting.draw( plot_observed,
	    plot_directory = os.path.join( plot_directory, plotnamebase + '_observed_ratioratio'),
    	    #ratio = {'histos':[(1,0),(3,2)], 'logY':False, 'style':ratiostyles, 'texY': 'mm / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    ratio =  ratioratiodef,
	    logX = True, logY = True, sorting = True,
	    #yRange = (3., "auto") if log else (0.001, "auto"),
	    yRange = (0.2, "auto") if True else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot_observed.histos))/2,1.,0.9), 2),
	    drawObjects = drawObjects( ),
	    histModifications = [ histmod ], # SetMoreLogLabels
      )

#
# expected vs bsm parameter points plots
#

ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
# expected histos
ee_histo_expected = getHisto( ee_expected, 'e/expected', LineColor=ROOT.kRed )
mm_histo_expected = getHisto( mm_expected, 'm/expected', LineColor=ROOT.kBlue)
histos = []
histos.append( ee_histo_expected ) # histo 0
histos.append( mm_histo_expected ) # histo 1
ee_signal   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = args.int)
mm_signal   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = args.int)
histos.append( getHisto( ee_signal  , 'e/' + nametag , **sty_ee_bsm) )
histos.append( getHisto( mm_signal  , 'm/' + nametag , **sty_mm_bsm) )

plot_bsm =  Plot.fromHisto( 'BSM' , histos, texX = 'M_{ll} (GeV)', texY = 'Number of Events') 

#Ratio
#ratiodef =  {	'histos': [(0,1),(2,3)], #(4,5)],
#   		'logY':  False,
#		'style': ratiostyles_scan, #my change: function or list of functions (take histo as argument)
#		'texY': '(e / \mu)',
#		'yRange': (0, 2),
#		'drawObjects':[]}
		
plotting.draw( plot_bsm,
	    plot_directory = os.path.join( plot_directory, plotnamebase + '_ratio'),
    	    #ratio = {'histos':[(1,0),(3,2)], 'logY':False, 'style':ratiostyles, 'texY': 'mm / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    ratio =  ratiodef,
	    logX = True, logY = True, sorting = True,
	    #yRange = (3., "auto") if log else (0.001, "auto"),
	    yRange = (0.2, "auto") if True else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot_observed.histos))/2,1.,0.9), 2),
	    drawObjects = drawObjects( ),
	    histModifications = [ histmod ], # SetMoreLogLabels
      )
		
##RatioRatio
#ratioratiodef =  {	'histos': [[(2,3),(0,1)]], #[(5,4),(1,0)]],
#   			'logY':  False,
#			'style': ratiostyles_scan, #my change: function or list of functions (take histo as argument)
#			'texY': '(e / \mu)',
#			'yRange': (0, 2),
#			'drawObjects':[]}

plotting.draw( plot_bsm,
	    plot_directory = os.path.join( plot_directory, plotnamebase + '_ratioratio'),
    	    #ratio = {'histos':[(1,0),(3,2)], 'logY':False, 'style':ratiostyles, 'texY': 'mm / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    ratio =  ratioratiodef,
	    logX = True, logY = True, sorting = True,
	    #yRange = (3., "auto") if log else (0.001, "auto"),
	    yRange = (0.2, "auto") if True else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot_observed.histos))/2,1.,0.9), 2),
	    drawObjects = drawObjects( ),
	    histModifications = [ histmod ], # SetMoreLogLabels
      )
