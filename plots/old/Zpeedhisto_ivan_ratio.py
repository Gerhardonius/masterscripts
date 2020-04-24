######################################################################################################
# What?
#
# 1. creates histos from ZPEEDmod moudule
# 2. uses RootTools to plot them, with Plot.fromHisto()
# Note: 2 different plots: 	1. expected vs observed
#				2. expected vs some bsm parameter points
#
######################################################################################################
import ROOT
from array import array
import os
import argparse

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
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel, getBSMcounts, myDecayWidth

from helpers import getHisto, customratiostyle, ratioStyle, getErrors, getratioErrors, canvasmod, getRatioRatioHist_stat, getRatioHist_stat

#
# argparser
# 
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='Zpeedhistos_ivan_test')
args = argParser.parse_args()

#
# Plot directory
#
plotdirname = args.plot_directory

#
# Physics
#
models = [      
		'VV01',
#		'VV03','VV05',
#		'LL01', 'LL03','LL05',
#		'LR01', 'LR03','LR05',
#		'RR01', 'RR03','RR05',
#		'RL01', 'RL03','RL05',
		]
MZps = [2000]
mllrange=[1000,3000]

#
# Styles
# Note: without my tweeks, ratio styles come from numerator histo
# observed styles
sty_ee_obs=	{'LineColor':ROOT.kRed,		'style': 'dotted',	'LineWidth':2, } 
sty_mm_obs=	{'LineColor':ROOT.kBlue,	'style': 'dotted',	'LineWidth':2, } 

# expected styles
sty_ee_exp=	{'LineColor':ROOT.kRed,		'style': 'solid',	'LineWidth':3, } 
sty_mm_exp=	{'LineColor':ROOT.kBlue,	'style': 'solid',	'LineWidth':3, } 

# bsm styles
gs = [1,0.5]
# length must equal length of gs
sty_ee_bsm=[ 	{'LineColor':ROOT.kRed,		'style': 'dashed',	'LineWidth':2, }, 
		{'LineColor':ROOT.kRed, 	'style': 'dotted', 	'LineWidth':2, }, ]
sty_mm_bsm=[ 	{'LineColor':ROOT.kBlue,	'style': 'dashed', 	'LineWidth':2, }, 
		{'LineColor':ROOT.kBlue,	'style': 'dotted', 	'LineWidth':2, }, ]

# ratio styles
# observed 
ratiostyles_observed = []
ratiostyles_observed.append(lambda histo: customratiostyle( histo, LineColor = ROOT.kGray, 	LineWidth = 3,	style = 'solid')) 
ratiostyles_observed.append(lambda histo: customratiostyle( histo, LineColor = ROOT.kViolet, 	LineWidth = 1, 	style = 'solid')) 
# scan
ratiostyles_scan = []
ratiostyles_scan.append(lambda histo: 	  customratiostyle( histo, LineColor = ROOT.kGray, 	LineWidth = 3, 	style = None)) 
ratiostyles_scan.append(lambda histo: 	  customratiostyle( histo, LineColor = ROOT.kViolet, 	LineWidth = 1, 	style = None)) 
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
      (0.15, 0.95, 'Calculation'), 
      (0.60, 0.95, 'L=%3.1f fb{}^{-1} (13 TeV)' % 139)
    ]
    return [tex.DrawLatex(*l) for l in lines] 

# takes list of Plot as argument
def drawPlots_observed(plots, plotdirname, subdir, ratios, ratiotext, ratiostyles):
  '''
	ratios = [[(1,0),(1,0)],(1,0)] - first entry creates ratio of ratio
	filled with e_exp, mu_exp, e_obs, mu_obs
  '''
  for log in [False, True]:
    if log: subsubdir = "log"
    else: subsubdir ="linear"
    plot_directory_ = os.path.join( plotdir, plotdirname, subdir, subsubdir)
    for plot in plots:
      if not max(l[0].GetMaximum() for l in plot.histos): continue # Empty plot
      #numberofhistos = len(plot.histos)
      if ratios:
	    ratiodef =  {	'histos': ratios,
           			'logY':  False,
				#'style': ratiostyles, #my change: function or list of functions (take histo as argument)
				'texY': ratiotext,
				'yRange': (0, 2),
				#'drawObjects':[],
 				#'drawObjects':[
					#getRatioRatioHist_stat( plot, [(2,0),(3,1)], LineColor = ROOT.kCyan, LineWidth = 2, style = 'solid' ),
					#getratioErrors( plot, [(3,2),(1,0)], 'stat', LineColor = ROOT.kCyan, LineWidth = 2, style = 'solid' ),
					#getratioErrors( plot, [(3,2),(1,0)], 'both', LineColor = ROOT.kBlue, LineWidth = 1, style = 'solid' ),
				#		],
			}
      else:
	      ratiodef = {}	
      plotting.draw(plot,
	    plot_directory = plot_directory_,
    	    #ratio = {'histos':[(1,0),(3,2)], 'logY':False, 'style':None, 'texY': 'mm / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    ratio =  ratiodef,
	    logX = log, logY = True, sorting = True,
	    #yRange = (3., "auto") if log else (0.001, "auto"),
	    yRange = (0.2, "auto") if log else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot.histos))/2,1.,0.9), 2),
	    # add errors for e expected and mu expected
            #getErrors(histo,flavor,errortype,LineColor = ROOT.kRed, LineWidth=1, style=None ):
	    drawObjects = drawObjects(), 
	    #+ [getErrors( plot.histos[0][0], 'e', 'stat', LineColor=ROOT.kBlack ), getErrors( plot.histos[0][0], 'e', 'both', LineColor=ROOT.kRed ) ]
	    #+ [getErrors( plot.histos[1][0], 'e', 'stat', LineColor=ROOT.kBlack ), getErrors( plot.histos[1][0], 'e', 'both', LineColor=ROOT.kBlue ) ],
            canvasModifications = [ canvasmod ],
	    )


# takes list of Plot as argument
def drawPlots(plots, plotdirname, subdir, ratios, ratiotext, ratiostyles):
  '''
	ratios = [[(1,0),(1,0)],(1,0)] - first entry creates ratio of ratio
  '''
  for log in [False, True]:
    if log: subsubdir = "log"
    else: subsubdir ="linear"
    plot_directory_ = os.path.join( plotdir, plotdirname, subdir, subsubdir)
    for plot in plots:
      if not max(l[0].GetMaximum() for l in plot.histos): continue # Empty plot
      #numberofhistos = len(plot.histos)
      if ratios:
	    ratiodef =  {	'histos': ratios,
           			'logY':  False,
				#'style': ratiostyles, #my change: function or list of functions (take histo as argument)
				'texY': ratiotext,
				'yRange': (0, 2),
				'drawObjects':[]}
      else:
	      ratiodef = {}	
      plotting.draw(plot,
	    plot_directory = plot_directory_,
    	    #ratio = {'histos':[(1,0),(3,2)], 'logY':False, 'style':None, 'texY': 'mm / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    ratio =  ratiodef,
	    logX = log, logY = True, sorting = True,
	    #yRange = (3., "auto") if log else (0.001, "auto"),
	    yRange = (0.2, "auto") if log else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot.histos))/2,1.,0.9), 2),
	    drawObjects = drawObjects( ),
        #normalize = args.normalize,
      )

#
# observed vs expeced plots
#
# one plot with standard model prediction and observed events
plots = []
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
plots.append( Plot.fromHisto( 'observed' , histos, texX = 'M_{ll} (GeV)', texY = 'Number of Events') )
# Ratio (mm/ee)_O/(mm/ee)_E
drawPlots(plots, plotdirname, 'observed', [(3,2)] , '(\mu / e)_{O} / ( \mu / e)_{E}' ,ratiostyles_observed)
#drawPlots_observed(plots, plotdirname, 'observed', [[(3,2),(1,0)]] , '(\mu / e)_{O} / ( \mu / e)_{E}' ,ratiostyles_observed)
#drawPlots_observed(plots, plotdirname, 'observed', [(3,2)] , '(\mu / e)_{O} / ( \mu / e)_{E}' ,ratiostyles_observed)

#
# expected vs bsm parameter points plots
#
# loop it
for model in models:
	plots = []
	for MZp in MZps:
		#mllrange=[ MZp -500, MZp + 500]
		ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
		mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
		# expected histos
		ee_histo_expected = getHisto( ee_expected, 'e/expected', LineColor=ROOT.kRed )
		mm_histo_expected = getHisto( mm_expected, 'm/expected', LineColor=ROOT.kBlue)
		histos = []
		histos.append( ee_histo_expected )
		histos.append( mm_histo_expected )
		for index, g in enumerate(gs):
			Zp_model =  getZpmodel(g, MZp, model = model,  WZp = 'auto')
			#width = Zp_model['Gamma']
			ee_signal   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
			mm_signal   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
			histos.append( getHisto( ee_signal  , 'e/'+ model + '/'+ str(MZp) + '/' + str(g).strip('0') , **sty_ee_bsm[index]) )
			histos.append( getHisto( mm_signal  , 'm/'+ model + '/'+ str(MZp) + '/' + str(g).strip('0') , **sty_mm_bsm[index]) )
		plots.append( Plot.fromHisto( model + '_' + str(MZp) , histos, texX = 'M_{ll} (GeV)', texY = 'Number of Events') )
	drawPlots(plots, plotdirname, model,ratiostyles_scan)

