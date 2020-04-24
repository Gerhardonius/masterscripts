import ROOT
from array import array
import os

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
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel, getBSMcounts 

plotdirname = 'Zpeedhistos'

models = [      'VV01','VV03','VV05'
		#'VV01', 'VV03','VV05',
		#'LL01', 'LL03','LL05',
		#'LR01', 'LR03','LR05',
		#'RR01', 'RR03','RR05',
		#'RL01', 'RL03','RL05',
		]
MZps = [1000,1500,2000]
gs = [1,0.75,0.5]
#gs = [1]

# define histo function
def getHisto( signal, name, LineColor = ROOT.kRed, style=None ):
	binning_low = array('d')
	for i in signal:
		binning_low.append( i[0])
	binning_low.append( signal[-1][1] )

	histo_signal   = ROOT.TH1D( name, name,len( signal ) , binning_low)
	#Line props
        histo_signal.SetLineColor( LineColor )
        histo_signal.SetLineWidth(2)

	if style == 'dotted': histo_signal.SetLineStyle( 3 )
	if style == 'dashed': histo_signal.SetLineStyle( 7 )
	if style == 'chained': histo_signal.SetLineStyle( 4 )
	#Marker
        #histo_signal.SetMarkerSize( markerSize )
        #histo_signal.SetMarkerStyle( markerStyle )
        #histo_signal.SetMarkerColor( color )
	# Additional
        histo_signal.drawOption = "hist"
        #if errors: histo_signal.drawOption+='E' #maybe also try 'e1'
        #histo_signal.drawOption+='E' #maybe also try 'e1'
        #histo_signal.drawOption+='e1' #maybe also try 'e1'
	histo_signal.legendText = name

	for i in range(len(signal)):
		# ROOT bins start with 1
		histo_signal.SetBinContent(    i+1, signal[i][2])
	return [histo_signal]

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

def ratioStyle( histo ):
	color = ROOT.kBlack
	markerStyle = 20
 	markerSize = 1
	width = 1
        #histo.SetLineColor( color )
        histo.SetMarkerSize( markerSize )
        histo.SetMarkerStyle( markerStyle )
        #histo.SetMarkerColor( color )
        #histo.SetFillColor( color )
        histo.SetLineWidth( width )
        histo.drawOption = "e1"
        #histo.drawOption = "E"
        return 

# takes list of Plot as argument
def drawPlots(plots, plotdirname, model):
  for log in [False, True]:
    if log: subDir = "log"
    else: subDir ="linear"
    plot_directory_ = os.path.join( plotdir, plotdirname, model, subDir)
    for plot in plots:
      if not max(l[0].GetMaximum() for l in plot.histos): continue # Empty plot
      numberofhistos = len(plot.histos)
      if (numberofhistos % 2) == 0: 
	      ratiodef = []
	      for i in range(0,numberofhistos,2):
		      ratiodef.append( (i+1,i) )
      plotting.draw(plot,
	    plot_directory = plot_directory_,
    	    #ratio = {'histos':[(1,0),(3,2)], 'logY':False, 'style':None, 'texY': 'mm / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	    ratio = None if not ratiodef else {'histos': ratiodef,
           					'logY':  False,
						'style': ratioStyle,
						'texY': '\mu \mu / ee',
						'yRange': (0, 2),
						'drawObjects':[]},
	#    ratio = None if not ratiodef else {	'gaugedhistos': ratiodef,
	#	    				'logY':False,
	#					'style': ratioStyle,
	#					'texY': '(\mu / e)_{SM-BSM}',
	#					'yRange': (-0.5, 0.5),
	#					'drawObjects':[]},
	    logX = log, logY = True, sorting = True,
	    yRange = (3., "auto") if log else (0.001, "auto"),
	    scaling = {},
	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot.histos))/2,1.,0.9), 2),
	    drawObjects = drawObjects( ),
        #normalize = args.normalize,
      )


# loop it
for model in models:
	plots = []
	for MZp in MZps:
		mllrange=[ MZp -500, MZp + 500]
		ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
		mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
		# expected histos
		ee_histo_expected = getHisto( ee_expected, 'ee_expected', LineColor=ROOT.kRed )
		mm_histo_expected = getHisto( mm_expected, 'mm_expected', LineColor=ROOT.kBlue)
		histos = []
		histos.append( ee_histo_expected )
		histos.append( mm_histo_expected )
		for g,style in zip(gs, ['dashed', 'chained', 'dotted']):
			Zp_model =  getZpmodel(g, MZp, model = model,  WZp = 'auto')
			#width = Zp_model['Gamma']
			ee_signal   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
			mm_signal   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
			histos.append( getHisto( ee_signal  , 'ee_'+ model + '_'+ str(MZp) + '_' + str(g) , LineColor=ROOT.kRed , style=style) )
			histos.append( getHisto( mm_signal  , 'mm_'+ model + '_'+ str(MZp) + '_' + str(g) , LineColor=ROOT.kBlue, style=style) )
		plots.append( Plot.fromHisto( model + '_' + str(MZp) , histos, texX = 'M_{ll} (GeV)', texY = 'Number of Events') )
	drawPlots(plots, plotdirname, model)

