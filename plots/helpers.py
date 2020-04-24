# helper functions for Zpeed plotting
import ROOT
from array import array
import math

def getHisto( counts, name, LineColor = ROOT.kRed, LineWidth = 3, style=None, errors=False ):
	'''
	returns histogram, from counts (list corresponding to ATLAS binning [[binlow, bin high, count],...])
	name: name tag for histogram
	'''
	binning_low = array('d')
	for i in counts:
		binning_low.append( i[0])
	binning_low.append( counts[-1][1] )

	histo   = ROOT.TH1D( name, name,len( counts ) , binning_low)
	#Line props
        histo.SetLineColor( LineColor )
        histo.SetLineWidth( LineWidth )

	if style == 'solid': histo.SetLineStyle( 1 )
	if style == 'dotted': histo.SetLineStyle( 3 )
	if style == 'dashed': histo.SetLineStyle( 7 )
	if style == 'chained': histo.SetLineStyle( 4 )
	#Marker
        #histo.SetMarkerSize( markerSize )
        #histo.SetMarkerStyle( markerStyle )
        #histo.SetMarkerColor( color )
	# Additional 
        histo.drawOption = "hist ][" #][ removes vertical line from first and last bin
        if errors:
		histo.drawOption+='E' #maybe also try 'e1'
        	histo.SetMarkerSize( 1 )
        	histo.SetMarkerStyle( 20 )
        	histo.SetMarkerColor( LineColor )
        #histo.drawOption+='E' #maybe also try 'e1'
        #histo.drawOption+='e1' #maybe also try 'e1'
	histo.legendText = name

	for i in range(len(counts)):
		# ROOT bins start with 1
		histo.SetBinContent(    i+1, counts[i][2])
	return [histo]

def getHisto_CMScomparison( counts, name, LineColor = ROOT.kRed, LineWidth = 3, style=None ):
	'''
	returns histogram, from counts (list corresponding to ATLAS binning [[binlow, bin high, count],...])
	name: name tag for histogram
	'''
	binning_low = array('d')
	for i in counts:
		binning_low.append( i[0])
	binning_low.append( counts[-1][1] )

	histo   = ROOT.TH1D( name, name,len( counts ) , binning_low)
	#Line props
        histo.SetLineColor( LineColor )
        histo.SetLineWidth( LineWidth )

	if style == 'solid': histo.SetLineStyle( 1 )
	if style == 'dotted': histo.SetLineStyle( 3 )
	if style == 'dashed': histo.SetLineStyle( 7 )
	if style == 'chained': histo.SetLineStyle( 4 )
	#Marker
        #histo.SetMarkerSize( markerSize )
        #histo.SetMarkerStyle( markerStyle )
        #histo.SetMarkerColor( color )
	# Additional 
        #histo.drawOption = "hist ][" #][ removes vertical line from first and last bin
        #if errors: histo.drawOption+='E' #maybe also try 'e1'
        #histo.drawOption+='E' #maybe also try 'e1'
        #histo.drawOption+='e1' #maybe also try 'e1'
	histo.legendText = name

	for i in range(len(counts)):
		# ROOT bins start with 1
		histo.SetBinContent(    i+1, counts[i][2])
	return [histo]

def ratioStyle( histo ):
	color = ROOT.kBlack
	color = histo.GetLineColor()
	markerStyle = 8
 	markerSize = 0.5
	width = 2
        histo.SetMarkerSize( markerSize )
        histo.SetMarkerStyle( markerStyle )
        histo.SetMarkerColor( color )
        #histo.SetFillColor( color )
        histo.SetLineWidth( width )
        histo.drawOption = "e1"
        histo.GetXaxis().SetMoreLogLabels(True)
        #histo.drawOption = "E"
        return 

def customratiostyle( histo, LineColor = ROOT.kRed, LineWidth = 3, style = None ):
	color = LineColor
	markerStyle = 8
 	markerSize = 0.5
	width = 2
        histo.SetMarkerSize( markerSize )
        histo.SetMarkerStyle( markerStyle )
        histo.SetMarkerColor( color )
        histo.SetLineColor( color )
        #histo.SetFillColor( color )
        histo.SetLineWidth( LineWidth )
	if style == 'solid':   histo.SetLineStyle( 1 )
	if style == 'dotted':  histo.SetLineStyle( 3 )
	if style == 'dashed':  histo.SetLineStyle( 7 )
	if style == 'chained': histo.SetLineStyle( 4 )
        histo.drawOption = "e1"
        #histo.GetXaxis().SetMoreLogLabels(True)
	#if errors: histo.drawOption += "E"
        return 

def getErrors( histo, flavor, errortype, LineColor = ROOT.kRed, LineWidth = 1, style = None ):
	'''Returns TGraphAysmmError with errors.
	Errors: stat / syst / both
	stat -> \sqrt(N)
	syst -> defined in getsyst (flavor dependent!)
	both -> /sqrt(stat**2 + syst**2)
	
	flavor = 'e' oder 'm'
	'''
	graph = ROOT.TGraphAsymmErrors( histo ) # copy histo
        graph.SetLineColor( LineColor )
        graph.SetLineWidth( LineWidth )
	if style == 'solid':   graph.SetLineStyle( 1 )
	if style == 'dotted':  graph.SetLineStyle( 3 )
	if style == 'dashed':  graph.SetLineStyle( 7 )
	if style == 'chained': graph.SetLineStyle( 4 )
	for i in range(histo.GetNbinsX()):
	    center    = histo.GetBinCenter(i)
    	    counts    = histo.GetBinContent(i)
	    if errortype=='stat':
	    	syst_up_error, syst_down_error = 0, 0
	    	stat_error = math.sqrt( counts )
	    if errortype=='syst':
	    	syst_up_error, syst_down_error = getsyst( center, flavor )
	    	stat_error = 0
	    if errortype=='both':
	    	syst_up_error, syst_down_error = getsyst( center, flavor )
	    	stat_error = math.sqrt( counts )

	    errUp   = math.sqrt( stat_error**2 + syst_up_error**2 )
	    errDown = math.sqrt( stat_error**2 + syst_down_error**2 )
	    graph.SetPoint( i, center, counts ) # point number, centervalue and counts
	    graph.SetPointError( i, 0, 0, errDown, errUp) # location of low and up error in counts
	return graph

def getsyst( mll, flavor ):
	''' returns (syst_up_error, syst_down_error) 
	    flavor = 'e' or 'm'
	'''
	if flavor == 'e':
		return (0.05, 0.08) 
	if flavor == 'm':
		return (0.05, 0.08) 

def getRatioHist_stat( plot, ratiodef, LineColor = ROOT.kRed, LineWidth = 1, style = None ):
	''' ratiodef=[(1,0)]
		
	'''
	# get ratio of each tuple
	tup = ratiodef[0]
	i_num, i_den = tup
	
	num = plot.histos[i_num][0]
	den = plot.histos[i_den][0]
	#h_ratioden = helpers.clone( dennum )
	h_ratio = num.Clone( )
	h_ratio.Divide( den )

	# copy ratio of ratios
	graph = ROOT.TGraphAsymmErrors( h_ratio ) # copy histo
        graph.SetLineColor( LineColor )
        graph.SetLineWidth( LineWidth )
	if style == 'solid':   graph.SetLineStyle( 1 )
	if style == 'dotted':  graph.SetLineStyle( 3 )
	if style == 'dashed':  graph.SetLineStyle( 7 )
	if style == 'chained': graph.SetLineStyle( 4 )
	
	for i in range(h_ratio.GetNbinsX()):
	    center    = h_ratio.GetBinCenter(i)
    	    counts    = h_ratio.GetBinContent(i)
	    if counts != 0:
	    	errornum  = math.sqrt( num.GetBinContent(i)) 
	    	errorden  = math.sqrt( den.GetBinContent(i)) 
	    	error = math.sqrt(
	    	    	   (errornum/ den.GetBinContent(i))**2
	    	    	   + (errorden * ( num.GetBinContent(i)/den.GetBinContent(i)**2))**2
	    	    	    ) 
            	#errUp   = num.GetBinErrorUp(bin)/den.GetBinContent(bin) if counts > 0 else 0
            	#errDown = num.GetBinErrorLow(bin)/den.GetBinContent(bin) if counts > 0 else 0
	    else:
		    error = 0
	    graph.SetPoint( i, center, counts ) # point number, centervalue and counts
	    graph.SetPointError( i, 0, 0, counts-error, counts+error) # location of low and up erro

	return graph

def getRatioRatioHist_stat( plot, ratiodef, LineColor = ROOT.kRed, LineWidth = 1, style = None ):
	''' ratiodef=[(1,0),(1,0)]
	    (obs/exp)e / (obs/exp)_m	
	'''

	# Numerator
	numtup = ratiodef[0]
	i_numnum, i_numden = numtup
	numnum = plot.histos[i_numnum][0]
	numden = plot.histos[i_numden][0]
	#h_rationum = helpers.clone( numnum )
	h_rationum = numnum.Clone( )
	h_rationum.Divide( numden )


	# Denumerator
	dentup = ratiodef[1]
	i_dennum, i_denden = dentup
	dennum = plot.histos[i_dennum][0]
	denden = plot.histos[i_denden][0]
	#h_ratioden = helpers.clone( dennum )
	h_ratioden = dennum.Clone( )
	h_ratioden.Divide( denden )

	# get ratio of tuple ratios
	#h_ratio = helpers.clone( h_rationum )
	h_ratio = h_rationum.Clone( )
	h_ratio.Divide( h_ratioden )

	# copy ratio of ratios
	graph = ROOT.TGraphAsymmErrors( h_ratio ) # copy histo
        graph.SetLineColor( LineColor )
        graph.SetLineWidth( LineWidth )
	if style == 'solid':   graph.SetLineStyle( 1 )
	if style == 'dotted':  graph.SetLineStyle( 3 )
	if style == 'dashed':  graph.SetLineStyle( 7 )
	if style == 'chained': graph.SetLineStyle( 4 )
	
	for i in range(1,h_ratio.GetNbinsX()+1):
	    print i
	    center    = h_ratio.GetBinCenter(i)
    	    counts    = h_ratio.GetBinContent(i)
	    # error numerator: no error in expected!
            numerr   = numnum.GetBinErrorUp( i)/numden.GetBinContent(i)
	    # error denumerator: no error in expected!
            denerr   = dennum.GetBinErrorUp( i)/denden.GetBinContent(i)
	    # set error 
	    a = (1/h_ratioden.GetBinContent(i)) * numerr
	    b = ( h_rationum.GetBinContent(i) / h_ratioden.GetBinContent(i)**2) * denerr
	    error = math.sqrt( a**2 + b**2) 
	    graph.SetPoint( i, center, counts ) # point number, centervalue and counts
	    print error
	    print counts
	    #graph.SetPointError( i, 0, 0, errDown, errUp) # location of low and up erro

	return graph


def getratioErrors( plot, ratiodef, errortype, LineColor = ROOT.kRed, LineWidth = 1, style = None ):
	''' ratiodef=[(1,0),(1,0)]
		
	'''
	# get ratio of each tuple
	numtup = ratiodef[0]
	i_numnum, i_numden = numtup
	dentup = ratiodef[1]
	i_dennum, i_denden = dentup
	
	numnum = plot.histos[i_numnum][0]
	numden = plot.histos[i_numden][0]
	#h_rationum = helpers.clone( numnum )
	h_rationum = numnum.Clone( )
	h_rationum.Divide( numden )
	
	dennum = plot.histos[i_dennum][0]
	denden = plot.histos[i_denden][0]
	#h_ratioden = helpers.clone( dennum )
	h_ratioden = dennum.Clone( )
	h_ratioden.Divide( denden )

	# get ratio of tuple ratios
	#h_ratio = helpers.clone( h_rationum )
	h_ratio = h_rationum.Clone( )
	h_ratio.Divide( h_ratioden )

	# copy ratio of ratios
	graph = ROOT.TGraphAsymmErrors( h_ratio ) # copy histo
        graph.SetLineColor( LineColor )
        graph.SetLineWidth( LineWidth )
	if style == 'solid':   graph.SetLineStyle( 1 )
	if style == 'dotted':  graph.SetLineStyle( 3 )
	if style == 'dashed':  graph.SetLineStyle( 7 )
	if style == 'chained': graph.SetLineStyle( 4 )
	flavor = 'e'
	for i in range(h_ratio.GetNbinsX()):
	    center    = h_ratio.GetBinCenter(i)
    	    counts    = h_ratio.GetBinContent(i)
	    if errortype=='stat':
		numerror = math.sqrt( 
		( math.sqrt(numnum.GetBinContent(i)) / numden.GetBinContent(i) )**2
		+ (numnum.GetBinContent(i) * math.sqrt( numden.GetBinContent(i))
			/ numden.GetBinContent(i)**2 )**2 )
		denerror = math.sqrt( 
		( math.sqrt(dennum.GetBinContent(i)) / denden.GetBinContent(i) )**2
		+ (dennum.GetBinContent(i) * math.sqrt( denden.GetBinContent(i))
			/ denden.GetBinContent(i)**2 )**2 )
		errorratiotatio = math.sqrt( 
				( math.sqrt(h_rationum.GetBinContent(i)) / h_ratioden.GetBinContent(i) )**2
		+ (h_rationum.GetBinContent(i) * math.sqrt( h_ratioden.GetBinContent(i))
			/ h_ratioden.GetBinContent(i)**2 )**2 )



	    	syst_up_error, syst_down_error = 0, 0
	    	stat_error = math.sqrt( counts )

	    #if errortype=='syst':
	    #	syst_up_error, syst_down_error = getsyst( center, flavor )
	    #	stat_error = 0
	    #if errortype=='both':
	    #	syst_up_error, syst_down_error = getsyst( center, flavor )
	    #	stat_error = math.sqrt( counts )

	    errUp   = math.sqrt( stat_error**2 + syst_up_error**2 )
	    errDown = math.sqrt( stat_error**2 + syst_down_error**2 )
	    graph.SetPoint( i, center, counts ) # point number, centervalue and counts
	    graph.SetPointError( i, 0, 0, errDown, errUp) # location of low and up error in counts
	return graph

def canvasmod(c1):
	'''
	modifies canvas to 50% histo and 50% ratio
	for use in plotting.draw( ... , canvasModifications = [ canvasmod ],)
	'''
    	#default_widths = {'y_width':500, 'x_width':500, 'y_ratio_width':200}
    	default_widths = {'y_width':500, 'x_width':500, 'y_ratio_width':400}
        default_widths['y_width'] += default_widths['y_ratio_width']
        scaleFacRatioPad = default_widths['y_width']/float( default_widths['y_ratio_width'] )
        y_border = default_widths['y_ratio_width']/float( default_widths['y_width'] )
	##scaleFacRatioPad = 3.5
	#scaleFacRatioPad = 1
	##y_border = 0.285714285714
	#y_border = 0.5

        topPad = c1.cd(1)
        topPad.SetBottomMargin(0)
        topPad.SetLeftMargin(0.15)
        topPad.SetTopMargin(0.07)
        topPad.SetRightMargin(0.05)
        topPad.SetPad(topPad.GetX1(), y_border, topPad.GetX2(), topPad.GetY2())
        bottomPad = c1.cd(2)
        bottomPad.SetTopMargin(0)
        bottomPad.SetRightMargin(0.05)
        bottomPad.SetLeftMargin(0.15)
        bottomPad.SetBottomMargin(scaleFacRatioPad*0.13)
        bottomPad.SetPad(bottomPad.GetX1(), bottomPad.GetY1(), bottomPad.GetX2(), y_border)
 
def histmod( histo ):
        histo.GetXaxis().SetMoreLogLabels(True)

def sigmaNNLONLO( ):
	'''
	NNLONLOResult from FEWZ
 	Sigma (pb)                  =    0.0663804
	Error (pb)                  =    0.000287699
	'''
	return 0.0663804

def kfactors( mll, fromorder = 'LOLO'):
	'''
	scale factors to scale to QCD=NNLO, EW=NLO, with LO photon-induced channel
	NNLONLOResult
 	Sigma (pb)                  =    0.0663804
	Error (pb)                  =    0.000287699
	'''
	#LOLOResult
 	#Sigma (pb)                  =    5.8249646906586468E-002
 	#Error (pb)                  =    2.3479473026150554E-004

	#k_factors_NNLONLO_divLOLO.dat 
	# Sigma (pb)                  =    1.13958
	# Integration Error (pb)      =    0.006748
	# PDF Error (pb)              =     0.0016863
	#      bin           weight    numerical error       + pdf error       - pdf error

	fewzresultLOLO= [
	[   633.71,           1.14293,         0.0146296,        0.00154739],
	[   699.31,           1.14002,         0.0168865,        0.00173758],
	[   771.70,           1.13653,         0.0199843,        0.00175704],
	[   851.58,           1.14411,         0.0233043,        0.00172754],
	[   939.74,           1.13655,         0.0276172,        0.00173424],
	[  1037.02,           1.13223,         0.0321744,        0.00211361],
	[  1144.37,           1.13041,         0.0382359,        0.00204192],
	[  1262.83,           1.14544,         0.0472154,         0.0020696],
	[  1393.55,           1.12461,         0.0523375,         0.0019766],
	[  1537.81,           1.11666,         0.0645216,        0.00223417],
	[  1696.99,           1.13254,         0.0788112,        0.00225689],
	[  1872.66,           1.15824,          0.101181,        0.00319801],
	[  2066.51,           1.15737,          0.115678,        0.00268008],
	[  2280.43,           1.12946,          0.139462,        0.00303208],
	[  2516.49,           1.11131,          0.156973,        0.00360731],
	[  2777.00,           1.10205,          0.190529,        0.00415401],
	[  3064.47,            1.2212,          0.251611,         0.0239936],
	[  3381.68,           1.18647,          0.364204,         0.0101545],
	[  3731.74,           1.19299,          0.495799,           0.11497],
	[  4118.05,           1.10611,          0.998639,          0.034114],
	]

	#NLOLOResult
	# Sigma (pb)                  =    6.6959865130739565E-002
	# Error (pb)                  =    2.4069359578688575E-004
	#k_factors_NNLONLO_divNLOLO.dat
	# Sigma (pb)                  =    0.991345
	# Integration Error (pb)      =    0.00558451
	# PDF Error (pb)              =   0.000293093
	#      bin           weight    numerical error       + pdf error       - pdf error

	fewzresultNLOLO= [
	[   633.71,          0.997476,         0.0121458,       0.000232178],
	[   699.31,          0.993394,         0.0139627,       0.000364744],
	[   771.70,          0.986194,         0.0163665,       0.000309928],
	[   851.58,          0.998293,         0.0191815,       0.000308954],
	[   939.74,          0.991154,         0.0226816,       0.000265498],
	[  1037.02,           0.98021,         0.0261297,         0.0005731],
	[  1144.37,          0.966547,          0.030483,       0.000405649],
	[  1262.83,          0.986199,         0.0378816,       0.000409093],
	[  1393.55,          0.969681,         0.0421238,       0.000458799],
	[  1537.81,          0.952241,         0.0510669,       0.000509801],
	[  1696.99,          0.962479,         0.0619131,       0.000464505],
	[  1872.66,           0.99096,         0.0799802,        0.00119106],
	[  2066.51,          0.987013,         0.0909011,        0.00102448],
	[  2280.43,          0.976162,          0.111876,        0.00127798],
	[  2516.49,          0.951645,          0.124255,        0.00281668],
	[  2777.00,          0.901004,          0.141042,        0.00224946],
	[  3064.47,          0.996055,          0.183071,         0.0115983],
	[  3381.68,           0.96745,          0.266998,        0.00693478],
	[  3731.74,          0.998284,          0.372585,         0.0961186],
	[  4118.05,          0.919905,          0.897293,         0.0309906],
	]

	if fromorder == 'LOLO': fewzresult=fewzresultLOLO
	if fromorder == 'NLOLO': fewzresult=fewzresultNLOLO

	( mllcenter, kfactor, err_num, err_pdf) = min( fewzresult, key= lambda listentry: abs( listentry[0] - mll))
	return kfactor

#
# RootTools helpers
#
from RootTools.core.Sample import Sample
from RootTools.core.TreeVariable import TreeVariable

def sigmafromSample( sample, fromorder = None ):
	''' sum u all weights in sample
	    to get LO,NNLO: dl_mass is used as measure which k factor to apply
	    ( even if no dl_mass is reconstructed it gives the value closest, so the smallest mll)
	'''
	print 'Calculate sigma from sample ', sample.name
	variables = [ TreeVariable.fromString('dl_mass/F' ), TreeVariable.fromString('weight/F' ) ]

	r = sample.treeReader( variables = variables)
	r.start()
	totalweight = 0
	if fromorder == None:
		while r.run():
			totalweight += r.event.weight
	else:
		while r.run():
			totalweight += kfactors( r.event.dl_mass, fromorder= fromorder) * r.event.weight 

	print 'Sigma from sample ', sample.name
	print 'Sigma: ', totalweight
	return totalweight

#from RootTools.plot.Plot import Plot
#import RootTools.plot.plotting as plotting
#def addhistotoPlot(origPlot, counts):
#	'''
#	Adds an additional histogram to Plot (RootTools instance)
#	Plot: Plot instance
#	counts: counts for the additional histogram
#	
#	Note: not tested!
#	'''
#	if len(counts)!=origPlot.histos[0][0].GetNbinsX():
#        	raise ValueError( "counts do not match binning" )
#	else:
#		#get plots
#		histos = [] # list of [[histo],[histo],..]
#		for histo in origPlot.histos:
#			histos.append([histo[0]])
#		# clone first histo and fill with counts
#		histo = histos[0][0].Clone()	
#		for i in range(histo.GetNbinsX()): # 1 to Nbins-1
#			histo.SetBinContent(i, counts[i-1])
#		histos.append([histo])
#
#	# explicitly set defaults, otherwisw it will not work
#	Plot.setDefaults(stack = None, weight = None, selectionString = None)
#	newplot=Plot.fromHisto( origPlot.name, histos, texX = origPlot.texX, texY = 'Number of Events')
#	# add legend stuff, tricky since it is defined in sample
#	#for orig, new in zip(origPlot.histos,newplot.histos):
#	print 'Check addhistotoPlot function, it has not been tested yet'
#	return newplot

def getsystematics( origplot, nr_sysweight = 0, variable = 'dl_mass/F' ):
	''' returns list of list with minvals, maxvals for every sample in origplot
	for use in drawPlots
	      result = getsystematics( plot, nr_sysweight = 5, )
      	      graphs = getAsymTgraphs ( result, plots[-1])
	'''
	# specify variables
	read_variables = [ 'weight/F', variable ]
	for i in range(1, nr_sysweight + 1):
    		read_variables.append( 'sysweight_' + str(i).zfill(3) + '/F')

	plots = []

	def make_lambda( string_attribute):
		return lambda event, sample: (origplot.weight( event, sample)/event.weight) * getattr( event, string_attribute )
 	# create plots with sys weights	
	for i in range(1, nr_sysweight + 1):
	    plots.append(Plot( name = origplot.name + '_sys_' + str(i).zfill(3),
	      texX = origplot.name + '_sys_' + str(i).zfill(3),
	      texY = 'Number of Events',
	      attribute = origplot.attributes[0],
	      binning= origplot.binning,
	      #weight = lambda event, sample: event.weight ,
	      weight = make_lambda( 'sysweight_' + str(i).zfill(3)),
	    ))

	# create histos
	plotting.fill(plots, read_variables = read_variables)#, sequence = sequence, max_events = -1)

	#for plot in plots:
	#	plotting.draw(plot,
	#	    plot_directory = '/mnt/hephy/pheno/gerhard/Plots',
    	#	    #ratio = {'histos':[(1,0)], 'logY':True, 'style':None, 'texY': '\mu \mu / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
	#	    logX = True, logY = True, sorting = True,
	#	    yRange = (0.2, "auto"), 
	#	    scaling = {},
	#	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot.histos))/2,1.,0.9), 2),
	#	    #drawObjects = drawObjects( ),
	#	    #drawObjects = drawObjects( ) + [ ee_histo_expected[0] , mm_histo_expected[0] ] ,
        #	#normalize = args.normalize,
      	#		)

	# for every sample
	nr_samples = len(plots[0].histos)
	result = []

	# get bins from first histo of origplot
	numberofbins = plots[0].histos[0][0].GetNbinsX()
	for k in range(nr_samples):

		# loop over histos and get min and max values for each bin
		minvals = [0] * numberofbins
		maxvals = [0] * numberofbins
		count = 0
		for i, plot in enumerate(plots):
		    histo = plot.histos[0][0]
		    # initialize min max with first histo
		    for binnr in range( numberofbins ):
		    	if i==0:
		        	count = histo.GetBinContent( binnr )
		            	maxvals[binnr] = count
		            	minvals[binnr] = count
		    # update if lower or higher ones are in the other histos
		    	else:
		        	count = histo.GetBinContent( binnr )
		        	if count > maxvals[binnr]:
		            		maxvals[binnr] = count
		        	if count < minvals[binnr]:
		            		minvals[binnr] = count
	
		result.append( [minvals, maxvals] )

	return result

def getAsymTgraphs( result, plot ):
	''' returns list of TGraphAsyummErrors
		based on result from getsystematics
	'''
	graphs = []
	for i, ( minvals, maxvals) in enumerate(result):
		histo = plot.histos[i][0]
		graph = ROOT.TGraphAsymmErrors(histo)
		counts = [0] * len(minvals)
		for i in range(len(minvals)):
	    		counts[i] = histo.GetBinContent(i)
	    		center    = histo.GetBinCenter(i)
	    		errUp   = maxvals[i] - counts[i] 
	    		errDown = counts[i] - minvals[i] 
	    		graph.SetPoint( i, center, counts[i]) # point number, centervalue and counts
	    		graph.SetPointError( i, 0, 0, errDown, errUp)
		graphs.append(graph)
	return graphs

#
# Statistics
#

# moved to stathelpers.py

#def getsystematics( variable='dl_mass/F', nr_sysweight = 0, lumi = 139., binning=[600/20,1200,1800] ):
#	# specify variables
#	read_variables=[ variable ]
#	for i in range(1, nr_sysweight + 1):
#    		read_variables.append( 'sysweight_' + str(i).zfill(3) + '/F')
#
#	plots = []
#
#	def make_lambda( string_attribute, lumi=None):
#		if lumi:
#			return lambda event, sample: 10**3 * lumi * getattr( event, string_attribute )
#		else:
#			return lambda event, sample: getattr( event, string_attribute )
# 	# create plots with sys weights	
#	for i in range(1, nr_sysweight + 1):
#	    plots.append(Plot( name = variable.split('/')[0] + str(i).zfill(3),
#	      texX = variable.split('/')[0] + str(i).zfill(3), texY = 'Number of Events',
#	      attribute = make_lambda( variable.split('/')[0] ),
#	      binning= binning,
#	      weight = make_lambda( 'sysweight_' + str(i).zfill(3), lumi=lumi ),
#	    ))
#
#	# create histos
#	plotting.fill(plots, read_variables = read_variables)#, sequence = sequence, max_events = -1)
#
#	#for plot in plots:
#	#	plotting.draw(plot,
#	#	    plot_directory = '/mnt/hephy/pheno/gerhard/Plots',
#    	#	    #ratio = {'histos':[(1,0)], 'logY':True, 'style':None, 'texY': '\mu \mu / ee', 'yRange': (0.2, 0.8), 'drawObjects':[]},
#	#	    logX = True, logY = True, sorting = True,
#	#	    yRange = (0.2, "auto"), 
#	#	    scaling = {},
#	#	    legend =  ( (0.17,0.9-0.05*sum(map(len, plot.histos))/2,1.,0.9), 2),
#	#	    #drawObjects = drawObjects( ),
#	#	    #drawObjects = drawObjects( ) + [ ee_histo_expected[0] , mm_histo_expected[0] ] ,
#        #	#normalize = args.normalize,
#      	#		)
#
#	# get bins from firs histo
#	numberofbins = plots[0].histos[0][0].GetNbinsX()
#
#	# loop over histos and get min and max values for each bin
#	minvals = [0] * numberofbins
#	maxvals = [0] * numberofbins
#	count = 0
#	for i, plot in enumerate(plots):
#	    histo = plot.histos[0][0]
#	    # initialize min max with first histo
#	    for binnr in range( numberofbins ):
#	    	if i==0:
#	        	count = histo.GetBinContent( binnr )
#	            	maxvals[binnr] = count
#	            	minvals[binnr] = count
#	    # update if lower or higher ones are in the other histos
#	    	else:
#	        	count = histo.GetBinContent( binnr )
#	        	if count > maxvals[binnr]:
#	            		maxvals[binnr] = count
#	        	if count < minvals[binnr]:
#	            		minvals[binnr] = count
#	return minvals, maxvals
