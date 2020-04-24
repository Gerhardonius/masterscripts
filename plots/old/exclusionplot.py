######################################################################################################
# What?
#
# 1. creates exlusion plots from ZPEEDmod module
# Goal: implement various statistical methods
#	1. Zpeed implementation of CLs
#
#####################################################################################################

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from directories.directories import plotdir
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel, getBSMcounts, myDecayWidth

from stathelpers import CLsZpeed #getHisto, customratiostyle, ratioStyle, getErrors, getratioErrors, canvasmod

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='Zpeedexclusion_test')
#argParser.add_argument('--plot_directory',  action='store', default='test', help="name of plot subdir")
args = argParser.parse_args()

#
# directories
#
plotdirname = 'Exclusionplots'
plot_directory = os.path.join( plotdir, plotdirname, args.plot_directory )
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory


#
# define stat methods
#

# moved to helpers

#def twotimesCLsZpeed( ee_observed, ee_expected, ee_signal, mm_observed, mm_expected, mm_signal):
#	return 2*CLsZpeed( ee_observed, ee_expected, ee_signal, mm_observed, mm_expected, mm_signal)

methods = {	'CLsZpeed':CLsZpeed,
	#	'newmeht': twotimesCLsZpeed,
		}
colors = ['tomato','seagreen','gold']

models = [      'VV01', 
		#'VV03','VV05',
		#'LL01', 'LL03','LL05',
		#'LR01', 'LR03','LR05',
		#'RR01', 'RR03','RR05',
		#'RL01', 'RL03','RL05',
		]

#
# Create a grid
#
masslist= np.linspace( 500., 2500., 10)
glist   = np.linspace( 0.1, 1., 5)
M, G = np.meshgrid(masslist, glist)
nrpoints = np.product(np.shape(M))
print 'Model points: ' + str( nrpoints )

for model in models:
	print 'Working on: ', model
	#
	# calculate and plot for every mass point
	#

	# prepare Matrix for Zs values (one for each method)
	Zs=[]
	for method in methods.keys():
		Zs.append( np.zeros(np.shape(M)) )
	# perform statistics for every g,M value	
	for Mint,MZp in enumerate(masslist):
		for gint,g in enumerate(glist):
			# define model (needed for mll range)
			Zp_model =  getZpmodel(g, MZp, model = model,  WZp = 'auto')
			#width = Zp_model['Gamma']
			## define mll range
			#mllrange=[ MZp - 3.*width, MZp + 3.*width]
			## SM counts
			#ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
			#ee_observed_bin = [ x[2] for x in ee_observed ]
			#ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
			#ee_expected_bin = [ x[2] for x in ee_expected ]
			#mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
			#mm_observed_bin = [ x[2] for x in mm_observed ]
			#mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
			#mm_expected_bin = [ x[2] for x in mm_expected ]
			## BSM counts
			#ee_signal   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
			#ee_signal_bin = [ x[2] for x in ee_signal ]
			#mm_signal   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = True)
			#mm_signal_bin = [ x[2] for x in mm_signal ]
			# loop over statistical methods
			for Z, method in zip(Zs,methods.keys()):
	        		Z[gint,Mint] = methods[method]( Zp_model, searchwindow = [-3.,3.], withint = True)
				#print Z[gint,Mint]
	#cntr1 =ax.contour(x,y,f2,colors='red' )
	#cntr2 =ax.contour(x,y,f1,colors='blue')
	#h1,_ = cntr1.legend_elements()
	#h2,_ = cntr2.legend_elements()
	#ax.legend([h1[0], h2[0]], ['Contour 1', 'Contour 2'])
	#plt.show()

	fig, ax = plt.subplots()
	contourplots = []
	# contour plot
	for Z, method, color in zip(Zs,methods.keys(),colors):
		# exclusion line
		levels = [0, 0.05]
		contourplots.append(ax.contour(M, G, Z, levels, colors=color))
		# filles exclusion area
		#contour_filled = plt.contourf(M, G, Z, levels, colors=['grey','cyan'])


	# label each method	
	ax.legend([h.legend_elements()[0][0] for h in contourplots],  methods.keys() )
	plt.title( model + ' (points: ' + str(len(masslist)) + 'x' + str(len(glist)) + ')')
	plt.xlabel('Mzp/GeV')
	plt.ylabel('g')
	for xc in masslist:
		plt.axvline(x=xc, color='grey', alpha=0.2)
	for xc in glist:
		plt.axhline(y=xc, color='grey', alpha=0.2)
	#plt.show()
	plt.savefig( os.path.join( plot_directory, model + '.pdf') )
	plt.savefig( os.path.join( plot_directory, model) )
	print 'Figure saved as ', os.path.join( plot_directory, model)
	
