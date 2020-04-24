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
import sys

from directories.directories import plotdir
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel_sep, getBSMcounts, myDecayWidth

from stathelpers import CLsZpeed

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='Gridexclusion_test')
argParser.add_argument('--small',       action='store_true',     help='Only one mass', )
argParser.add_argument('--int',       	action='store_true',     help='Include interference', )
args = argParser.parse_args()

#
# directories
#
plotdirname = 'Grid_Exclusionplots'
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

models = [      'VV', 
		#'LL',
		#'LR',
		#'RR',
		#'RL',
		]

#
# Create a grid
#
masslist= [500,1000,1500,2000]
gelist = np.linspace( 0, 1., 21) #smallest 0, largest 1, separation 0.05
gmlist = np.linspace( 0, 1., 21)
if args.small:
	masslist= [1000]
	gelist = np.linspace( 0, 1., 6) #smallest 0, largest 1, separation 0.05
	gmlist = np.linspace( 0, 1., 6)
GE, GM = np.meshgrid(gelist, gmlist)
nrpoints = np.product(np.shape(GE))
print 'Model points: ' + str( nrpoints )

#
# Make figure
#

for model in models:
	if args.int: 
		file_name = model + '_int.pdf'
	else:
		file_name = model + '_noint.pdf'
	#
	# Create Figure and list of plots
	#
	fig, axs = plt.subplots(2, 2)
	fig.suptitle( model )
	for ax in axs.flat:
		ax.set(xlabel=r'$g_e$', ylabel=r'$g_{\mu}$')
	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
		ax.label_outer()
	plots = []
	plots.append( axs[0,0] )
	plots.append( axs[0,1] )
	plots.append( axs[1,0] )
	plots.append( axs[1,1] )
	
	for plot, mass in zip(plots,masslist):
		plot.set_title(r'$M_{Zp}=$' + str(mass) + ' GeV')
		plot.grid(True)
	#
	# loop over masslist
	#
	for plot, MZp in zip(plots,masslist):
		print "working on Mzp= ", MZp
		#
		# prepare Matrix for Zs values (one for each method)
		#
		Zs=[]
		for method in methods.keys():
			Zs.append( np.zeros(np.shape(GE)) )
		#
		# loop over couplings,to fill Z
		#
		# perform statistics for every ge, gm value	
		for ge_idx, ge in enumerate(gelist):
			for gm_idx, gm in enumerate(gmlist):
				# define model (needed for mll range)
				Zp_model =  getZpmodel_sep(ge, gm, MZp, model = model,  WZp = 'auto')
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
				#ee_signal   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = args.int )
				#ee_signal_bin = [ x[2] for x in ee_signal ]
				#mm_signal   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = args.int )
				#mm_signal_bin = [ x[2] for x in mm_signal ]
				# loop over statistical methods to fill Z
				for Z, method in zip(Zs,methods.keys()):
 					Z[gm_idx,ge_idx] = methods[method]( Zp_model, searchwindow = [-3.,3.], withint = args.int)
					#print Z[gm_idx,ge_idx]<0.5, gm, ge 
		#
		# plot Z
		#
		# make contour plot
		contourplots = [] # gather contourplots for legend
		for Z, method, color in zip(Zs,methods.keys(),colors):
			# exclusion line
			levels = [0, 0.05]
			contourplots.append(plot.contour(GE, GM, Z, levels, colors=color))
			# filles exclusion area
			contour_filled = plot.contourf(GE, GM, Z, levels, colors=['grey','cyan'])
			# label each method	
			#plot.legend([h.legend_elements()[0][0] for h in contourplots],  methods.keys() )
		plot.legend([h.legend_elements()[0][0] for h in contourplots],  methods.keys() )

	fig.savefig( os.path.join(plot_directory, file_name ))
