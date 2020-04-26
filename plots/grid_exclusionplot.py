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

from stathelpers import CLsZpeed_withexpected, CLsRatio_withexpected

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='Gridexclusion_test')
argParser.add_argument('--small',       action='store_true',     help='Only one mass', )
argParser.add_argument('--int',       	action='store_true',     help='Include interference', )
argParser.add_argument('--points',      type=int, default=10,    choices=[ 10, 20], help='Number of couplings per dimension', )
argParser.add_argument('--window',   type=float, default=3.,  help='Search window -xGamma', )
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

def CLsRatio_withexpected_reversed( 	Zp_model, searchwindow= [-3.,3.],	withint = False ):
	return CLsRatio_withexpected( 	Zp_model, searchwindow= searchwindow, 	withint = withint, variant = 'rev' )

def CLsRatio_withexpected_alberto( 	Zp_model, searchwindow= [-3.,3.], 	withint = False ):
	return CLsRatio_withexpected( 	Zp_model, searchwindow= searchwindow, 	withint = withint, variant = 'alb' )


methods = {	'CLs-mll':	CLsZpeed_withexpected,
		'CLs-ratio': 	CLsRatio_withexpected,
		'CLs-ratio-rev':CLsRatio_withexpected_reversed,
		'CLs-ratio-det':CLsRatio_withexpected_alberto,
		}
colors = ['tomato','seagreen','gold','royalblue']

models = [      'VV', 
		#'LL',
		#'LR',
		#'RR',
		#'RL',
		]

#
# Create a grid
#
masslist= [500,750,1000,1250]
gelist = np.linspace( 0, 1., args.points + 1) #smallest 0, largest 1, separation 0.1
gmlist = np.linspace( 0, 1., args.points + 1)
if args.small:
	masslist= [1000]
	gelist = np.linspace( 0, 1., 4) #smallest 0, largest 1, separation 0.05
	gmlist = np.linspace( 0, 1., 4)
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
		E1s=[]
		for method in methods.keys():
			E1s.append( np.zeros(np.shape(GE)) )
		E2s=[]
		for method in methods.keys():
			E2s.append( np.zeros(np.shape(GE)) )
		E3s=[]
		for method in methods.keys():
			E3s.append( np.zeros(np.shape(GE)) )
		E4s=[]
		for method in methods.keys():
			E4s.append( np.zeros(np.shape(GE)) )
		E5s=[]
		for method in methods.keys():
			E5s.append( np.zeros(np.shape(GE)) )
		#
		# loop over couplings,to fill Z
		#
		# perform statistics for every ge, gm value	
		for ge_idx, ge in enumerate(gelist):
			for gm_idx, gm in enumerate(gmlist):
				# define model (needed for mll range)
				Zp_model =  getZpmodel_sep(ge, gm, MZp, model = model,  WZp = 'auto')
				for Z, E1, E2, E3, E4, E5, method in zip(Zs, E1s, E2s, E3s, E4s, E5s, methods.keys()):
					E1[gm_idx,ge_idx], E2[gm_idx,ge_idx], E3[gm_idx,ge_idx], E4[gm_idx,ge_idx], E5[gm_idx,ge_idx], Z[gm_idx,ge_idx]= methods[method]( Zp_model, searchwindow = [- args.window, args.window], withint = args.int)

		#
		# plot Z
		#
		# make contour plot
		contourplots = []
		# contour plot
		for Z, E1, E2, E3, E4, E5, method, color in zip(Zs, E1s, E2s, E3s, E4s, E5s, methods.keys(), colors):
			# exclusion line
			levels = [0, 0.05]
			contourplots.append(plot.contour(GE, GM, Z, levels,  colors=color, linewidths=2., linestyles='solid'   ))
			contourplots.append(plot.contour(GE, GM, E1, levels, colors=color, linewidths=1., linestyles='dotted' ))
			contourplots.append(plot.contour(GE, GM, E2, levels, colors=color, linewidths=1., linestyles='dashdot' ))
			contourplots.append(plot.contour(GE, GM, E3, levels, colors=color, linewidths=1., linestyles='dashed'))
			contourplots.append(plot.contour(GE, GM, E4, levels, colors=color, linewidths=1., linestyles='dashdot' ))
			contourplots.append(plot.contour(GE, GM, E5, levels, colors=color, linewidths=1., linestyles='dotted' ))
		#	# filles exclusion area
		#	contour_filled = plt.contourf(GE, GM, Z, levels, colors=['grey','cyan'])
		#plot.legend([h.legend_elements()[0][0] for h in contourplots],  methods.keys() )
		#plot.legend([contourplots[0].legend_elements()[0][0], contourplots[6].legend_elements()[0][0]],  methods.keys() )
		legendelements = []
		nr = 0
		for method in methods.keys(): 
			legendelements.append(contourplots[nr].legend_elements()[0][0])
			nr += 6
		plot.legend( legendelements,  methods.keys() )

	fig.savefig( os.path.join(plot_directory, file_name ))
	print 'Exclusion plot saved as ', os.path.join( plot_directory, file_name )
