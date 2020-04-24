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

from stathelpers import CLsZpeed_withexpected, CLspython_ratio_Tevatron_gauss

argParser = argparse.ArgumentParser(description = "Argument parser")
#argParser.add_argument('--plot_directory',     action='store',      default='OneMass_exclusion_test')
argParser.add_argument('--small',       action='store_true',     help='Only four parameter points', )
argParser.add_argument('--M',       	type=float, default=500.,help='Mass point', )
argParser.add_argument('--model',       default='VV' ,help='Mass point', )
argParser.add_argument('--int',       	action='store_true',     help='Include interference', )
argParser.add_argument('--points',      type=int, default=10,    choices=[ 10, 20], help='Number of couplings per dimension', )
argParser.add_argument('--window',   type=float, default=3.,  help='Search window -xGamma', )
args = argParser.parse_args()

#
# directories
#
# plotdirname also for Teststatistik
plotdirname = args.model + '_' + str(int(args.M)) + '_int' if args.int else args.model + '_' + str(int(args.M))
if args.small: plotdirname = plotdirname + '_small'
plotdirectory = os.path.join( plotdir, 'FinalPlots', plotdirname )
if not os.path.exists( plotdirectory ):
	os.makedirs(   plotdirectory )
print 'plotdirectory: ', plotdirectory

# plot_directory for files created by this very script
plot_directory = os.path.join( plotdirectory, 'Exclusion_' + '_P' + str(args.points) + '_SWpm' + str(int(args.window*10)) )
if not os.path.exists(plot_directory):
	os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory

#
# define stat methods
#
def CLspython_ratio_Tevatron_gauss_wrapper( Zp_model,  searchwindow=[-3.,3.], withint = True ):
	return CLspython_ratio_Tevatron_gauss( Zp_model,  searchwindow=searchwindow, withint = withint, plotname= None, plotdirectory = plotdirectory, N=10**4, hilumi=False )

methods = {	
		'CLs_mll':	CLsZpeed_withexpected,
		'CLS_ratio': 	CLspython_ratio_Tevatron_gauss_wrapper,
		}

colors = ['tomato','seagreen','gold']
models = [ args.model ]

#
# Create a grid
#
masslist= [args.M]
gelist = np.linspace( 0, 1., args.points + 1) #smallest 0, largest 1, separation 0.1
gmlist = np.linspace( 0, 1., args.points + 1)
if args.small:
	gelist = np.linspace( 0, 1., 2)
	gmlist = np.linspace( 0, 1., 2)
GE, GM = np.meshgrid(gelist, gmlist)
np.savetxt( os.path.join( plot_directory, 'GE'), GE )
np.savetxt( os.path.join( plot_directory, 'GM'), GM )
nrpoints = np.product(np.shape(GE))
print 'Model points: ' + str( nrpoints )

#
# Make figure
#
for model in models:
	#
	# loop over masslist
	#
	for MZp in masslist:
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

		for Z, E1, E2, E3, E4, E5, method in zip(Zs, E1s, E2s, E3s, E4s, E5s, methods.keys()):
			np.savetxt( os.path.join( plot_directory, plotdirname +  '_Z_' + method), Z )
			np.savetxt( os.path.join( plot_directory, plotdirname + '_E1_' + method), E1 )
			np.savetxt( os.path.join( plot_directory, plotdirname + '_E2_' + method), E2 )
			np.savetxt( os.path.join( plot_directory, plotdirname + '_E3_' + method), E3 )
			np.savetxt( os.path.join( plot_directory, plotdirname + '_E4_' + method), E4 )
			np.savetxt( os.path.join( plot_directory, plotdirname + '_E5_' + method), E5 )

	fig, ax = plt.subplots()
	contourplots = []
	# contour plot
	for Z, E1, E2, E3, E4, E5, method, color in zip(Zs, E1s, E2s, E3s, E4s, E5s, methods.keys(), colors):
		# exclusion line
		levels = [0, 0.05]
		contourplots.append(ax.contour(GE, GM, Z, levels,  colors=color, linewidths=2., linestyles='solid'   ))
		contourplots.append(ax.contour(GE, GM, E1, levels, colors=color, linewidths=1., linestyles='dotted' ))
		contourplots.append(ax.contour(GE, GM, E2, levels, colors=color, linewidths=1., linestyles='dashdot' ))
		contourplots.append(ax.contour(GE, GM, E3, levels, colors=color, linewidths=1., linestyles='dashed'))
		contourplots.append(ax.contour(GE, GM, E4, levels, colors=color, linewidths=1., linestyles='dashdot' ))
		contourplots.append(ax.contour(GE, GM, E5, levels, colors=color, linewidths=1., linestyles='dotted' ))
	#	# filles exclusion area
	#	contour_filled = plt.contourf(GE, GM, Z, levels, colors=['grey','cyan'])

	# label each method	
	#ax.legend([h.legend_elements()[0][0] for h in contourplots],  methods.keys() )
	ax.legend([contourplots[0].legend_elements()[0][0], contourplots[6].legend_elements()[0][0]],  methods.keys() )
	plt.title( model + ' ' + str(int(masslist[0])) + ' (points: ' + str(len(gelist)) + 'x' + str(len(gmlist)) + ')')
	plt.xlabel(r'$g_e$')
	plt.ylabel(r'$g_\mu$')
	plt.grid(True)

	plt.savefig( os.path.join( 		plot_directory, plotdirname  + '.pdf') )
	print 'Exclusion plot saved as ', os.path.join( plot_directory, plotdirname  + '.pdf')
