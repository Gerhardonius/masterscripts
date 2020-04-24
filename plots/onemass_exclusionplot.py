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

from stathelpers import CLsZpeed, CLspython_ratio_Tevatron_gauss

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='OneMass_exclusion_test')
argParser.add_argument('--small',       action='store_true',     help='Only four parameter points', )
argParser.add_argument('--M',       	type=float, default=500.,help='Mass point', )
argParser.add_argument('--int',       	action='store_true',     help='Include interference', )
args = argParser.parse_args()

#
# directories
#
plotdirname = 'OneMass_Exclusionplots'
if args.small:
	plot_directory = os.path.join( plotdir, plotdirname, args.plot_directory + '_small' )
else: 
	plot_directory = os.path.join( plotdir, plotdirname, args.plot_directory  )
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory

#
# define stat methods
#

# moved to helpers
# Interface
#CLsZpeed( Zp_model,  searchwindow=[-3.,3.], withint = True ):
#CLspython_ratio_Tevatron_gauss( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2, hilumi=False ):
def CLspython_ratio_Tevatron_gauss_wrapper( Zp_model,  searchwindow=[-3.,3.], withint = True):
	return CLspython_ratio_Tevatron_gauss( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= 'withexpected', N=10**4, hilumi=False )

methods = {	
		#'CLs_mll':CLsZpeed,
		'CLS_ratio': CLspython_ratio_Tevatron_gauss_wrapper,
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
masslist= [args.M]
gelist = np.linspace( 0, 1., 11) #smallest 0, largest 1, separation 0.1
gmlist = np.linspace( 0, 1., 11)

gelist = np.linspace( 0, 0.5, 11) #smallest 0, largest 0.5, separation 0.05
gmlist = np.linspace( 0, 0.5, 11)
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
				#######################################################
				#for Z, method in zip(Zs,methods.keys()):
 				#	Z[gm_idx,ge_idx] = methods[method]( Zp_model, searchwindow = [-3.,3.], withint = args.int)
				#######################################################
				# NEW
				for Z, method in zip(Zs,methods.keys()):
					E1s[0][gm_idx,ge_idx], _, E2s[0][gm_idx,ge_idx], _, E3s[0][gm_idx,ge_idx], Z[gm_idx,ge_idx]= methods[method]( Zp_model, searchwindow = [-3.,3.], withint = args.int)
				# NEW OFF

		#######################################################
		#for Z, method in zip(Zs,methods.keys()):
		#	np.savetxt( os.path.join( plot_directory, model + '_' + method), Z )
		#######################################################
		np.savetxt( os.path.join( plot_directory, model + '_Z'), Zs[0] )
		np.savetxt( os.path.join( plot_directory, model + '_E1'), E1s[0] )
		np.savetxt( os.path.join( plot_directory, model + '_E2'), E2s[0] )
		np.savetxt( os.path.join( plot_directory, model + '_E3'), E3s[0] )


	fig, ax = plt.subplots()
	contourplots = []
	# contour plot
	##########################################################################
	#for Z, method, color in zip(Zs,methods.keys(),colors):
	#	# exclusion line
	#	levels = [0, 0.05]
	#	contourplots.append(ax.contour(GE, GM, Z, levels, colors=color))
	#	# filles exclusion area
	#	contour_filled = plt.contourf(GE, GM, Z, levels, colors=['grey','cyan'])
	##########################################################################
	# NEW
	levels = [0, 0.05]
	contourplots.append(ax.contour(GE, GM, Zs[0], levels, colors= 'black'))
	contourplots.append(ax.contour(GE, GM, E1s[0], levels, colors= 'green'))
	contourplots.append(ax.contour(GE, GM, E2s[0], levels, colors= 'red'))
	contourplots.append(ax.contour(GE, GM, E3s[0], levels, colors= 'blue'))
	# NEW OFF

	# label each method	
	##########################################################################
	#ax.legend([h.legend_elements()[0][0] for h in contourplots],  methods.keys() )
	##########################################################################
	plt.title( model + ' ' + str(int(masslist[0])) + ' (points: ' + str(len(gelist)) + 'x' + str(len(gmlist)) + ')')
	plt.xlabel(r'$g_e$')
	plt.ylabel(r'$g_m$')
	plt.grid(True)
	if args.int: 
		plt.savefig( os.path.join( plot_directory, model + '_' + str(int(masslist[0])) + '_int.pdf') )
		print 'Figure saved as ', os.path.join( plot_directory, model + '_' + str(int(masslist[0])) + '_int.pdf')
	else:
		plt.savefig( os.path.join( plot_directory, model + ' ' + str(int(masslist[0])) + '_noint.pdf') )
		print 'Figure saved as ', os.path.join( plot_directory, model + '_' + str(int(masslist[0])) + '_noint.pdf')
