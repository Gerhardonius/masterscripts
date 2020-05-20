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
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors 
import sys

from directories.directories import plotdir
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel_sep, getZpmodel_semi, getZpmodel_gen, getBSMcounts, myDecayWidth
from ZPEEDmod.ATLAS_13TeV import ee_resolution, mm_resolution # for Gamma_eff 

from stathelpers import CLsZpeed_withexpected, CLsRatio_withexpected

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='Gridexclusion_test')
argParser.add_argument('--small',       action='store_true',     help='Only one mass', )
argParser.add_argument('--int',       	action='store_true',     help='Include interference', )
argParser.add_argument('--points',      type=int, default=10,    choices=[ 10, 20], help='Number of couplings per dimension', )
argParser.add_argument('--window',   type=float, default=2.,  help='Search window -xGammaEff', )
argParser.add_argument('--width',   type=float, default=0.,  help='Width in %, if 0 -> auto', )
argParser.add_argument('--LEP', 	action='store_true',  help='insert LEP limits', )
args = argParser.parse_args()

if args.width==0:
	args.width='auto'
	
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


methods = {	'Resonance search':	CLsZpeed_withexpected,
		'Ratio search': 	CLsRatio_withexpected,
		#'CLs-ratio-rev':CLsRatio_withexpected_reversed,
		#'CLs-ratio-det':CLsRatio_withexpected_alberto,
		}
colors = ['tomato','seagreen','gold','mediumorchid']

models = [      'VV', 
		#'LL',
		#'LR',
		#'RR',
		#'RL',
		]

lepcolor = 'royalblue'
LEPlimits = {	'VV': {'ge': lambda mzp: np.sqrt(2*np.pi)*mzp/20600. 	, 'gm': lambda mzp, ge: 9. if ge==0 else (4*np.pi*(mzp/18900.)**2)/ge},
		'LL': {'ge': lambda mzp: np.sqrt(np.pi/2.)*mzp/8700.	, 'gm': lambda mzp, ge: 9. if ge==0 else (np.pi*(mzp/12200.)**2)/ge},
		'LR': {'ge': lambda mzp: np.sqrt(np.pi/2.)*mzp/8700.	, 'gm': lambda mzp, ge: 9. if ge==0 else (np.pi*(mzp/9100.)**2)/ge},
		'RR': {'ge': lambda mzp: np.sqrt(np.pi/2.)*mzp/8600.	, 'gm': lambda mzp, ge: 9. if ge==0 else (np.pi*(mzp/11600.)**2)/ge},
		'RL': {'ge': lambda mzp: np.sqrt(np.pi/2.)*mzp/8600.	, 'gm': lambda mzp, ge: 9. if ge==0 else (np.pi*(mzp/9100.)**2)/ge},}
LEPlimits2 = {	'VV': lambda mzp, ge: 9. if ge==0 else ( (4*np.pi*(mzp/18900.)**2)/ge if ge<np.sqrt(2*np.pi)*mzp/20600. else -1.) ,
		'LL': lambda mzp, ge: 9. if ge==0 else ( (np.pi*(mzp/12200.)**2)/ge if ge<np.sqrt(np.pi/2.)*mzp/8700. else -1.) ,
		'LR': lambda mzp, ge: 9. if ge==0 else ( (np.pi*(mzp/9100.)**2)/ge if ge<np.sqrt(np.pi/2.)*mzp/8700. else -1.) ,
		'RR': lambda mzp, ge: 9. if ge==0 else ( (np.pi*(mzp/11600.)**2)/ge if ge<np.sqrt(np.pi/2.)*mzp/8600. else -1.) ,
		'RL': lambda mzp, ge: 9. if ge==0 else ( (np.pi*(mzp/9100.)**2)/ge if ge<np.sqrt(np.pi/2.)*mzp/8600. else -1.) ,}

#
# Create a grid
#
masslist= [500,750,1000,1250]
gelist = np.linspace( 0, 1., args.points + 1) #smallest 0, largest 1, separation 0.1
gmlist = np.linspace( 0, 1., args.points + 1)
# generic
gp = 0.1 # for generic
gelist = np.linspace( 0, .01, args.points + 1) #smallest 0, largest 1, separation 0.1
gmlist = np.linspace( 0, .01, args.points + 1)
# generic OFF
if args.small:
	masslist= [500]
	#masslist= [500,750,1000,1250]
	gelist = np.linspace( 0, 1., 3) #smallest 0, largest 1, separation 0.05
	gmlist = np.linspace( 0, 1., 3)
	# generic
	gelist = np.linspace( 0, 0.01, 3) #smallest 0, largest 1, separation 0.05
	gmlist = np.linspace( 0, 0.01, 3)
	# generic OFF
GE, GM = np.meshgrid(gelist, gmlist)
np.savetxt( os.path.join( plot_directory, 'GE'), GE )
np.savetxt( os.path.join( plot_directory, 'GM'), GM )
nrpoints = np.product(np.shape(GE))
print 'Model points: ' + str( nrpoints )

widths = np.zeros(np.shape(GE))

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
	fig, axs = plt.subplots(2, 2, figsize=(6,6), constrained_layout=True )
	#fig, axs = plt.subplots(2, 2, figsize=(6,6))
	if args.int:
		fig.suptitle( model + ' model with interference\n' )
	else:
		fig.suptitle( model + ' model\n' )
	# generic
	fig.suptitle( 'generic model\n' )
	# generic OFF

	#fig.suptitle( '\n' )
	#fig.tight_layout( pad = 1.0)
	#fig.tight_layout( )
	#fig.subplots_adjust(wspace=0.2)

	for ax in axs.flat:
		ax.set(xlabel=r'$g_e$', ylabel=r'$g_{\mu}$')
	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
		ax.label_outer()
		#ax.set(adjustable='box-forced', aspect='equal')
		ax.set(aspect='equal')
		#ax.set_xticks( [0.,0.2,0.4,0.6,0.8,1.])
		#ax.set_yticks( [0.,0.2,0.4,0.6,0.8,1.])
		# generic
		#ax.set_xticks( [0.,0.002,0.004,0.006,0.008,0.01])
		#ax.set_yticks( [0.,0.002,0.004,0.006,0.008,0.01])
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
		ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,-2))
		ax.yaxis.major.formatter._useMathText = True
		ax.xaxis.major.formatter._useMathText = True
		# generic OFF

	plots = []
	plots.append( axs[0,0] )
	plots.append( axs[0,1] )
	plots.append( axs[1,0] )
	plots.append( axs[1,1] )
	
	for plot, mass in zip(plots,masslist):
		plot.set_title(r'$M_{Zp}=$' + str(mass) + ' GeV')
		# generic
		#plot.set_title(r'$M_{Zp}=$' + str(mass) + ' GeV, ' + r'$g_p=$' + str(gp) )
		plot.set_title(r'$M_{Zp}=$' + str(mass) + ' GeV')
		# generic OFF
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
				#Zp_model =  getZpmodel_sep(ge, gm, MZp, model = model,  WZp = args.width)
				# generic
				Zp_model =  getZpmodel_gen(ge, gm, MZp, gp= gp,  WZp = args.width) #gp=0.003 is a good value
				# generic OFF
				#Zp_model =  getZpmodel_semi(ge, gm, MZp, model = model,  WZp = args.width)
				Gamma_eff = np.sqrt( Zp_model['Gamma']**2 + ((ee_resolution(Zp_model['MZp']) + mm_resolution(Zp_model['MZp']))/2.)**2 ) 
				searchregion = [ Zp_model['MZp'] - args.window * Gamma_eff, Zp_model['MZp'] + args.window * Gamma_eff ]
				widths[gm_idx,ge_idx] = (Zp_model['Gamma']/Zp_model['MZp'])*100.
				for Z, E1, E2, E3, E4, E5, method in zip(Zs, E1s, E2s, E3s, E4s, E5s, methods.keys()):
					E1[gm_idx,ge_idx], E2[gm_idx,ge_idx], E3[gm_idx,ge_idx], E4[gm_idx,ge_idx], E5[gm_idx,ge_idx], Z[gm_idx,ge_idx]= methods[method]( Zp_model, sig_range = searchregion, withint = args.int)
		for Z, E1, E2, E3, E4, E5, method in zip(Zs, E1s, E2s, E3s, E4s, E5s, methods.keys()):
			np.savetxt( os.path.join( plot_directory, plotdirname +  '_Z_' + str(MZp) + method), Z )
			#np.savetxt( os.path.join( plot_directory, plotdirname + '_E1_' + str(MZp) + method), E1 )
			#np.savetxt( os.path.join( plot_directory, plotdirname + '_E2_' + str(MZp) + method), E2 )
			np.savetxt( os.path.join( plot_directory, plotdirname + '_E3_' + str(MZp) + method), E3 )
			#np.savetxt( os.path.join( plot_directory, plotdirname + '_E4_' + str(MZp) + method), E4 )
			#np.savetxt( os.path.join( plot_directory, plotdirname + '_E5_' + str(MZp) + method), E5 )


		#
		# plot LEP limits
		#
		if args.LEP:
			lep_ge = np.linspace( 0, 1., 501) #smallest 0, largest 1
			# LEP Limits ver 1
			## ge - vertical line
			#plot.axvline( LEPlimits[model]['ge'](MZp), color=lepcolor, linewidth=2, linestyle='solid')
			## gm - function
			#lep, = plot.plot( lep_ge, [LEPlimits[model]['gm'](MZp,lepge) for lepge in lep_ge], color= lepcolor, linewidth=2, linestyle='solid', )

			# LEP Limits ver 2
			lep, = plot.plot( lep_ge, [LEPlimits2[model](MZp,lepge) for lepge in lep_ge], color= lepcolor, linewidth=2, linestyle='solid', )

			plot.set_ylim(0,1)

		#
		# plot Z
		#
		# make contour plot
		contourplots = []
		# contour plot
		for Z, E1, E2, E3, E4, E5, method, color in zip(Zs, E1s, E2s, E3s, E4s, E5s, methods.keys(), colors):
			# exclusion line
			levels = [0, 0.05]
			contourplots.append(plot.contour(GE, GM, Z, levels,  colors=color, linewidths=2., linestyles='solid'  ))
			#contourplots.append(plot.contour(GE, GM, E1, levels, colors=color, linewidths=1., linestyles='dotted' ))
			#contourplots.append(plot.contour(GE, GM, E2, levels, colors=color, linewidths=1., linestyles='dashdot' ))
			contourplots.append(plot.contour(GE, GM, E3, levels, colors=color, linewidths=1., linestyles='dashed'))
			#contourplots.append(plot.contour(GE, GM, E4, levels, colors=color, linewidths=1., linestyles='dashdot' ))
			#contourplots.append(plot.contour(GE, GM, E5, levels, colors=color, linewidths=1., linestyles='dotted' ))
		#	# filles exclusion area
		#	contour_filled = plt.contourf(GE, GM, Z, levels, colors=['grey','cyan'])
		#plot.legend([h.legend_elements()[0][0] for h in contourplots],  methods.keys() )
		#plot.legend([contourplots[0].legend_elements()[0][0], contourplots[6].legend_elements()[0][0]],  methods.keys() )
	
		# widths contour
		#CS = plot.contourf(GE, GM, widths, 8, cmap=plt.cm.Greys, alpha=0.5)

		# generic
		#def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
		#	new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
		#			'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		#			 cmap(np.linspace(minval, maxval, n)))
		#	return new_cmap
		#new_cmap = truncate_colormap(plt.cm.Greys, 0.2, 1.0)
		#CS = plot.contourf(GE, GM, widths, 8, cmap=new_cmap, alpha=0.5)
		# generic OFF

		#
		# auto legend
		#
		legendelements = []
		nr = 0
		for method in methods.keys(): 
			legendelements.append(contourplots[nr].legend_elements()[0][0])
			nr += 3
		#plot.legend( legendelements,  methods.keys() )
		#
		# custom legend
		#
		#custom_lines = (Line2D([0], [0], color='blue', linewidth=2.),
		#                Line2D([0], [0], color='red', linewidth=2.),
		#		Line2D([0], [0], color='black' , linewidth=2.),)

		custom_lines = (Line2D([0], [0], color=colors[0], linewidth=2.),
		                Line2D([0], [0], color=colors[1], linewidth=2.),
				Line2D([0], [0], color=lepcolor , linewidth=2.),)
		labellist = tuple(methods.keys())
		labellist = labellist + ('LEP',)

		# generic
		custom_lines = (Line2D([0], [0], color=colors[0], linewidth=2.),
		                Line2D([0], [0], color=colors[1], linewidth=2.),)
		labellist = tuple(methods.keys())
		# generic OFF


	# legend
	#fig.legend( custom_lines, labellist, ncol=3 , loc='center', bbox_to_anchor=(0.5,0.92), frameon=False )
	# generic
	fig.legend( custom_lines, labellist, ncol=3 , loc='center', bbox_to_anchor=(0.5,0.93), frameon=False )
	# generic OFF

	# cbar for widht
	# Var A: bottom
	#cbar = fig.colorbar(CS, ax=axs[1, :3], location='bottom', shrink=0.9)
	# Var B: right
	# generic
	#cbar = fig.colorbar(CS, ax=axs[:, 1], location='right', shrink=0.96)
	#cbar.ax.set_ylabel(r'$\Gamma$ [%]')
	# generic OFF

	fig.savefig( os.path.join(plot_directory, file_name ))
	print 'Exclusion plot saved as ', os.path.join( plot_directory, file_name )
