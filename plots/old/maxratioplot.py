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
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel_sep, getBSMcounts, myDecayWidth

from helpers import CLsZpeed #getHisto, customratiostyle, ratioStyle, getErrors, getratioErrors, canvasmod

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='Maxratioplot_test')
#argParser.add_argument('--plot_directory',  action='store', default='test', help="name of plot subdir")
args = argParser.parse_args()

#
# directories
#
plotdirname = 'Maxratioplots'
plot_directory = os.path.join( plotdir, plotdirname, args.plot_directory )
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory


#
# define stat methods
#

models = [      'VV', 
	#	'LL',
	#	'LR',
	#	'RR',
	#	'RL'
		]
withinterference = False
#
# Create a grid
#
masslist= np.linspace( 300., 3000., 10)

for model in models:
	print 'Working on: ', model
	file_name = model + '.pdf'
	for ge, gm in zip([1.,0.,1.,0.],[0.,1.,1.,0.]):
		MaxR=[]
		uncertR = []
		#
		# calculate and plot for every mass point
		#
		for MZp in masslist:
			Zp_model =  getZpmodel_sep( ge, gm, MZp, model = model,  WZp = 'auto')
			# reduce to range, to speed up calculation - only here the
			width = Zp_model['Gamma']
			mllrange=[ MZp - 3.*width, MZp + 3.*width]
			# SM counts
			ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
			ee_expected_bin = [ x[2] for x in ee_expected ]
			mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
			mm_observed_bin = [ x[2] for x in mm_observed ]
			mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
			mm_expected_bin = [ x[2] for x in mm_expected ]
			# BSM counts
			ee_signal   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withinterference )
			ee_signal_bin = [ x[2] for x in ee_signal ]                                                                       
			mm_signal   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withinterference )
			mm_signal_bin = [ x[2] for x in mm_signal ]
			#
			# calculate ratio
			#
			R = []
			uncert = []
			for binnr in range(len(mm_signal_bin)):
				# Ratio: if data = BSM
				#R.append( (ee_expected_bin[binnr]/(ee_signal_bin[binnr]))/(mm_expected_bin[binnr]/(mm_signal_bin[binnr])  ))	
				#R.append( (ee_signal_bin[binnr]/(ee_expected_bin[binnr]))/(mm_signal_bin[binnr]/(mm_expected_bin[binnr])  ))	
				R.append( (mm_signal_bin[binnr]/(mm_expected_bin[binnr]))/(ee_signal_bin[binnr]/(ee_expected_bin[binnr])  ))	
				# error if observations are SM expectations
				# numerator: (Data/MC)_e - data->expected - poisson error, MC->prediction - no error
				num_error = 1.* np.sqrt(ee_expected_bin[binnr])/ee_expected_bin[binnr]
				den_error = 1.* np.sqrt(mm_expected_bin[binnr])/mm_expected_bin[binnr]
				uncert.append( 1. * np.sqrt( num_error**2 + den_error**2 )  )
			MaxR.append( max(R) )		
			uncertR.append( uncert[np.argmax(np.array(R), axis=0)] )
		if (ge==0 and gm==0):
			plt.errorbar( masslist, MaxR,yerr=uncertR, xerr=None, label=r'$g_e=$' + str(ge) + '/' + r'$g_{\mu}=$' + str(gm))
		else:
			plt.plot( masslist, MaxR, label=r'$g_e=$' + str(ge) + '/' + r'$g_{\mu}=$' + str(gm))
	plt.title( model )
	plt.legend()
	plt.grid(True)
	plt.ylim( top = 1.3)
	plt.ylim( bottom = 0.7)
	plt.xlabel('Mzp/GeV')
	plt.ylabel(r'$((BSM/SM)_{ee}/(BSM/SM)_{\mu\mu})_{max}$')
	plt.savefig( os.path.join( plot_directory, file_name) )
