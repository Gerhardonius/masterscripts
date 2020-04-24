import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

import ZPEED.dileptons_functions_modified as df_mod
from ZPEEDmod.Zpeedcounts import mydsigmadmll, getZpmodel, myDecayWidth
from directories.directories import plotdir

#
# argparser
# 
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='test')
args = argParser.parse_args()


#
# directories
#
plotdirname = 'Interference'
plot_directory = os.path.join( plotdir, plotdirname, args.plot_directory )
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory


def mydsigmadmll_mod(mll, Zp_model, species = "ee", quarktype = None, partonlevel = True):
	''' wrapper fro dsigmadmll_full_mod (returns differential cross section of the Z' only signal)
	added lepton flavor dependend coupling
	'''
  	if species == "ee":
  		sp = 0
  		glv = Zp_model['gev']
  		gla = Zp_model['gea']
  	elif species == "mm":
  		sp = 1
  		glv = Zp_model['gmv']
  		gla = Zp_model['gma']
  	else:
  	  	print("Unknown species requested")
  	  	return 0
  	return df_mod.dsigmadmll_full_mod(mll, Zp_model['MZp'], Zp_model['Gamma'], np.array([Zp_model['guv'],Zp_model['gdv']]), glv, np.array([Zp_model['gua'],Zp_model['gda']]), gla, sp, quarktype = quarktype , partonlevel = partonlevel )

def mydsigmadmll_wint_mod(mll, Zp_model, species = "ee", quarktype = None, partonlevel = True):
	''' wrapper fro dsigmadmll_wint_full_mod (returns differential cross section of the Z' only signal)
	added lepton flavor dependend coupling
	'''
  	if species == "ee":
  		sp = 0
  		glv = Zp_model['gev']
  		gla = Zp_model['gea']
  	elif species == "mm":
  		sp = 1
  		glv = Zp_model['gmv']
  		gla = Zp_model['gma']
  	else:
  	  	print("Unknown species requested")
  	  	return 0
  	return df_mod.dsigmadmll_wint_full_mod(mll, Zp_model['MZp'], Zp_model['Gamma'], np.array([Zp_model['guv'],Zp_model['gdv']]), glv, np.array([Zp_model['gua'],Zp_model['gda']]), gla, sp, quarktype = quarktype , partonlevel = partonlevel )


#
# Plots for different widths
#

g = 1.
models = ['VV05', 'RR05','LL05']
Masses = [1000. ,2000.]
Gammas = [0.005,0.03,0.05] # width fraction
Gammacolors = ['darkorange','dodgerblue','magenta']
quarktypes = ['u','d']

for quarktype in quarktypes:
	for model in models:
		for MZp in Masses:
			#sqrts_vals = np.linspace( 1000., 2000., num=100)
			sqrts_vals = np.linspace( MZp - 150. , MZp + 150., num=100)
		
			for WZp, color in zip(Gammas,Gammacolors):
			#
			# define Zp model
			#
				Zp_model = getZpmodel(g, MZp, model = model,  WZp = WZp )
				sigma = np.zeros_like( sqrts_vals )
				sigma_int = np.zeros_like( sqrts_vals )
				for i, sqrts in enumerate(sqrts_vals):
					sigma[i] = mydsigmadmll_mod( sqrts , Zp_model, species = "mm", quarktype = quarktype, partonlevel = True)
					sigma_int[i] = mydsigmadmll_wint_mod( sqrts , Zp_model, species = "mm", quarktype = quarktype, partonlevel = True)
		
				plt.plot( sqrts_vals,	sigma,		label=r'$ Z^{\prime}, \Gamma$ = ' + str(WZp*100) + ' %', color=color, linestyle = 'dashed')
				plt.plot( sqrts_vals,	sigma_int,	label=r'$ Z^{\prime} + int, \Gamma$ = ' + str(WZp*100) + ' %', color=color, linestyle = 'solid')
		
			plt.legend()
			plt.ylabel(r'$\sigma/fb$')
			plt.yscale( 'log' )
			plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
			plt.savefig( os.path.join( plot_directory, model + '_' + str(MZp) + '_' + quarktype + '_dimuon.pdf' ))
			plt.clf()
	
#
# Plots for different models and auto width
#
g = 1.
models = ['VV05', 'RR05','LL05']
modelcolors = ['darkorange','dodgerblue','magenta']
Masses = [1000. ,2000.]
quarktypes = ['u','d']

for quarktype in quarktypes:
	for MZp in Masses:
		sqrts_vals = np.linspace( MZp - 150. , MZp + 150., num=100)
		for model,color in zip(models,modelcolors):
		
			#
			# define Zp model
			#
			Zp_model = getZpmodel(g, MZp, model = model,  WZp = 'auto' )
			fraction = np.zeros_like( sqrts_vals )
			for i, sqrts in enumerate(sqrts_vals):
				fraction[i] = mydsigmadmll_wint_mod( sqrts , Zp_model, species = "mm", quarktype = quarktype, partonlevel = True) / mydsigmadmll_mod( sqrts , Zp_model, species = "mm", quarktype = quarktype, partonlevel = True)
		
			plt.plot( sqrts_vals,	fraction,		label=model , color=color, linestyle = 'solid')
		
		plt.legend()
		plt.ylabel(r'$ (Z^{\prime} + int) / Z^{\prime}$')
		#plt.yscale( 'log' )
		plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
		plt.savefig( os.path.join( plot_directory, 'Interferece_' + str(MZp) + '_' + quarktype + '_dimuon.pdf' ))
		plt.clf()
		
	
#
# Plots for interferences separately
#
g = 1.
terms = ['aint','Zint']
models = ['VV05', 'RR05','LL05']
termcolors = ['darkorange','dodgerblue']
Masses = [1000.]
quarktypes = ['u','d']

for quarktype in quarktypes:
	for model in models:
		for MZp in Masses:
			sqrts_vals = np.linspace( MZp - 150. , MZp + 150., num=100)
			for term, color in zip(terms,termcolors):
				#
				# define Zp model
				#
				Zp_model = getZpmodel(g, MZp, model = model,  WZp = 'auto' )
				sigma = np.zeros_like( sqrts_vals )
				for i, sqrts in enumerate(sqrts_vals):
					sigma[i] = mydsigmadmll_wint_mod( sqrts , Zp_model, species = "mm", quarktype = quarktype, partonlevel = term)
			
				plt.plot( sqrts_vals,	sigma,		label=term , color=color, linestyle = 'solid')
			
			for i, sqrts in enumerate(sqrts_vals):
				sigma[i] = mydsigmadmll_mod( sqrts , Zp_model, species = "mm", quarktype = quarktype, partonlevel = True)
			plt.plot( sqrts_vals,	sigma,		label=r'$Z^{\prime}$' , color='magenta', linestyle = 'solid')

			plt.legend()
			plt.ylabel(r'$\sigma/fb$')
			plt.axhline(y=0., color='black', linestyle='-')
			#plt.yscale( 'log' )
			plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
			plt.savefig( os.path.join( plot_directory, 'SeparatedTerms_' + str(MZp) + '_' + model + '_' + quarktype + '_dimuon.pdf' ))
			plt.clf()
		
	



