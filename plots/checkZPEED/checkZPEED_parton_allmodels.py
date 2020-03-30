import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

import ZPEED.dileptons_functions_modified as df_mod
from ZPEEDmod.Zpeedcounts import mydsigmadmll, getZpmodel, myDecayWidth
from directories.directories import plotdir

from madgraph_crossections import Zp_nointerference, Zp_withinterference
#sigmafromfile, Zp_wint_interference, Zp_wint_interference2
#
# argparser
# 
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--plot_directory',     action='store',      default='test')
args = argParser.parse_args()


#
# directories
#
plotdirname = 'CheckZPEED_parton'
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
# Make plots
#

g = 1.
MZp = 1500.
models = ['VV05','LL05','RR05']

for model in models:
	# make subdir
	plot_directory = os.path.join( plotdir, plotdirname, args.plot_directory, model)
	if not os.path.exists(plot_directory):
    		os.makedirs(plot_directory) 

	#obtain Zp_model
	Zp_model = getZpmodel(g, MZp, model = model,  WZp = 'auto')

	# create empty arrays
	sqrts_vals = np.linspace( 1000., 2000., num=100)
	sigma_u_mm = np.zeros_like( sqrts_vals )
	sigma_d_mm = np.zeros_like( sqrts_vals )
	sigma_u_ee = np.zeros_like( sqrts_vals )
	sigma_d_ee = np.zeros_like( sqrts_vals )

	Zpeedmode = { 	'noInter' 	: (mydsigmadmll_mod, Zp_nointerference),
			'wiInter'	: (mydsigmadmll_wint_mod, Zp_withinterference), }

	for mode, (Zpeedfunction, Madfunction) in Zpeedmode.items():

		# plot Zp without interference
		
		for i, sqrts in enumerate(sqrts_vals):
			sigma_u_mm[i] = Zpeedfunction( sqrts , Zp_model, species = "mm", quarktype = 'u', partonlevel = True)
			sigma_d_mm[i] = Zpeedfunction( sqrts , Zp_model, species = "mm", quarktype = 'd', partonlevel = True)
			sigma_u_ee[i] = Zpeedfunction( sqrts , Zp_model, species = "ee", quarktype = 'u', partonlevel = True)
			sigma_d_ee[i] = Zpeedfunction( sqrts , Zp_model, species = "ee", quarktype = 'd', partonlevel = True)

		madgraph_mll, madgraph_sigma_u_mm = Madfunction( model= model ,initial='uux', final='mm' )
		madgraph_mll, madgraph_sigma_d_mm = Madfunction( model= model ,initial='ddx', final='mm' )
		madgraph_mll, madgraph_sigma_u_ee = Madfunction( model= model ,initial='uux', final='ee' )
		madgraph_mll, madgraph_sigma_d_ee = Madfunction( model= model ,initial='ddx', final='ee' )

		#
		# Plot
		#
		plt.plot( madgraph_mll ,madgraph_sigma_u_mm, label='Madgraph_u_mm')
		plt.plot( sqrts_vals, 	sigma_u_mm,		label='ZPEED_u_mm')
		plt.legend()
		plt.ylabel(r'$\sigma/fb$')
		plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
		plt.savefig( os.path.join( plot_directory, model + '_' + mode + '_1500_up_dimuon.pdf' ))
		plt.clf()
		
		plt.plot( madgraph_mll ,madgraph_sigma_d_mm, label='Madgraph_d_mm')
		plt.plot( sqrts_vals, 	sigma_d_mm,		label='ZPEED_d_mm')
		plt.legend()
		plt.ylabel(r'$\sigma/fb$')
		plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
		plt.savefig( os.path.join( plot_directory, model + '_' + mode + '_1500_do_dimuon.pdf' ))
		plt.clf()
		
		plt.plot( madgraph_mll ,madgraph_sigma_u_ee, label='Madgraph_u_ee')
		plt.plot( sqrts_vals, 	sigma_u_ee,		label='ZPEED_u_ee')
		plt.legend()
		plt.ylabel(r'$\sigma/fb$')
		plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
		plt.savefig( os.path.join( plot_directory, model + '_' + mode + '_1500_up_dielec.pdf' ))
		plt.clf()
		
		plt.plot( madgraph_mll ,madgraph_sigma_d_ee, label='Madgraph_d_ee')
		plt.plot( sqrts_vals, 	sigma_d_ee,		label='ZPEED_d_ee')
		plt.legend()
		plt.ylabel(r'$\sigma/fb$')
		plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
		plt.savefig( os.path.join( plot_directory, model + '_' + mode + '_1500_do_dielec.pdf' ))
		plt.clf()


