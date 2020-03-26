import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

import ZPEED.dileptons_functions_modified as df_mod
from ZPEEDmod.Zpeedcounts import mydsigmadmll, getZpmodel, myDecayWidth
from directories.directories import plotdir

from madgraph_crossections import sigmafromfile, Zp_wint_interference
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


def getZpmodel_Mathematica( MZp, WZp = 'auto'):
	# couplings to leptons, notation according to Madgraph
	gV2x2 = 1.
	gV1x1 = 0.5
	gA1x1 = 0.5
	gA2x2 = 1.
	gV3x3 = 0.
	gA3x3 = 0.

	# parameters taken from dilepton_functions.py for consistency
	# Weak Mixing Angle
	sw2 = 0.23126# PDG (Olive et al 2014) value at Q^2 = m_Z^2
	# Fine Structure Constant:
	alpha_e = 1/128.0# PDG (Olive et al 2014) value at Q^2 = m_W^2

	Lambda = 10000.
	muscale = MZp
	inducedcouplingfactor = (alpha_e/((1-sw2)*np.pi))*(gV1x1 + gV2x2 + gV3x3)*np.log(Lambda/muscale)

	# Define Zp model parameter point
	#
	Zp_model = {
	  'MZp': MZp,  #Zp mass
	  'mDM': 0.,   #Dark matter mass
	
	  # Zp-DM coupling
	  'gxv': 0.,     #Zp-DM vector coupling
	  'gxa': 0.,     #Zp-DM axial coupling
	
	  # Zp lepton coupling
	  'gev': gV1x1 + (-1./2.)*inducedcouplingfactor,   #Zp-el vector coupling
	  'gmv': gV2x2 + (-1./2.)*inducedcouplingfactor,   #Zp-mu vector coupling
	  'gtv': gV3x3 + (-1./2.)*inducedcouplingfactor,   #Zp-ta vector coupling
	  'gea': gA1x1 + (-1./6.)*inducedcouplingfactor,   #Zp-el axial-vector coupling
	  'gma': gA2x2 + (-1./6.)*inducedcouplingfactor,   #Zp-mu axial-vector coupling
	  'gta': gA3x3 + (-1./6.)*inducedcouplingfactor,   #Zp-ta axial-vector coupling

	  # Zp-quark coupling
	  'guv': ( 5./18.)*inducedcouplingfactor,	#Zp-up-type-quark vector coupling
	  'gdv': (-1./18.)* inducedcouplingfactor,#Zp-down-type-quark vector coupling
	  'gua': ( 1./6.)*inducedcouplingfactor,   #Zp-up-type-quark axial coupling
	  'gda': (-1./6.)*inducedcouplingfactor,  #Zp-down-type-quark axial coupling
	}

	# The couplings to neutrinos follow from SM gauge invariance and the fact that right-handed neutrinos do not exist
	Zp_model['gnev'] =  (Zp_model['gev'] - Zp_model['gea'])
	Zp_model['gnea'] =  (Zp_model['gea'] - Zp_model['gev'])
	Zp_model['gnmv'] =  (Zp_model['gmv'] - Zp_model['gma'])
	Zp_model['gnma'] =  (Zp_model['gma'] - Zp_model['gmv'])
	Zp_model['gntv'] =  (Zp_model['gtv'] - Zp_model['gta'])
	Zp_model['gnta'] =  (Zp_model['gta'] - Zp_model['gtv'])

	if WZp == 'auto':
		Zp_model['Gamma'] = myDecayWidth(Zp_model)
	else:
		Zp_model['Gamma'] = WZp * Zp_model['MZp']
	
	return Zp_model



def pdg( mll, Zp_model ):
	''' returns cross section in fb
	'''
	m = Zp_model['MZp']
	w = Zp_model['Gamma']
	Vi = Zp_model['guv']
	Ai = Zp_model['gua']
	Li = Vi - Ai 
	Ri = Vi + Ai 

	Vf = Zp_model['gmv']
	Af = Zp_model['gma']
	Lf = Vf - Af 
	Rf = Vf + Af 

	return (1/3.) *1/(64.*4*np.pi**2) * (1/mll**2) * (mll**4/((mll**2-m**2)**2 + (m*w)**2)) * (Li**2 + Ri**2)*(Lf**2 + Rf**2)*(16*np.pi/3)*0.3894*10**12

#
# define Zp model
#

# compare to Mathematica
Zp_model = getZpmodel_Mathematica( 2000 ,  WZp = 'auto')
print 'Zp_model_Mathematica'
print Zp_model

sqrts_vals = np.linspace( 1500., 2500., num=100)
sigma = np.zeros_like( sqrts_vals )
sigma_pdg = np.zeros_like( sqrts_vals )

for i, sqrts in enumerate(sqrts_vals):
	sigma[i] = mydsigmadmll_mod( sqrts , Zp_model, species = "mm", quarktype = 'u', partonlevel = True)
	sigma_pdg[i] = pdg( sqrts , Zp_model)
madgraph_mll, madgraph_sigma = sigmafromfile( 'cross_section_uux_zp_mm_Mathematica.txt')

# Plot
plt.plot( madgraph_mll ,madgraph_sigma, label='Madgraph')
plt.plot( sqrts_vals, 	sigma, 		label='ZPEED')
#plt.plot( sqrts_vals, 	sigma_pdg,	label='PDG')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'Mathematica_uux_zp_mm.pdf' ))
plt.clf()

# different model
g = 1.
MZp = 1500.
model = 'VV05'
Zp_model = getZpmodel(g, MZp, model = model,  WZp = 'auto')
print Zp_model

sqrts_vals = np.linspace( 1000., 2000., num=100)
sigma_u_mm = np.zeros_like( sqrts_vals )
sigma_d_mm = np.zeros_like( sqrts_vals )
sigma_u_ee = np.zeros_like( sqrts_vals )
sigma_d_ee = np.zeros_like( sqrts_vals )


for i, sqrts in enumerate(sqrts_vals):
	sigma_u_mm[i] = mydsigmadmll_mod( sqrts , Zp_model, species = "mm", quarktype = 'u', partonlevel = True)
	sigma_d_mm[i] = mydsigmadmll_mod( sqrts , Zp_model, species = "mm", quarktype = 'd', partonlevel = True)
	sigma_u_ee[i] = mydsigmadmll_mod( sqrts , Zp_model, species = "ee", quarktype = 'u', partonlevel = True)
	sigma_d_ee[i] = mydsigmadmll_mod( sqrts , Zp_model, species = "ee", quarktype = 'd', partonlevel = True)
madgraph_mll, madgraph_sigma_u_mm = sigmafromfile( 'cross_section_uux_zp_mm.txt')
madgraph_mll, madgraph_sigma_d_mm = sigmafromfile( 'cross_section_ddx_zp_mm.txt')
madgraph_mll, madgraph_sigma_u_ee = sigmafromfile( 'cross_section_uux_zp_ee.txt')
madgraph_mll, madgraph_sigma_d_ee = sigmafromfile( 'cross_section_ddx_zp_ee.txt')

#
# Plot
#
plt.plot( madgraph_mll ,madgraph_sigma_u_mm, label='Madgraph_u_mm')
plt.plot( sqrts_vals, 	sigma_u_mm,		label='ZPEED_u_mm')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_up_dimuon.pdf' ))
plt.clf()

plt.plot( madgraph_mll ,madgraph_sigma_d_mm, label='Madgraph_d_mm')
plt.plot( sqrts_vals, 	sigma_d_mm,		label='ZPEED_d_mm')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_do_dimuon.pdf' ))
plt.clf()

plt.plot( madgraph_mll ,madgraph_sigma_u_ee, label='Madgraph_u_ee')
plt.plot( sqrts_vals, 	sigma_u_ee,		label='ZPEED_u_ee')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_up_dielec.pdf' ))
plt.clf()

plt.plot( madgraph_mll ,madgraph_sigma_d_ee, label='Madgraph_d_ee')
plt.plot( sqrts_vals, 	sigma_d_ee,		label='ZPEED_d_ee')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_do_dielec.pdf' ))
plt.clf()


# Interference
sqrts_vals = np.linspace( 1000., 2000., num=100)
sigma_int_u_mm = np.zeros_like( sqrts_vals )
sigma_int_d_mm = np.zeros_like( sqrts_vals )
sigma_int_u_ee = np.zeros_like( sqrts_vals )
sigma_int_d_ee = np.zeros_like( sqrts_vals )

for i, sqrts in enumerate(sqrts_vals):
	sigma_int_u_mm[i] = mydsigmadmll_wint_mod( sqrts , Zp_model, species = "mm", quarktype = 'u', partonlevel = True)
	sigma_int_d_mm[i] = mydsigmadmll_wint_mod( sqrts , Zp_model, species = "mm", quarktype = 'd', partonlevel = True)
	sigma_int_u_ee[i] = mydsigmadmll_wint_mod( sqrts , Zp_model, species = "ee", quarktype = 'u', partonlevel = True)
	sigma_int_d_ee[i] = mydsigmadmll_wint_mod( sqrts , Zp_model, species = "ee", quarktype = 'd', partonlevel = True)
madgraph_mll, madgraph_sigma_int_u_mm = Zp_wint_interference( initial='uux', final='mm' )
madgraph_mll, madgraph_sigma_int_d_mm = Zp_wint_interference( initial='ddx', final='mm' )
madgraph_mll, madgraph_sigma_int_u_ee = Zp_wint_interference( initial='uux', final='ee' )
madgraph_mll, madgraph_sigma_int_d_ee = Zp_wint_interference( initial='ddx', final='ee' )

plt.plot( madgraph_mll ,madgraph_sigma_int_u_mm, label='Madgraph_int_u_mm')
plt.plot( sqrts_vals, 	sigma_int_u_mm,		label='ZPEED_int_u_mm')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_up_dimuon_wint.pdf' ))
plt.clf()

plt.plot( madgraph_mll ,madgraph_sigma_int_d_mm, label='Madgraph_int_d_mm')
plt.plot( sqrts_vals, 	sigma_int_d_mm,		label='ZPEED_int_d_mm')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_do_dimuon_wint.pdf' ))
plt.clf()

plt.plot( madgraph_mll ,madgraph_sigma_int_u_ee, label='Madgraph_int_u_ee')
plt.plot( sqrts_vals, 	sigma_int_u_ee,		label='ZPEED_int_u_ee')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_up_dielec_wint.pdf' ))
plt.clf()

plt.plot( madgraph_mll ,madgraph_sigma_int_d_ee, label='Madgraph_int_d_ee')
plt.plot( sqrts_vals, 	sigma_int_d_ee,		label='ZPEED_int_d_ee')

#madgraph_mll, madgraph_sigma_d_z_ee = sigmafromfile( 'cross_section_ddx_z_ee.txt')
#madgraph_mll, madgraph_sigma_d_zp_ee = sigmafromfile( 'cross_section_ddx_zp_ee.txt')
#madgraph_mll, madgraph_sigma_d_a_ee = sigmafromfile( 'cross_section_ddx_a_ee.txt')
#madgraph_mll, madgraph_sigma_d_tot_ee = sigmafromfile( 'cross_section_ddx_tot_ee.txt')
#plt.plot( madgraph_mll ,madgraph_sigma_d_z_ee, label='Madgraph_d_z_ee')
#plt.plot( madgraph_mll ,madgraph_sigma_d_zp_ee, label='Madgraph_d_zp_ee')
#plt.plot( madgraph_mll ,madgraph_sigma_d_a_ee, label='Madgraph_d_a_ee')
#plt.plot( madgraph_mll ,madgraph_sigma_d_tot_ee, label='Madgraph_d_tot_ee')
plt.legend()
plt.ylabel(r'$\sigma/fb$')
plt.xlabel(r'$\sqrt{\hat{s}}/GeV$')
plt.savefig( os.path.join( plot_directory, 'VV05_1500_do_dielec_wint.pdf' ))
plt.clf()

