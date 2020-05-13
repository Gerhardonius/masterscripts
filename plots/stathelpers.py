# helper functions for Zpeed plotting
import ROOT
from array import array
import math
import matplotlib.pyplot as plt
import os
import sys

#
# Statistics
#
from ZPEED.chi2_CLs import get_likelihood
from ZPEED.ATLAS_13TeV import calculate_chi2
from ZPEED.ATLAS_13TeV_calibration import xi_function
from ZPEEDmod.Zpeedcounts import mydsigmadmll, mydsigmadmll_wint, myDecayWidth #allows for lepton flavor non universality
from ZPEEDmod.Zpeedcounts import getSMcounts, getZpmodel_sep, getBSMcounts, myDecayWidth
from ZPEEDmod.ATLAS_13TeV import calculate_chi2_noweight
from helpers import getHisto
from directories.directories import plotdir 

import numpy as np
import scipy.optimize as optimize
from scipy.integrate import quad
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import chi2

def sqrt_safe(x):
  return np.sqrt(max(x,0))

#
# CLs Zpeed variants
#
def CLsZpeed( Zp_model,  searchwindow=[-3.,3.], withint = True ):
	# search region
	Mlow = Zp_model['MZp'] + searchwindow[0] *Zp_model['Gamma']
	Mhigh = Zp_model['MZp']+ searchwindow[1] *Zp_model['Gamma']
	sig_range = [Mlow,Mhigh]

	if withint:
		#Step 2: Calculate differential cross section (including detector efficiency)
		# lambda functions (needed to calculate \hat{mu}
		ee_signal_with_interference = lambda x : xi_function(x, "ee") * mydsigmadmll_wint(x, Zp_model, "ee")
		mm_signal_with_interference = lambda x : xi_function(x, "mm") * mydsigmadmll_wint(x, Zp_model, "mm")	
		#Step 3: Create likelihood functions, returns chi2 test statistic as function of mu
		chi2_with_interference, chi2_Asimov_with_interference = calculate_chi2(ee_signal_with_interference, mm_signal_with_interference, signal_range=sig_range)
		# evaluates teststatistic and calculates CLs in asmptotic limit
		result_with_interference = get_likelihood(chi2_with_interference, chi2_Asimov_with_interference)
  		#return [chi2(1),Delta_chi2(1), CLs(1)]
		return result_with_interference[2]
		#return result_with_interference

	else:
		ee_signal = lambda x : xi_function(x, "ee") * mydsigmadmll(x, Zp_model, "ee")
		mm_signal = lambda x : xi_function(x, "mm") * mydsigmadmll(x, Zp_model, "mm")	

		chi2, chi2_Asimov = calculate_chi2(ee_signal, mm_signal, signal_range=sig_range)
		result = get_likelihood(chi2, chi2_Asimov)
  		return result[2]
  		#return result

def CLsZpeed_withexpected( Zp_model,  searchwindow=[-3.,3.], withint = True ):
	# search region
	Mlow = Zp_model['MZp'] + searchwindow[0] *Zp_model['Gamma']
	Mhigh = Zp_model['MZp']+ searchwindow[1] *Zp_model['Gamma']
	sig_range = [Mlow,Mhigh]

	if withint:
		#Step 2: Calculate differential cross section (including detector efficiency)
		# lambda functions (needed to calculate \hat{mu}
		ee_signal_with_interference = lambda x : xi_function(x, "ee") * mydsigmadmll_wint(x, Zp_model, "ee")
		mm_signal_with_interference = lambda x : xi_function(x, "mm") * mydsigmadmll_wint(x, Zp_model, "mm")	
		#Step 3: Create likelihood functions, returns chi2 test statistic as function of mu
		chi2, chi2_Asimov = calculate_chi2_noweight(ee_signal_with_interference, mm_signal_with_interference, signal_range=sig_range)
		# evaluates teststatistic and calculates CLs in asmptotic limit
		result = get_likelihood(chi2, chi2_Asimov)
  		#return [chi2(1),Delta_chi2(1), CLs(1)]
		#CLs_obs = result_with_interference[2]
		tmpresult = result

	else:
		ee_signal = lambda x : xi_function(x, "ee") * mydsigmadmll(x, Zp_model, "ee")
		mm_signal = lambda x : xi_function(x, "mm") * mydsigmadmll(x, Zp_model, "mm")	

		chi2, chi2_Asimov = calculate_chi2_noweight(ee_signal, mm_signal, signal_range=sig_range)
		result = get_likelihood(chi2, chi2_Asimov)
  		#CLs_obs = result[2]
		tmpresult = result

	quantiles = [-2.,-1.,0.,1.,2.]
	sqrtq_exp = [ x + sqrt_safe( chi2_Asimov(1.) ) for x in quantiles ]
  	CLs = [0.] * 5
	for i in range(len(CLs)):
		CLs[i]=(1 - norm.cdf( sqrtq_exp[i]) )/norm.cdf(sqrt_safe(chi2_Asimov(1.)) - sqrtq_exp[i])
	CLs.append( result[2] )	
	xvals = [ x**2 for x in sqrtq_exp] + [ tmpresult[1]]

	return CLs

def CLsZpeed_sampling( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, plotdirectory=None, N=10**2 ):
	# use plotname to save plot in Teststatistik directory
	# if None grab variable plotdirname (defined in onemass exclusion plot) to save them in subdirectory there 
	if plotdirectory==None:
		plotdirectory = os.path.join(plotdir, 'Teststatistik')
	if not os.path.exists(  plotdirectory):
		os.makedirs(    plotdirectory)

	if plotname==None:
		filename = os.path.join( plotdirectory, Zp_model['name'] + '_CLsZpeed_' + str(N) + '.pdf' ) 
	else:
		filename = os.path.join( plotdirectory, Zp_model['name'] + '_CLsZpeed_' + plotname + '_' + str(N) + '.pdf' ) 
	print 'Start sampling for: ', filename

	#
	# get ZPEED result to compare to and get Asimov data for asymptotic formula
	#
	Mlow = Zp_model['MZp'] + searchwindow[0] *Zp_model['Gamma']
	Mhigh = Zp_model['MZp']+ searchwindow[1] *Zp_model['Gamma']
	sig_range = [Mlow,Mhigh]

	if withint:
		#Step 2: Calculate differential cross section (including detector efficiency)
		# lambda functions (needed to calculate \hat{mu}
		ee_signal_with_interference = lambda x : xi_function(x, "ee") * mydsigmadmll_wint(x, Zp_model, "ee")
		mm_signal_with_interference = lambda x : xi_function(x, "mm") * mydsigmadmll_wint(x, Zp_model, "mm")	
		#Step 3: Create likelihood functions, returns chi2 test statistic as function of mu
		chi2, chi2_Asimov = calculate_chi2_noweight(ee_signal_with_interference, mm_signal_with_interference, signal_range=sig_range)
		# evaluates teststatistic and calculates CLs in asmptotic limit
		result = get_likelihood(chi2_with_interference, chi2_Asimov_with_interference)
  		#return [chi2(1),Delta_chi2(1), CLs(1)]
		Zpeedresult = result[2]

	else:
		ee_signal = lambda x : xi_function(x, "ee") * mydsigmadmll(x, Zp_model, "ee")
		mm_signal = lambda x : xi_function(x, "mm") * mydsigmadmll(x, Zp_model, "mm")	

		chi2, chi2_Asimov = calculate_chi2_noweight(ee_signal, mm_signal, signal_range=sig_range)
		result = get_likelihood(chi2, chi2_Asimov)
  		Zpeedresult = result[2]

	#
	# get counts
	#
	width = Zp_model['Gamma']
	MZp = Zp_model['MZp']
	# define mll range
	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
	# SM counts
	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
	ee_observed_bin = [ x[2] for x in ee_observed ]
	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
	ee_expected_bin = [ x[2] for x in ee_expected ]
	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
	mm_observed_bin = [ x[2] for x in mm_observed ]
	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
	mm_expected_bin = [ x[2] for x in mm_expected ]
	# BSM counts: note: this is actually SM+signal
	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
	ee_bsm_bin = [ x[2] for x in ee_bsm ]
	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
	mm_bsm_bin = [ x[2] for x in mm_bsm ]

	print 'this are the counts'
	print 'ee observed', ee_observed_bin
	print 'ee expexted', ee_expected_bin
	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
	print 'ee signal', ee_signal_bin 

	print 'mm observed', mm_observed_bin
	print 'mm expexted', mm_expected_bin
	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
	print 'mm signal',   mm_signal_bin

	#
	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
	# test statistik q=-2ln( Likelihood(mu=1) / Likelihood(muhat)) = -2 ( ln L (mu=1) - ln L (mu=muhat)) 
	#
	# some helpers
	def getloglikelihood( r, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin):
		loglikelihood_ee = 0.	
		loglikelihood_mm = 0.	
		for i in range(len(ee_observed_bin)):
			loglikelihood_ee += poisson.logpmf( ee_observed_bin[i] , mu= ee_expected_bin[i] + r * ee_signal_bin[i] )	
		for i in range(len(mm_observed_bin)):
			loglikelihood_mm += poisson.logpmf( mm_observed_bin[i] , mu= mm_expected_bin[i] + r * mm_signal_bin[i] )	
		return -2.*loglikelihood_ee, -2.*loglikelihood_mm

	# stolen from ZPEED
	def minimum(chi2):
		chi2_min = optimize.minimize(chi2, 1., method = 'Nelder-Mead', options = {'ftol':.01, 'maxiter':100} )
		if not(chi2_min['success']):
		  print('Warning: Failed to find minimal chi2')
		
		if chi2_min['x'][0] > 0:
		  mu_min = chi2_min['x'][0]
		else:
		  mu_min = 0

		return mu_min

	#
	# observed test statistic
	#
	muhat = minimum( lambda x: sum(getloglikelihood( x, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin)))
	qobs = sum(getloglikelihood( 1., ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin,	mm_signal_bin)) - sum(getloglikelihood( muhat, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin))

	#
	# sampling
	#
	def gettoydata( expectations_ee, expectations_mm ):
		toydata_ee = []
		for exp in expectations_ee:
			toydata_ee.append(poisson.rvs(exp))
		toydata_mm = []
		for exp in expectations_mm:
			toydata_mm.append(poisson.rvs(exp))
		return toydata_ee, toydata_mm

	def get_qlist( ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin, mu=0, N=100):
		q = []
		for i in range(N):
			if (np.log10(i+1) % 1) == 0:
				print i+1   
			toydata_ee, toydata_mm = gettoydata( [b + mu *s for b,s in zip(ee_expected_bin, ee_signal_bin) ] , [b + mu *s for b,s in zip(mm_expected_bin, mm_signal_bin) ] )

			muhat = minimum( lambda x: sum(getloglikelihood( x, toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin)))
			#UNG
			if muhat > 1.:
				q.append( 0.)
			else:
				# UNG: Here, always the tested hypotheses need s to be in the numerator!!! even for qb dist ( i want f(q_mu | mu=0), the first q_mu = hypothsis
				q.append( sum(getloglikelihood( 1. , toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin)) -
				sum(getloglikelihood( muhat , toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin)) )
		return q

	# sampled test statistic
	qsb = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=1., N=N)
	qb  = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=0., N=N)

	#
	# histo of sampled test statistik
	#
	# plot stuff
	plotdirname = 'Teststatistik'
	plot_directory = os.path.join( plotdir, plotdirname )
	if not os.path.exists(plot_directory):
    		os.makedirs(plot_directory) 
	# prepare plot
	numberofbins = 50 #eachside of qobs
	spacing = max( qobs - min(qsb), max(qsb)-qobs, qobs - min(qb), max(qb)-qobs, )/(numberofbins)
	binning = []
	for i in range(-numberofbins,numberofbins+1):
		binning.append( qobs + i*spacing)
	qsb_hist = plt.hist( qsb, bins=binning, density = True, label = 'S+B', log=True, alpha=0.5 ,color='r' )
	qb_hist = plt.hist( qb, bins=binning, density = True,  label = 'B', log=True   , alpha=0.5 ,color='b' )

	#
	# obtain CLs and q values: [-2sig, -sig, median, sig, 2sig, observed]
	#
	CLs_values, q_values = getCLs_values( qsb_hist, qb_hist, qobs, spacing, numberofbins)

	#
	# plot q_values, make pretty and save plot
	#
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
	for q_value, tag in zip(q_values[:-1], [ r'$q_{exp}^{2 \sigma}$', r'$q_{exp}^{ \sigma}$', r'$q_{exp}^{ med}$', r'$q_{exp}^{-\sigma}$', r'$q_{exp}^{-2 \sigma}$' ]):
		plt.axvline( q_value, color = 'black', linestyle='dashed', label= tag )
	plt.xlabel(r'$q_{obs}$')
	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\hat{\mu})}$')

	#
	# plot asymptotic
	#
	def asymptotic_b(x, asimov, testmu=1.):
		# asimov evaluatat at testmu
		sig = sqrt_safe( testmu**2/asimov )
		result = 0
		if x <= (testmu/sig)**2:
			result += (1/2.) * (1/sqrt_safe(2*np.pi)) * (1/sqrt_safe(x)) * np.exp( -(1/2.)*(sqrt_safe(x)-testmu/sig)**2)
		else:
			result += 1/(sqrt_safe(2*np.pi)*(2*testmu/sig)) * np.exp( -(1/2.)*(x-(testmu**2/sig**2))**2/(2*testmu/sig)**2 ) 
		if x==0:
			return result + norm.cdf( -testmu/sig )
		else:
			return result

	def asymptotic_sb(x, asimov, testmu=1.):
		# asimov evaluatat at testmu
		sig = sqrt_safe( testmu**2/asimov )
		result = 0
		if x <= (testmu/sig)**2:
			result += (1/2.) * (1/sqrt_safe(2*np.pi)) * (1/sqrt_safe(x)) * np.exp( -(x/2.) )
		else:
			result += 1/(sqrt_safe(2*np.pi)*(2*testmu/sig)) * np.exp( -(1/2.)*(x+(testmu**2/sig**2))**2/(2*testmu/sig)**2 ) 
		if x==0:
			return result +1/2.
		else:
			return result

	qlist = np.linspace( binning[0], binning[-1],1000)
	plt.plot( qlist, [ asymptotic_b(q, chi2_Asimov(1.) ) for q in qlist], label ='Asym. B', color='b')
	plt.plot( qlist, [ asymptotic_sb(q,chi2_Asimov(1.) ) for q in qlist], label ='Asym. S+B', color='r')
	plt.legend()

	# copied from with expected to get CLs from Asymptotic
	quantiles = [-2.,-1.,0.,1.,2.]
	sqrtq_exp = [ x + sqrt_safe( chi2_Asimov(1.) ) for x in quantiles ]
  	CLs = [0.] * 5
	for i in range(len(CLs)):
		CLs[i]=(1 - norm.cdf( sqrtq_exp[i]) )/norm.cdf(sqrt_safe(chi2_Asimov(1.)) - sqrtq_exp[i])
	print 'check'
	print CLs
	print Zpeedresult

	CLs_asym_obs_str = "%.3f" % Zpeedresult
	CLs_asym_exp_str = "%.3f" % CLs[2]
	CLs_obs_str = "%.3f" % CLs_values[-1]
	CLs_exp_str = "%.3f" % CLs_values[2]
	#plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
	#plt.title( 'CLs_obs=' + CLs_obs_str + ' / CLs_exp=' + CLs_exp_str + ' / N ' + str(N) )
	plt.title( 'Sampling: CLs_obs=' + CLs_obs_str + ' / CLs_exp=' + CLs_exp_str + ' / N ' + str(N) + '\n' + 'Asymptotic: CLs_obs=' + CLs_asym_obs_str + ' / CLs_exp=' + CLs_asym_exp_str )
	plt.savefig( filename )
	print 'Teststatistik saved as ', filename
	plt.clf()

	return CLs_values

def CLsZpeed_uncert( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, plotdirectory=None, N=10**2, Lunc = 0.03, Lnom=139. ):
	# use plotname to save plot in Teststatistik directory
	# if None grab variable plotdirname (defined in onemass exclusion plot) to save them in subdirectory there 
	if plotdirectory==None:
		plotdirectory = os.path.join(plotdir, 'Teststatistik')
	if not os.path.exists(  plotdirectory):
		os.makedirs(    plotdirectory)

	if plotname==None:
		filename = os.path.join( plotdirectory, Zp_model['name'] + '_CLsZpeed_LuncP' + str(Lunc*100) + '_' + str(N) + '.pdf' ) 
	else:
		filename = os.path.join( plotdirectory, Zp_model['name'] + '_CLsZpeed_LuncP' + str(Lunc*100) + '_' + plotname + '_' + str(N) + '.pdf' ) 
	print 'Start sampling for: ', filename

	#
	# get ZPEED result to compare to and get Asimov data for asymptotic formula
	#
	Mlow = Zp_model['MZp'] + searchwindow[0] *Zp_model['Gamma']
	Mhigh = Zp_model['MZp']+ searchwindow[1] *Zp_model['Gamma']
	sig_range = [Mlow,Mhigh]

	#
	# get counts
	#
	width = Zp_model['Gamma']
	MZp = Zp_model['MZp']
	# define mll range
	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
	# SM counts
	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =Lnom)
	ee_observed_bin = [ x[2] for x in ee_observed ]
	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =Lnom)
	ee_expected_bin = [ x[2] for x in ee_expected ]
	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =Lnom)
	mm_observed_bin = [ x[2] for x in mm_observed ]
	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =Lnom)
	mm_expected_bin = [ x[2] for x in mm_expected ]
	# BSM counts: note: this is actually SM+signal
	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =Lnom, mllrange =  mllrange, withinterference = withint )
	ee_bsm_bin = [ x[2] for x in ee_bsm ]
	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =Lnom, mllrange =  mllrange, withinterference = withint )
	mm_bsm_bin = [ x[2] for x in mm_bsm ]

	print 'this are the counts'
	print 'ee observed', ee_observed_bin
	print 'ee expexted', ee_expected_bin
	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
	print 'ee signal', ee_signal_bin 

	print 'mm observed', mm_observed_bin
	print 'mm expexted', mm_expected_bin
	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
	print 'mm signal',   mm_signal_bin

	#
	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
	# test statistik q=-2ln( Likelihood(mu=1) / Likelihood(muhat)) = -2 ( ln L (mu=1) - ln L (mu=muhat)) 
	#
	# some helpers
	def getloglikelihood( r, L, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=139., Lunc=Lunc):
		'''r...signalstrenght
		   L...Luminosity 
		   Lnom..Measurement 
		   Lunc..Measurement
		'''
		loglikelihood_ee = 0.	
		loglikelihood_mm = 0.	
		for i in range(len(ee_observed_bin)):
			loglikelihood_ee += poisson.logpmf( ee_observed_bin[i] , mu= ((L/Lnom)*ee_expected_bin[i] + r * (L/Lnom)*ee_signal_bin[i]) )	
		for i in range(len(mm_observed_bin)):
			loglikelihood_mm += poisson.logpmf( mm_observed_bin[i] , mu= ((L/Lnom)*mm_expected_bin[i] + r * (L/Lnom)*mm_signal_bin[i]) )	
		loglikelihood = loglikelihood_mm + loglikelihood_ee 
		# Nuisances: Measured values are the parameters of the pdf
		loglikelihood += norm.logpdf( L, loc=Lnom, scale=Lunc) 
		return -2.*loglikelihood

	def get_MLestimate(ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=139., Lunc=Lunc):
		''' geturns the mu and L that minimizes the likelihood
		'''
		chi2_min = optimize.minimize( lambda x: getloglikelihood( x[0], x[1], ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc), [1., 139.], method = 'Nelder-Mead', options = {'ftol':.01, 'maxiter':100} )
		if not(chi2_min['success']):
		  print('Warning: Failed to find minimal chi2')
		
		muhat = chi2_min['x'][0] 
		Lhat = chi2_min['x'][1] 

		if muhat < 0:
			muhat = 0

		return muhat, Lhat

	def get_condMLestimate(r, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=139., Lunc=Lunc):
		''' returns the L that miminmizes the likelihood for the fixed hypotheses r
		'''
		chi2_min = optimize.minimize( lambda x: getloglikelihood( r, x, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc), 139., method = 'Nelder-Mead', options = {'ftol':.01, 'maxiter':100} )
		if not(chi2_min['success']):
		  print('Warning: Failed to find minimal chi2')
		
		Lhathat = chi2_min['x'][0] 

		return Lhathat

	
	muhat_obs, Lhat_obs = get_MLestimate(ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc)
	print 'Max Likelihood for mu and L: ', muhat_obs, Lhat_obs
	Lhatr0_obs = get_condMLestimate(0., ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc)
	Lhatr1_obs = get_condMLestimate(1., ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc)
	print 'Lhat for background only: ', Lhatr0_obs
	print 'Lhat for signal + background:', Lhatr1_obs

	#
	# observed test statistic
	#
	qobs = getloglikelihood( 1., Lhatr1_obs, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc) - getloglikelihood( muhat_obs, Lhat_obs, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc)

	#
	# sampling
	#
	def gettoydata( expectations_ee, expectations_mm ):
		toydata_ee = []
		for exp in expectations_ee:
			toydata_ee.append(poisson.rvs(exp))
		toydata_mm = []
		for exp in expectations_mm:
			toydata_mm.append(poisson.rvs(exp))
		return toydata_ee, toydata_mm

	def get_qlist( ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin, mu=0, L=139., N=100):
		''' mu and L are used to generate data
		    but for evaluating the teststatistic, new values are fitted - 
		'''
		q = []
		for i in range(N):
			if (np.log10(i+1) % 1) == 0:
				print i+1   
			toydata_ee, toydata_mm = gettoydata( [(L/Lnom)*b + mu*(L/Lnom)*s for b,s in zip(ee_expected_bin, ee_signal_bin) ] , [(L/Lnom)*b + mu*(L/Lnom)*s for b,s in zip(mm_expected_bin, mm_signal_bin) ] )
			# get ML estimate and conditional exstimate
			muhat_toy, Lhat_toy = get_MLestimate( toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc)
			Lhathat_toy = get_condMLestimate(1., toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc)
			#UNG
			if muhat_toy > 1.:
				q.append( 0.)
			else:
				# UNG: Here, always the tested hypotheses need s to be in the numerator!!! even for qb dist ( i want f(q_mu | mu=0), the first q_mu = hypothsis
				q.append( getloglikelihood( 1., Lhathat_toy, toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc) - getloglikelihood( muhat_toy, Lhat_toy, toydata_ee , ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc) )
		return q

	# sampled test statistic
	qsb = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=1., L=Lhatr1_obs, N=N)
	qb  = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=0., L=Lhatr0_obs, N=N)

	#
	# histo of sampled test statistik
	#
	# plot stuff
	plotdirname = 'Teststatistik'
	plot_directory = os.path.join( plotdir, plotdirname )
	if not os.path.exists(plot_directory):
    		os.makedirs(plot_directory) 
	# prepare plot
	numberofbins = 50 #eachside of qobs
	spacing = max( qobs - min(qsb), max(qsb)-qobs, qobs - min(qb), max(qb)-qobs, )/(numberofbins)
	binning = []
	for i in range(-numberofbins,numberofbins+1):
		binning.append( qobs + i*spacing)
	qsb_hist = plt.hist( qsb, bins=binning, density = True, label = 'S+B', log=True, alpha=0.5 ,color='r' )
	qb_hist = plt.hist( qb, bins=binning, density = True,  label = 'B', log=True   , alpha=0.5 ,color='b' )

	#
	# obtain CLs and q values: [-2sig, -sig, median, sig, 2sig, observed]
	#
	CLs_values, q_values = getCLs_values( qsb_hist, qb_hist, qobs, spacing, numberofbins)

	#
	# plot q_values, make pretty and save plot
	#
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
	for q_value, tag in zip(q_values[:-1], [ 'r$q_{exp}^{2 \sigma}$', 'r$q_{exp}^{ \sigma}$', 'r$q_{exp}^{ med}$', r'$q_{exp}^{-\sigma}$', r'$q_{exp}^{-2 \sigma}$' ]):
		plt.axvline( q_value, color = 'black', linestyle='dashed', label= tag )
	plt.xlabel(r'$q_{obs}$')
	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1, \hat{\hat{L}})}{ \mathcal{L}(\hat{\mu}, \hat{L})}$')

	##
	## plot asymptotic
	##
	#def asymptotic_b(x, asimov, testmu=1.):
	#	# asimov evaluatat at testmu
	#	sig = sqrt_safe( testmu**2/asimov )
	#	result = 0
	#	if x <= (testmu/sig)**2:
	#		result += (1/2.) * (1/sqrt_safe(2*np.pi)) * (1/sqrt_safe(x)) * np.exp( -(1/2.)*(sqrt_safe(x)-testmu/sig)**2)
	#	else:
	#		result += 1/(sqrt_safe(2*np.pi)*(2*testmu/sig)) * np.exp( -(1/2.)*(x-(testmu**2/sig**2))**2/(2*testmu/sig)**2 ) 
	#	if x==0:
	#		return result + norm.cdf( -testmu/sig )
	#	else:
	#		return result

	#def asymptotic_sb(x, asimov, testmu=1.):
	#	# asimov evaluatat at testmu
	#	sig = sqrt_safe( testmu**2/asimov )
	#	result = 0
	#	if x <= (testmu/sig)**2:
	#		result += (1/2.) * (1/sqrt_safe(2*np.pi)) * (1/sqrt_safe(x)) * np.exp( -(x/2.) )
	#	else:
	#		result += 1/(sqrt_safe(2*np.pi)*(2*testmu/sig)) * np.exp( -(1/2.)*(x+(testmu**2/sig**2))**2/(2*testmu/sig)**2 ) 
	#	if x==0:
	#		return result +1/2.
	#	else:
	#		return result

	#qasimov_b = getloglikelihood( 1., Lhatr1_obs, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc) - getloglikelihood( muhat_obs, Lhat_obs, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, Lnom=Lnom, Lunc=Lunc)


	#qlist = np.linspace( binning[0], binning[-1],1000)
	#plt.plot( qlist, [ asymptotic_b(q, chi2_Asimov(1.) ) for q in qlist], label ='Asym. B', color='b')
	#plt.plot( qlist, [ asymptotic_sb(q,chi2_Asimov(1.) ) for q in qlist], label ='Asym. S+B', color='r')
	plt.legend()

	CLs_obs_str = "%.3f" % CLs_values[-1]
	CLs_exp_str = "%.3f" % CLs_values[2]
	#plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
	plt.title( 'CLs_obs=' + CLs_obs_str + ' / CLs_exp=' + CLs_exp_str + ' / N ' + str(N) )
	plt.savefig( filename )
	print 'Teststatistik saved as ', filename
	plt.clf()

	return CLs_values


#
# CLs Ratio variants
#
def CLsRatio_withexpected( Zp_model,  searchwindow=[-3.,3.], withint = True , variant = None):

	#
	# get counts
	#
	width = Zp_model['Gamma']
	MZp = Zp_model['MZp']
	# define mll range
	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
	# SM counts
	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
	ee_observed_bin = [ x[2] for x in ee_observed ]
	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
	ee_expected_bin = [ x[2] for x in ee_expected ]
	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
	mm_observed_bin = [ x[2] for x in mm_observed ]
	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
	mm_expected_bin = [ x[2] for x in mm_expected ]
	# BSM counts: note: this is actually SM+signal
	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
	ee_bsm_bin = [ x[2] for x in ee_bsm ]
	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
	mm_bsm_bin = [ x[2] for x in mm_bsm ]

	print 'this are the counts'
	print 'ee observed', ee_observed_bin
	print 'ee expexted', ee_expected_bin
	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
	print 'ee signal', ee_signal_bin 

	print 'mm observed', mm_observed_bin
	print 'mm expexted', mm_expected_bin
	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
	print 'mm signal',   mm_signal_bin

	# build ratios
	y_obs = []
	for i in range(len(ee_observed_bin)):
		if mm_observed_bin[i] != 0:
			y_obs.append( float(ee_observed_bin[i])/float(mm_observed_bin[i]) )
		else:
			y_obs.append( 1. )
			print 'WARNING mm_observed_bin[%i]=0'%i

	y_splusb = []
	for i in range(len(ee_observed_bin)):
		y_splusb.append( float(ee_expected_bin[i] + 1.*ee_signal_bin[i])/float(mm_expected_bin[i] + 1.*mm_signal_bin[i]) )
	uncert_splusb = []
	for i in range(len(ee_observed_bin)):
		#uncert_splusb.append(y_splusb[i] * np.sqrt( 1./(ee_expected_bin[i]+1.*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+1.*mm_signal_bin[i])) ) 
		# this way, the asyptotic works
		uncert_splusb.append(y_obs[i] * np.sqrt( 1./(ee_observed_bin[i]) + 1./(mm_observed_bin[i])) ) 

	y_b = []
	for i in range(len(ee_observed_bin)):
		y_b.append(  float(ee_expected_bin[i] + 0.*ee_signal_bin[i])/float(mm_expected_bin[i] + 0.*mm_signal_bin[i]) )

	uncert_b = []
	for i in range(len(ee_observed_bin)):
		#uncert_b.append(y_b[i] * np.sqrt( 1./(ee_expected_bin[i]+0.*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+0.*mm_signal_bin[i])) ) 
		# this way, the asyptotic works
		uncert_b.append(y_obs[i] * np.sqrt( 1./(ee_observed_bin[i]) + 1./(mm_observed_bin[i])) ) 

	if variant=='rev':
		for i in range(len(y_obs)):
			y_obs[i] = 1./y_obs[i]
		for i in range(len(y_splusb)):
			y_splusb[i] = 1./y_splusb[i]
		for i in range(len(y_obs)):
			y_b[i] = 1./y_b[i]

	if variant=='alb':
		for i in range(len(y_obs)):
			y_obs[i] = y_obs[i]/y_b[i]
		for i in range(len(y_splusb)):
			y_splusb[i] = y_splusb[i]/y_b[i]
		for i in range(len(y_obs)):
			y_b[i] = y_b[i]/y_b[i] #st, it is 1.

	print 'this are the ratios'
	print 'y_obs:', y_obs
	print 'y_splusb:', y_splusb
	print 'y_b:', y_b

	#
	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2  ln L (mu=1) + 2 ln L (mu=0)) = chi2(s+b) - chi2(b) 
	# Recall: -2 loglikelihood of gaus -> chi2, here i used gauss implementation, but there is no difference
	#
	# some helpers
	def getteststatistik( y_obs, y_splusb, uncert_splusb, y_b, uncert_b ):
		splusb = 0.
		b = 0.
		for i in range(len(y_obs)):
			splusb += norm.logpdf( y_obs[i] , loc = y_splusb[i], scale = uncert_splusb[i])	
			b += norm.logpdf( y_obs[i] , loc = y_b[i], scale =uncert_b[i])	
		return -2*(splusb - b)

	#
	# observed test statistic
	#
	qobs = getteststatistik( y_obs, y_splusb, uncert_splusb, y_b, uncert_b )
	# asimovs differ only by sign
	qasimov_b = getteststatistik(  y_b, 	y_splusb, uncert_splusb, y_b, uncert_b )
	qasimov_sb = getteststatistik( y_splusb,y_splusb, uncert_splusb, y_b, uncert_b )

	sig_sb_old = np.sqrt(1./ abs(qasimov_sb) ) 
	loc_sb = -1./sig_sb_old**2 
	scale_sb = 2./sig_sb_old 

	sig_b_old = np.sqrt(1./ abs(qasimov_b) ) 
	loc_b = 1./sig_b_old**2 
	scale_b = 2./sig_b_old 

	# expected
	quantiles = [-2.,-1.,0.,1.,2.]
	q_exp = [ x * scale_b + loc_b for x in quantiles ]
  	CLs = [0.] * 5
	for i in range(len(CLs)):
		CLs[i]=(1.- norm.cdf( (q_exp[i] - loc_sb)/ scale_sb  ))/( 1. - norm.cdf( (q_exp[i] - loc_b )/ scale_b))
	xvals = q_exp

	# observed
	CLs.append( ( 1. - norm.cdf( (qobs - loc_sb )/ scale_sb))/(1. - norm.cdf( (qobs - loc_b)/ scale_b  )))
	xvals.append( qobs )

	return CLs #xvals

def CLsRatio_sampling( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, plotdirectory = None, N=10**2, variant=None ):
	# use plotname to save plot in Teststatistik directory
	# if None grab variable plotdirname (defined in onemass exclusion plot) to save them in subdirectory there 
	if plotdirectory==None:
		plotdirectory = os.path.join(plotdir, 'Teststatistik')
	if not os.path.exists(  plotdirectory):
		os.makedirs(    plotdirectory)

	if plotname==None:
		filename = os.path.join( plotdirectory, Zp_model['name'] + '_CLsRatio_' + str(N) + '.pdf' ) 
	else:
		filename = os.path.join( plotdirectory, Zp_model['name'] + '_CLsRatio_' + plotname + '_' + str(N) + '.pdf' ) 
	print 'Start sampling for: ', filename

	#
	# get counts
	#
	width = Zp_model['Gamma']
	MZp = Zp_model['MZp']
	# define mll range
	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
	# SM counts
	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
	ee_observed_bin = [ x[2] for x in ee_observed ]
	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
	ee_expected_bin = [ x[2] for x in ee_expected ]
	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
	mm_observed_bin = [ x[2] for x in mm_observed ]
	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
	mm_expected_bin = [ x[2] for x in mm_expected ]
	# BSM counts: note: this is actually SM+signal
	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
	ee_bsm_bin = [ x[2] for x in ee_bsm ]
	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
	mm_bsm_bin = [ x[2] for x in mm_bsm ]

	print 'this are the counts'
	print 'ee observed', ee_observed_bin
	print 'ee expexted', ee_expected_bin
	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
	print 'ee signal', ee_signal_bin 

	print 'mm observed', mm_observed_bin
	print 'mm expexted', mm_expected_bin
	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
	print 'mm signal',   mm_signal_bin

	# build ratios
	y_obs = []
	for i in range(len(ee_observed_bin)):
		if mm_observed_bin[i] != 0:
			y_obs.append( float(ee_observed_bin[i])/float(mm_observed_bin[i]) )
		else:
			y_obs.append( 1. )
			print 'WARNING mm_observed_bin[%i]=0'%i

	y_splusb = []
	for i in range(len(ee_observed_bin)):
		y_splusb.append( float(ee_expected_bin[i] + 1.*ee_signal_bin[i])/float(mm_expected_bin[i] + 1.*mm_signal_bin[i]) )
	uncert_splusb = []
	for i in range(len(ee_observed_bin)):
		#uncert_splusb.append(y_splusb[i] * np.sqrt( 1./(ee_expected_bin[i]+1.*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+1.*mm_signal_bin[i])) ) 
		# this way, the asyptotic works
		uncert_splusb.append(y_obs[i] * np.sqrt( 1./(ee_observed_bin[i]) + 1./(mm_observed_bin[i])) ) 

	y_b = []
	for i in range(len(ee_observed_bin)):
		y_b.append(  float(ee_expected_bin[i] + 0.*ee_signal_bin[i])/float(mm_expected_bin[i] + 0.*mm_signal_bin[i]) )

	uncert_b = []
	for i in range(len(ee_observed_bin)):
		#uncert_b.append(y_b[i] * np.sqrt( 1./(ee_expected_bin[i]+0.*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+0.*mm_signal_bin[i])) ) 
		# this way, the asyptotic works
		uncert_b.append(y_obs[i] * np.sqrt( 1./(ee_observed_bin[i]) + 1./(mm_observed_bin[i])) ) 

	if variant=='rev':
		for i in range(len(y_obs)):
			y_obs[i] = 1./y_obs[i]
		for i in range(len(y_splusb)):
			y_splusb[i] = 1./y_splusb[i]
		for i in range(len(y_obs)):
			y_b[i] = 1./y_b[i]

	if variant=='alb':
		for i in range(len(y_obs)):
			y_obs[i] = y_obs[i]/y_b[i]
		for i in range(len(y_splusb)):
			y_splusb[i] = y_splusb[i]/y_b[i]
		for i in range(len(y_obs)):
			y_b[i] = y_b[i]/y_b[i] #st, it is 1.

	print 'this are the ratios'
	print 'y_obs:', y_obs
	print 'y_splusb:', y_splusb
	print 'y_b:', y_b

	#
	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2  ln L (mu=1) + 2 ln L (mu=0)) = chi2(s+b) - chi2(b) 
	# Recall: -2 loglikelihood of gaus -> chi2, here i used gauss implementation, but there is no difference
	#
	# some helpers
	def getteststatistik( y_obs, y_splusb, uncert_splusb, y_b, uncert_b ):
		splusb = 0.
		b = 0.
		for i in range(len(y_obs)):
			splusb += norm.logpdf( y_obs[i] , loc = y_splusb[i], scale = uncert_splusb[i])	
			b += norm.logpdf( y_obs[i] , loc = y_b[i], scale =uncert_b[i])	
		return -2*(splusb - b)

	#
	# observed test statistic
	#
	qobs = getteststatistik( y_obs, y_splusb, uncert_splusb, y_b, uncert_b )
	# asimovs differ only by sign
	qasimov_b = getteststatistik( y_b, y_splusb, uncert_splusb, y_b, uncert_b )
	qasimov_sb = getteststatistik( y_splusb, y_splusb, uncert_splusb, y_b, uncert_b )

	#
	# sampling
	#
	def gettoydata( y_exp, y_uncert ):
		toydata = []
		for i in range(len(y_exp)):
			toydata.append(norm.rvs( loc=y_exp[i], scale= y_uncert[i]))
		return toydata

	def get_qlist( y_obs, y_splusb, uncert_splusb, y_b, uncert_b, mu=1., N=100):
		q = []
		for i in range(N):
			if (np.log10(i+1) % 1) == 0:
				print i+1   
			if mu==1.:
				toydata = gettoydata( y_splusb, uncert_splusb )
			if mu==0.:
				toydata = gettoydata( y_b, uncert_b )

			q.append( getteststatistik( toydata, y_splusb, uncert_splusb, y_b, uncert_b) )
		return q

	# sampled test statistic
	qsb = get_qlist( y_obs, y_splusb, uncert_splusb, y_b, uncert_b, mu=1., N=N)
	qb  = get_qlist( y_obs, y_splusb, uncert_splusb, y_b, uncert_b, mu=0., N=N)

	#
	# histo of sampled test statistik
	#
	plotdirname = 'Teststatistik'
	plot_directory = os.path.join( plotdir, plotdirname )
	if not os.path.exists(plot_directory):
    		os.makedirs(plot_directory) 
	# prepare plot and plot
	numberofbins = 50 #eachside of qobs
	spacing = max( qobs - min(qsb), max(qsb)-qobs, qobs - min(qb), max(qb)-qobs, )/(numberofbins)
	binning = []
	for i in range(-numberofbins,numberofbins+1):
		binning.append( qobs + i*spacing)
	qsb_hist = plt.hist( qsb, bins=binning, density = True, label = 'S+B', log=True, alpha=0.5 ,color='r' )
	qb_hist = plt.hist( qb, bins=binning, density = True,  label = 'B', log=True   , alpha=0.5 ,color='b' )
	# scale histo
	plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 

	#
	# obtain CLs and q values: [-2sig, -sig, median, sig, 2sig, observed]
	#
	CLs_values, q_values = getCLs_values( qsb_hist, qb_hist, qobs, spacing, numberofbins)

	#
	# plot q_values, make pretty and save plot
	#
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
	for q_value, tag in zip(q_values[:-1], [ r'$q_{exp}^{2 \sigma}$', r'$q_{exp}^{ \sigma}$', r'$q_{exp}^{ med}$', r'$q_{exp}^{-\sigma}$', r'$q_{exp}^{-2 \sigma}$' ]):
		plt.axvline( q_value, color = 'black', linestyle='dashed', label= tag )
	plt.xlabel(r'$q_{obs}$')
	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}, \mathcal{L}=\mathcal{G}$')
	#plt.legend()

	##
	## plot asymptotic
	##
	def asymptotic_b(x, qasimov):
		# asimov evaluatat at testmu
		sig = np.sqrt(1./ abs(qasimov) ) 

		loc = 1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)

	def asymptotic_sb(x, qasimov):
		# asimov evaluatat at testmu
		sig = np.sqrt(1./ abs(qasimov) )

		loc = -1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)

	qlist = np.linspace( binning[0], binning[-1],1000)
	# here differnt
	plt.plot( qlist, [ asymptotic_b( q, qasimov_sb) for q in qlist], label ='Asym. B',   color='b')
	plt.plot( qlist, [ asymptotic_sb(q, qasimov_b ) for q in qlist], label ='Asym. S+B', color='r')
	plt.legend()

	# replaced by copy of withexpected, see below
	#psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(qasimov_sb ) )
	#oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(qasimov_b ) )
	#CLs_asym = psb_asym[0]/oneminuspb_asym[0]
	#CLs_asym_str = "%.3f" % CLs_asym
	#plt.title( 'CLs=' + CLs_obs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )

	# copied from with expected to get median significance
	sig_sb_old = np.sqrt(1./ abs(qasimov_sb) ) 
	loc_sb = -1./sig_sb_old**2 
	scale_sb = 2./sig_sb_old 

	sig_b_old = np.sqrt(1./ abs(qasimov_b) ) 
	loc_b = 1./sig_b_old**2 
	scale_b = 2./sig_b_old 

	# expected
	quantiles = [-2.,-1.,0.,1.,2.]
	q_exp = [ x * scale_b + loc_b for x in quantiles ]
  	CLs = [0.] * 5
	for i in range(len(CLs)):
		CLs[i]=(1.- norm.cdf( (q_exp[i] - loc_sb)/ scale_sb  ))/( 1. - norm.cdf( (q_exp[i] - loc_b )/ scale_b))
	CLs_asym_obs = (1.- norm.cdf( (qobs - loc_sb)/ scale_sb  ))/( 1. - norm.cdf( (qobs - loc_b )/ scale_b))

	CLs_asym_obs_str = "%.3f" % CLs_asym_obs
	CLs_asym_exp_str = "%.3f" % CLs[2]
	CLs_obs_str = "%.3f" % CLs_values[-1]
	CLs_exp_str = "%.3f" % CLs_values[2]
	plt.title( 'Sampling: CLs_obs=' + CLs_obs_str + ' / CLs_exp=' + CLs_exp_str + ' / N ' + str(N) + '\n' + 'Asymptotic: CLs_obs=' + CLs_asym_obs_str + ' / CLs_exp=' + CLs_asym_exp_str )
	plt.savefig( filename )
	print 'Teststatistik saved as ', filename
	plt.clf()

	return CLs_values#, q_values 

def getCLs_values( qsb_hist, qb_hist, qobs, spacing, numberofbins):
	# hist[0][i]...heights
	# hist[1][i]...bin edges, one entry more

	# define p_b - for expectations: -2 sig, -sig, median, sig, 2 sig
	areas = [ 0.97725, 0.84135, 0.5, 0.15865, 0.02275 ]
	
	# get value of teststatistik where x % of sampled teststatistics are in qb_hist, and the corresponding acutal area oneminusp_bs
	xvals = [ 0. ] * 5
	oneminusp_bs = [ 0. ] * 5
	for i in range(len(areas)):
		j=len(qb_hist[0])-1 # reduce len by one to get max index
		while oneminusp_bs[i] < areas[i]: # integrate until we oneminusp_bs is bigger than area, acutal x will be a little bigger than the obtained one because of binning
			oneminusp_bs[i] += qb_hist[0][j] * spacing #integrate histo
			xvals[i] = qb_hist[1][j] # set xval to lower edge of bin, which was just added to integral
			j -= 1 
			if j==-1:
				break
	# add qobs
	xvals.append(qobs)
	oneminusp_bs.append(  sum(qb_hist[0][numberofbins:])*spacing ) #binning is made such that there are 50bins blow qobs 
	# get p value of qsb_hist at this x
	p_sbs =  [ 0. ] * 6
	# for expectation
	for i in range(len(xvals)-1):
		j=len(qsb_hist[0])-1
		while qsb_hist[1][j] >= xvals[i]:
			p_sbs[i] += qsb_hist[0][j] * spacing
			j -= 1 
			if j==-1:
				break
	# for observed: use binning
	p_sbs[-1] = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
	# get CLs values
	CLs = [0.] * 6
	#print '----------------------'
	for i in range(len(CLs)):
		CLs[i] = p_sbs[i] / oneminusp_bs[i]
		#print p_sbs[i], oneminusp_bs[i]
	#print '----------------------'
	
	return CLs, xvals

#
# old implementations:
#	Tevatron teststatistic in mll search
#	Tevatron teststatistic in mll search with gauss Likelihood
#	Tevatron teststatistic in mll search with chi2 (actuatlly not a Likelihood)
#	Tevatron teststatistic in ratio search with gauss Likelihood but Poisson sampling (WRONG!!)
#

#def CLspython_Tevatron( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2 ):
#	#
#	# get ZPEED result to compare to and get Asimov data for asymptotic formula
#	#
#	Mlow = Zp_model['MZp'] + searchwindow[0] *Zp_model['Gamma']
#	Mhigh = Zp_model['MZp']+ searchwindow[1] *Zp_model['Gamma']
#	sig_range = [Mlow,Mhigh]
#
#	if withint:
#		#Step 2: Calculate differential cross section (including detector efficiency)
#		# lambda functions (needed to calculate \hat{mu}
#		ee_signal_with_interference = lambda x : xi_function(x, "ee") * mydsigmadmll_wint(x, Zp_model, "ee")
#		mm_signal_with_interference = lambda x : xi_function(x, "mm") * mydsigmadmll_wint(x, Zp_model, "mm")	
#		#Step 3: Create likelihood functions, returns chi2 test statistic as function of mu
#		chi2, chi2_Asimov = calculate_chi2_noweight(ee_signal_with_interference, mm_signal_with_interference, signal_range=sig_range)
#		# evaluates teststatistic and calculates CLs in asmptotic limit
#		result = get_likelihood(chi2_with_interference, chi2_Asimov_with_interference)
#  		#return [chi2(1),Delta_chi2(1), CLs(1)]
#		Zpeedresult = result[2]
#
#	else:
#		ee_signal = lambda x : xi_function(x, "ee") * mydsigmadmll(x, Zp_model, "ee")
#		mm_signal = lambda x : xi_function(x, "mm") * mydsigmadmll(x, Zp_model, "mm")	
#
#		chi2, chi2_Asimov = calculate_chi2_noweight(ee_signal, mm_signal, signal_range=sig_range)
#		result = get_likelihood(chi2, chi2_Asimov)
#  		Zpeedresult = result[2]
#
#	#
#	# get counts
#	#
#	width = Zp_model['Gamma']
#	MZp = Zp_model['MZp']
#	# define mll range
#	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
#	# SM counts
#	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
#	ee_observed_bin = [ x[2] for x in ee_observed ]
#	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
#	ee_expected_bin = [ x[2] for x in ee_expected ]
#	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
#	mm_observed_bin = [ x[2] for x in mm_observed ]
#	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
#	mm_expected_bin = [ x[2] for x in mm_expected ]
#	# BSM counts: note: this is actually SM+signal
#	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	ee_bsm_bin = [ x[2] for x in ee_bsm ]
#	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	mm_bsm_bin = [ x[2] for x in mm_bsm ]
#
#	print 'this are the counts'
#	print 'ee observed', ee_observed_bin
#	print 'ee expexted', ee_expected_bin
#	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
#	print 'ee signal', ee_signal_bin 
#
#	print 'mm observed', mm_observed_bin
#	print 'mm expexted', mm_expected_bin
#	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
#	print 'mm signal',   mm_signal_bin
#
#	#
#	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
#	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2 ( ln L (mu=1) - ln L (mu=0)) 
#	#
#	# some helpers
#	def getloglikelihood( r, ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin):
#		loglikelihood_ee = 0.	
#		loglikelihood_mm = 0.	
#		for i in range(len(ee_observed_bin)):
#			loglikelihood_ee += poisson.logpmf( ee_observed_bin[i] , mu= ee_expected_bin[i] + r * ee_signal_bin[i] )	
#		for i in range(len(mm_observed_bin)):
#			loglikelihood_mm += poisson.logpmf( mm_observed_bin[i] , mu= mm_expected_bin[i] + r * mm_signal_bin[i] )	
#		return -2.*loglikelihood_ee, -2.*loglikelihood_mm
#	#
#	# observed test statistic
#	#
#	qobs = sum(getloglikelihood( 1., ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin,	mm_signal_bin)) - sum(getloglikelihood( 0., ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin))
#
#	#
#	# sampling
#	#
#	def gettoydata( expectations_ee, expectations_mm ):
#		toydata_ee = []
#		for exp in expectations_ee:
#			toydata_ee.append(poisson.rvs(exp))
#		toydata_mm = []
#		for exp in expectations_mm:
#			toydata_mm.append(poisson.rvs(exp))
#		return toydata_ee, toydata_mm
#
#	def get_qlist( ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin, mu=0, N=100):
#		q = []
#		for i in range(N):
#			if (np.log10(i+1) % 1) == 0:
#				print i+1   
#			toydata_ee, toydata_mm = gettoydata( [b + mu *s for b,s in zip(ee_expected_bin, ee_signal_bin) ] , [b + mu *s for b,s in zip(mm_expected_bin, mm_signal_bin) ] )
#
#			q.append( sum(getloglikelihood( 1. , toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin)) -
#			sum(getloglikelihood( 0. , toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin)) )
#		return q
#
#	# sampled test statistic
#	qsb = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=1., N=N)
#	qb  = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=0., N=N)
#
#	#
#	# histo of sampled test statistik
#	#
#	# plot stuff
#	plotdirname = 'Teststatistik'
#	plot_directory = os.path.join( plotdir, plotdirname )
#	if not os.path.exists(plot_directory):
#    		os.makedirs(plot_directory) 
#	# prepare plot
#	numberofbins = 50 #eachside of qobs
#	spacing = max( qobs - min(qsb), max(qsb)-qobs, qobs - min(qb), max(qb)-qobs, )/(numberofbins)
#	binning = []
#	for i in range(-numberofbins,numberofbins+1):
#		binning.append( qobs + i*spacing)
#	qsb_hist = plt.hist( qsb, bins=binning, density = True, label = 'S+B', log=True, alpha=0.5 ,color='r' )
#	qb_hist = plt.hist( qb, bins=binning, density = True,  label = 'B', log=True   , alpha=0.5 ,color='b' )
#	# scale histo
#	#plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
#	# extract p values from plot
#	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
#	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
#	#print qsb_hist[0][numberofbins:] #10bins above qobs 
#	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
#	#print qb_hist[0][:numberofbins] #10bins above qobs 
#	#print 'Integral qb:', sum(qb_hist[0])*spacing
#	# finish plot
#	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
#	plt.xlabel(r'$q_{obs}$')
#	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}$')
#	#
#	# plot asymptotic
#	#
#	def asymptotic_b(x, asimov, testmu=1.):
#		# asimov evaluatat at testmu
#		sig = sqrt_safe( testmu**2/asimov )
#
#		loc = 1./sig**2 # mu of gauss
#		scale = 2./sig # sigma of gauss
#		return norm.pdf(x, loc=loc, scale=scale)
#
#	def asymptotic_sb(x, asimov, testmu=1.):
#		# asimov evaluatat at testmu
#		sig = sqrt_safe( testmu**2/asimov )
#
#		loc = -1./sig**2 # mu of gauss
#		scale = 2./sig # sigma of gauss
#		return norm.pdf(x, loc=loc, scale=scale)
#
#	qlist = np.linspace( binning[0], binning[-1],1000)
#	plt.plot( qlist, [ asymptotic_b(q, chi2_Asimov(1.) ) for q in qlist], label ='Asym. B', color='b')
#	plt.plot( qlist, [ asymptotic_sb(q,chi2_Asimov(1.) ) for q in qlist], label ='Asym. S+B', color='r')
#	plt.legend()
#	#
#	# Evaluate CLs vals
#	#
#	# asymptotic
#	psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(chi2_Asimov(1.)) )
#	oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(chi2_Asimov(1.)) )
#	CLs_asym = psb_asym[0]/oneminuspb_asym[0]
#	# python sampling
#	CLs = psb_val/oneminuspb_val
#	#print 'psb_val:' , psb_val
#	#print 'oneminuspb_val:' , oneminuspb_val
#	#print 'CLs: ', CLs
#	CLs_str = "%.3f" % CLs
#	# Zpeed
#	CLs_asym_str = "%.3f" % CLs_asym
#	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
#	if plotname != None:
#		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
#		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
#	plt.clf()
#
#	return {'CLspython:':CLs, 'CLspython_asym:':CLs_asym}
#
#def CLspython_Tevatron_gauss( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2 ):
#	#
#	# get counts
#	#
#	width = Zp_model['Gamma']
#	MZp = Zp_model['MZp']
#	# define mll range
#	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
#	# SM counts
#	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
#	ee_observed_bin = [ x[2] for x in ee_observed ]
#	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
#	ee_expected_bin = [ x[2] for x in ee_expected ]
#	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
#	mm_observed_bin = [ x[2] for x in mm_observed ]
#	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
#	mm_expected_bin = [ x[2] for x in mm_expected ]
#	# BSM counts: note: this is actually SM+signal
#	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	ee_bsm_bin = [ x[2] for x in ee_bsm ]
#	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	mm_bsm_bin = [ x[2] for x in mm_bsm ]
#
#	print 'this are the counts'
#	print 'ee observed', ee_observed_bin
#	print 'ee expexted', ee_expected_bin
#	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
#	print 'ee signal', ee_signal_bin 
#
#	print 'mm observed', mm_observed_bin
#	print 'mm expexted', mm_expected_bin
#	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
#	print 'mm signal',   mm_signal_bin
#
#	#
#	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
#	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2* (loglik (s+b) -loglik (b) ) 
#	#
#	# some helpers
#	def getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin):
#		splusb = 0.
#		b = 0.
#		for i in range(len(ee_observed_bin)):
#			y_data = ee_observed_bin[i]
#			y_pred_splusb = ee_expected_bin[i] + 1.*ee_signal_bin[i]
#			uncert_splusb = np.sqrt((ee_expected_bin[i]+1.*ee_signal_bin[i]))
#			splusb += norm.logpdf( y_data , loc = y_pred_splusb, scale =uncert_splusb)	
#
#			y_pred_b = ee_expected_bin[i] + 0*ee_signal_bin[i]
#			uncert_b = np.sqrt((ee_expected_bin[i]+0*ee_signal_bin[i]))
#			b += norm.logpdf( y_data , loc = y_pred_b, scale =uncert_b)	
#		for i in range(len(mm_observed_bin)):
#			y_data = mm_observed_bin[i]
#			y_pred_splusb = mm_expected_bin[i] + 1.*mm_signal_bin[i]
#			uncert_splusb = np.sqrt((mm_expected_bin[i]+1.*mm_signal_bin[i]))
#			splusb += norm.logpdf( y_data , loc = y_pred_splusb, scale =uncert_splusb)	
#
#			y_pred_b = mm_expected_bin[i] + 0*mm_signal_bin[i]
#			uncert_b = np.sqrt((mm_expected_bin[i]+0*mm_signal_bin[i]))
#			b += norm.logpdf( y_data , loc = y_pred_b, scale =uncert_b)	
#
#		return -2*(splusb - b)
#
#	#
#	# observed test statistic
#	#
#	qobs = 		getteststatistik(  ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin,	mm_signal_bin)
#	qasimov_b = 	getteststatistik(  ee_expected_bin, ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_expected_bin,	mm_signal_bin)
#	qasimov_sb = 	getteststatistik(  [ b+s for b,s in zip(ee_expected_bin,ee_signal_bin) ], ee_expected_bin, ee_signal_bin,  [ b+s for b,s in zip(mm_expected_bin,mm_signal_bin) ], mm_expected_bin,	mm_signal_bin)
#
#	#
#	# sampling
#	#
#	def gettoydata( expectations_ee, expectations_mm ):
#		toydata_ee = []
#		for exp in expectations_ee:
#			toydata_ee.append(poisson.rvs(exp))
#		toydata_mm = []
#		for exp in expectations_mm:
#			toydata_mm.append(poisson.rvs(exp))
#		return toydata_ee, toydata_mm
#
#	def get_qlist( ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin, mu=0, N=100):
#		q = []
#		for i in range(N):
#			if (np.log10(i+1) % 1) == 0:
#				print i+1   
#			toydata_ee, toydata_mm = gettoydata( [b + mu *s for b,s in zip(ee_expected_bin, ee_signal_bin) ] , [b + mu *s for b,s in zip(mm_expected_bin, mm_signal_bin) ] )
#
#			q.append( getteststatistik( toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin))
#		return q
#
#	# sampled test statistic
#	qsb = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=1., N=N)
#	qb  = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=0., N=N)
#
#	#
#	# histo of sampled test statistik
#	#
#	# plot stuff
#	plotdirname = 'Teststatistik'
#	plot_directory = os.path.join( plotdir, plotdirname )
#	if not os.path.exists(plot_directory):
#    		os.makedirs(plot_directory) 
#	# prepare plot
#	numberofbins = 50 #eachside of qobs
#	spacing = max( qobs - min(qsb), max(qsb)-qobs, qobs - min(qb), max(qb)-qobs, )/(numberofbins)
#	binning = []
#	for i in range(-numberofbins,numberofbins+1):
#		binning.append( qobs + i*spacing)
#	qsb_hist = plt.hist( qsb, bins=binning, density = True, label = 'S+B', log=True, alpha=0.5 ,color='r' )
#	qb_hist = plt.hist( qb, bins=binning, density = True,  label = 'B', log=True   , alpha=0.5 ,color='b' )
#	# scale histo
#	#plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
#	# extract p values from plot
#	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
#	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
#	#print qsb_hist[0][numberofbins:] #10bins above qobs 
#	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
#	#print qb_hist[0][:numberofbins] #10bins above qobs 
#	#print 'Integral qb:', sum(qb_hist[0])*spacing
#	# finish plot
#	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
#	plt.xlabel(r'$q_{obs}$')
#	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}, \mathcal{L} = \mathcal{G}$')
#
#	#
#	# plot asymptotic
#	#
#	def asymptotic_b(x, qasimov):
#		# asimov evaluatat at testmu
#		sig = np.sqrt( 1./ abs(qasimov) )
#
#		loc = 1./sig**2 # mu of gauss
#		scale = 2./sig # sigma of gauss
#		return norm.pdf(x, loc=loc, scale=scale)
#
#	def asymptotic_sb(x, qasimov):
#		# asimov evaluatat at testmu
#		sig = np.sqrt( 1./ abs(qasimov) )
#
#		loc = -1./sig**2 # mu of gauss
#		scale = 2./sig # sigma of gauss
#		return norm.pdf(x, loc=loc, scale=scale)
#
#	qlist = np.linspace( binning[0], binning[-1],1000)
#	#plt.plot( qlist, [ asymptotic_b(q, qasimov_b)  for q in qlist], label ='Asym. B', color='b')
#	plt.plot( qlist, [ asymptotic_b(q, qasimov_sb)  for q in qlist], label ='Asym. B', color='b') # these are the correct ones..determined by trying
#	#plt.plot( qlist, [ asymptotic_sb(q, qasimov_sb)  for q in qlist], label ='Asym. S+B', color='r')
#	plt.plot( qlist, [ asymptotic_sb(q, qasimov_b)  for q in qlist], label ='Asym. S+B', color='r') # these are the correct ones..determined by trying
#	plt.legend()
#	#
#	# Evaluate CLs vals
#	#
#	# asymptotic
#	psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(qasimov_b) )
#	oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(qasimov_sb) )
#	CLs_asym = psb_asym[0]/oneminuspb_asym[0]
#	# python sampling
#	CLs = psb_val/oneminuspb_val
#	#print 'psb_val:' , psb_val
#	#print 'oneminuspb_val:' , oneminuspb_val
#	#print 'CLs: ', CLs
#	CLs_str = "%.3f" % CLs
#	# Zpeed
#	CLs_asym_str = "%.3f" % CLs_asym
#	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
#	if plotname != None:
#		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
#		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
#	plt.clf()
#
#	return {'CLspython:':CLs, 'CLspython_asym:':CLs_asym}
#
#def CLspython_Tevatron_chi2( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2 ):
#	# Note: this is basecally the same as gauss, with the backdraw that the asymtotic does not work out
#
#	#
#	# get counts
#	#
#	width = Zp_model['Gamma']
#	MZp = Zp_model['MZp']
#	# define mll range
#	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
#	# SM counts
#	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
#	ee_observed_bin = [ x[2] for x in ee_observed ]
#	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
#	ee_expected_bin = [ x[2] for x in ee_expected ]
#	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
#	mm_observed_bin = [ x[2] for x in mm_observed ]
#	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
#	mm_expected_bin = [ x[2] for x in mm_expected ]
#	# BSM counts: note: this is actually SM+signal
#	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	ee_bsm_bin = [ x[2] for x in ee_bsm ]
#	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	mm_bsm_bin = [ x[2] for x in mm_bsm ]
#
#	print 'this are the counts'
#	print 'ee observed', ee_observed_bin
#	print 'ee expexted', ee_expected_bin
#	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
#	print 'ee signal', ee_signal_bin 
#
#	print 'mm observed', mm_observed_bin
#	print 'mm expexted', mm_expected_bin
#	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
#	print 'mm signal',   mm_signal_bin
#
#	#
#	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
#	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = Now use chi2 
#	#
#	# some helpers
#	def getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin):
#		splusb = 0.
#		b = 0.
#		for i in range(len(ee_observed_bin)):
#			y_data = ee_observed_bin[i]
#			y_pred_splusb = ee_expected_bin[i] + 1.*ee_signal_bin[i]
#			uncert_splusb = np.sqrt((ee_expected_bin[i]+1.*ee_signal_bin[i]))
#			splusb += (y_data - y_pred_splusb)**2/uncert_splusb**2	
#
#			y_pred_b = ee_expected_bin[i] + 0*ee_signal_bin[i]
#			uncert_b = np.sqrt((ee_expected_bin[i]+0*ee_signal_bin[i]))
#			b += (y_data - y_pred_b)**2/uncert_b**2	
#		for i in range(len(mm_observed_bin)):
#			y_data = mm_observed_bin[i]
#			y_pred_splusb = mm_expected_bin[i] + 1.*mm_signal_bin[i]
#			uncert_splusb = np.sqrt((mm_expected_bin[i]+1.*mm_signal_bin[i]))
#			splusb += (y_data - y_pred_splusb)**2/uncert_splusb**2	
#
#			y_pred_b = mm_expected_bin[i] + 0*mm_signal_bin[i]
#			uncert_b = np.sqrt((mm_expected_bin[i]+0*mm_signal_bin[i]))
#			b += (y_data - y_pred_b)**2/uncert_b**2	
#
#		return splusb - b
#
#	#
#	# observed test statistic
#	#
#	qobs = 		getteststatistik(  ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin,	mm_signal_bin)
#	# Asmivos: Note: qasimov_sb is negative! 
#	qasimov_b = 	getteststatistik(  ee_expected_bin, ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_expected_bin,	mm_signal_bin)
#	qasimov_sb = 	getteststatistik(  [ b+s for b,s in zip(ee_expected_bin,ee_signal_bin) ], ee_expected_bin, ee_signal_bin,  [ b+s for b,s in zip(mm_expected_bin,mm_signal_bin) ], mm_expected_bin,	mm_signal_bin)
#
#	#
#	# sampling
#	#
#	def gettoydata( expectations_ee, expectations_mm ):
#		toydata_ee = []
#		for exp in expectations_ee:
#			toydata_ee.append(poisson.rvs(exp))
#		toydata_mm = []
#		for exp in expectations_mm:
#			toydata_mm.append(poisson.rvs(exp))
#		return toydata_ee, toydata_mm
#
#	def get_qlist( ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin, mu=0, N=100):
#		q = []
#		for i in range(N):
#			if (np.log10(i+1) % 1) == 0:
#				print i+1   
#			toydata_ee, toydata_mm = gettoydata( [b + mu *s for b,s in zip(ee_expected_bin, ee_signal_bin) ] , [b + mu *s for b,s in zip(mm_expected_bin, mm_signal_bin) ] )
#
#			q.append( getteststatistik( toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin))
#		return q
#
#	# sampled test statistic
#	qsb = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=1., N=N)
#	qb  = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=0., N=N)
#
#	#
#	# histo of sampled test statistik
#	#
#	# plot stuff
#	plotdirname = 'Teststatistik'
#	plot_directory = os.path.join( plotdir, plotdirname )
#	if not os.path.exists(plot_directory):
#    		os.makedirs(plot_directory) 
#	# prepare plot
#	numberofbins = 50 #eachside of qobs
#	spacing = max( qobs - min(qsb), max(qsb)-qobs, qobs - min(qb), max(qb)-qobs, )/(numberofbins)
#	binning = []
#	for i in range(-numberofbins,numberofbins+1):
#		binning.append( qobs + i*spacing)
#	qsb_hist = plt.hist( qsb, bins=binning, density = True, label = 'S+B', log=True, alpha=0.5 ,color='r' )
#	qb_hist = plt.hist( qb, bins=binning, density = True,  label = 'B', log=True   , alpha=0.5 ,color='b' )
#	# scale histo
#	# plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
#	# extract p values from plot
#	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
#	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
#	#print qsb_hist[0][numberofbins:] #10bins above qobs 
#	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
#	#print qb_hist[0][:numberofbins] #10bins above qobs 
#	#print 'Integral qb:', sum(qb_hist[0])*spacing
#	# finish plot
#	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
#	plt.xlabel(r'$q_{obs}$')
#	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}, -2 \ln \mathcal{L} \approx \chi^2$')
#	#
#	# plot asymptotic
#	#
##
#	def asymptotic_b(x, qasimov):
#		# asimov evaluatat at testmu
#		sig = np.sqrt( 1./ abs(qasimov) )
#
#		loc = 1./sig**2 # mu of gauss
#		scale = 2./sig # sigma of gauss
#		return norm.pdf(x, loc=loc, scale=scale)
#
#	def asymptotic_sb(x, qasimov):
#		# asimov evaluatat at testmu
#		sig = np.sqrt( 1./ abs(qasimov) )
#
#		loc = -1./sig**2 # mu of gauss
#		scale = 2./sig # sigma of gauss
#		return norm.pdf(x, loc=loc, scale=scale)
#	qlist = np.linspace( binning[0], binning[-1],1000)
#	plt.plot( qlist, [ asymptotic_b(q, qasimov_sb)  for q in qlist], label ='Asym. B', color='b')
#	plt.plot( qlist, [ asymptotic_sb(q,qasimov_b)  for q in qlist], label ='Asym. S+B', color='r')
#	plt.legend()
#	#
#	# Evaluate CLs vals
#	#
#	# asymptotic
#	psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(qasimov_b)) 
#	oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(qasimov_sb) )
#	CLs_asym = psb_asym[0]/oneminuspb_asym[0]
#	# python sampling
#	CLs = psb_val/oneminuspb_val
#	#print 'psb_val:' , psb_val
#	#print 'oneminuspb_val:' , oneminuspb_val
#	#print 'CLs: ', CLs
#	CLs_str = "%.3f" % CLs
#	# Zpeed
#	CLs_asym_str = "%.3f" % CLs_asym
#	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
#	#plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=?')
#	if plotname != None:
#		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
#		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
#	plt.clf()
#
#	return {'CLspython:':CLs, 'CLspython_asym:':CLs_asym}
#
#def CLspython_ratio_Tevatron_gauss( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, plotdirectory = None, N=10**2, hilumi=False ):
#	# use plotname to save plot in Teststatistik directory
#	# if None grab variable plotdirname (defined in onemass exclusion plot) to save them in subdirectory there 
#	if plotname==None:
#		filename = os.path.join( plotdirectory, 'Teststatistic', Zp_model['name'] + '_' + str(N) + '.pdf' ) 
#		print filename
#		if not os.path.exists( os.path.join( plotdirectory, 'Teststatistic')):
#			os.makedirs(   os.path.join( plotdirectory, 'Teststatistic'))
#	else:
#		filename = os.path.join( plotdirectory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) 
#	print 'Start sampling for: ', filename
#
#	#
#	# get counts
#	#
#	width = Zp_model['Gamma']
#	MZp = Zp_model['MZp']
#	# define mll range
#	mllrange=[ MZp + searchwindow[0]*width, MZp + searchwindow[1]*width]
#	# SM counts
#	ee_observed = getSMcounts( 'ee', counttype='observed', mllrange = mllrange, lumi =139.)
#	ee_observed_bin = [ x[2] for x in ee_observed ]
#	ee_expected = getSMcounts( 'ee', counttype='expected', mllrange = mllrange, lumi =139.)
#	ee_expected_bin = [ x[2] for x in ee_expected ]
#	mm_observed = getSMcounts( 'mm', counttype='observed', mllrange = mllrange, lumi =139.)
#	mm_observed_bin = [ x[2] for x in mm_observed ]
#	mm_expected = getSMcounts( 'mm', counttype='expected', mllrange = mllrange, lumi =139.)
#	mm_expected_bin = [ x[2] for x in mm_expected ]
#	# BSM counts: note: this is actually SM+signal
#	ee_bsm   = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	ee_bsm_bin = [ x[2] for x in ee_bsm ]
#	mm_bsm   = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange =  mllrange, withinterference = withint )
#	mm_bsm_bin = [ x[2] for x in mm_bsm ]
#
#	print 'this are the counts'
#	print 'ee observed', ee_observed_bin
#	print 'ee expexted', ee_expected_bin
#	ee_signal_bin = [ bsm-sm for bsm,sm in zip(ee_bsm_bin, ee_expected_bin)]
#	print 'ee signal', ee_signal_bin 
#
#	print 'mm observed', mm_observed_bin
#	print 'mm expexted', mm_expected_bin
#	mm_signal_bin = [ bsm-sm for bsm,sm in zip(mm_bsm_bin, mm_expected_bin)]
#	print 'mm signal',   mm_signal_bin
#
#	#
#	# teststatistik, remember observations are k's from poission, and s*mu+bkg are parameters of poisson
#	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2  ln L (mu=1) + 2 ln L (mu=0)) = chi2(s+b) - chi2(b) 
#	# Recall: -2 loglikelihood of gaus -> chi2, here i used gauss implementation, but there is no difference
#	#
#	# some helpers
#	def getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, hilumi=False):
#		splusb = 0.
#		b = 0.
#		for i in range(len(ee_observed_bin)):
#			if mm_observed_bin[i] != 0:
#				y_data = float(ee_observed_bin[i])/float(mm_observed_bin[i])
#				y_pred_splusb = float(ee_expected_bin[i] + 1.*ee_signal_bin[i])/float(mm_expected_bin[i] + 1.*mm_signal_bin[i])
#				if not hilumi:
#					uncert_splusb = y_pred_splusb * np.sqrt( 1./(ee_expected_bin[i]+1.*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+1.*mm_signal_bin[i])) 
#					# to compare to asymptotics
#					#uncert_splusb = y_data * np.sqrt( 1./ee_observed_bin[i] + 1./mm_observed_bin[i]) 
#				else:
#					uncert_splusb = y_pred_splusb * np.sqrt( 1./(21.9*(ee_expected_bin[i]+1.*ee_signal_bin[i])) + 1./(21.9*(mm_expected_bin[i]+1*mm_signal_bin[i]))) 
#					# to compare to asymptotics
#					#uncert_splusb = y_data * np.sqrt( 1./(21.9*ee_observed_bin[i]) + 1./(21.9*mm_observed_bin[i])) 
#				splusb += norm.logpdf( y_data , loc = y_pred_splusb, scale =uncert_splusb)	
#
#				y_pred_b = float( ee_expected_bin[i] + 0*ee_signal_bin[i])/float(mm_expected_bin[i] + 0*mm_signal_bin[i])
#				if not hilumi:
#					uncert_b = y_pred_b * np.sqrt( 1./(ee_expected_bin[i]+0*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+0*mm_signal_bin[i])) 
#					# to compare to asymptotics
#					#uncert_b = y_data * np.sqrt( 1./ee_observed_bin[i] + 1./mm_observed_bin[i]) 
#				else:
#					uncert_b = y_pred_b * np.sqrt( 1./(21.9*(ee_expected_bin[i]+0*ee_signal_bin[i])) + 1./(21.9*(mm_expected_bin[i]+0*mm_signal_bin[i]))) 
#					# to compare to asymptotics
#					#uncert_b = y_data * np.sqrt( 1./(21.9*ee_observed_bin[i]) + 1./(21.9*mm_observed_bin[i]) )
#				b += norm.logpdf( y_data , loc = y_pred_b, scale =uncert_b)	
#			else:
#				print 'WARNING mm_observed_bin[%i]=0'%i
#				splusb += 0.
#				b += 0.
#			return -2*(splusb - b)
#
#	#
#	# observed test statistic
#	#
#	qobs = getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, hilumi=hilumi) 
#	#qasimov_b = getteststatistik( ee_expected_bin, ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_expected_bin, mm_signal_bin, hilumi=hilumi) 
#	#qasimov_sb= getteststatistik(  [ b+s for b,s in zip(ee_expected_bin,ee_signal_bin) ], ee_expected_bin, ee_signal_bin,  [ b+s for b,s in zip(mm_expected_bin,mm_signal_bin) ], mm_expected_bin,	mm_signal_bin, hilumi=hilumi)
#
#	#
#	# sampling
#	#
#	def gettoydata( expectations_ee, expectations_mm ):
#		toydata_ee = []
#		for exp in expectations_ee:
#			toydata_ee.append(poisson.rvs(exp))
#		toydata_mm = []
#		for exp in expectations_mm:
#			toydata_mm.append(poisson.rvs(exp))
#		return toydata_ee, toydata_mm
#
#	def get_qlist( ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin, mu=0, N=100, hilumi=hilumi):
#		q = []
#		for i in range(N):
#			if (np.log10(i+1) % 1) == 0:
#				print i+1   
#			toydata_ee, toydata_mm = gettoydata( [b + mu *s for b,s in zip(ee_expected_bin, ee_signal_bin) ] , [b + mu *s for b,s in zip(mm_expected_bin, mm_signal_bin) ] )
#
#			q.append( getteststatistik( toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin, hilumi=hilumi) )
#		return q
#
#	# sampled test statistic
#	qsb = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=1., N=N, hilumi=hilumi)
#	qb  = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=0., N=N, hilumi=hilumi)
#
#	#
#	# histo of sampled test statistik
#	#
#	plotdirname = 'Teststatistik'
#	plot_directory = os.path.join( plotdir, plotdirname )
#	if not os.path.exists(plot_directory):
#    		os.makedirs(plot_directory) 
#	# prepare plot and plot
#	numberofbins = 50 #eachside of qobs
#	spacing = max( qobs - min(qsb), max(qsb)-qobs, qobs - min(qb), max(qb)-qobs, )/(numberofbins)
#	binning = []
#	for i in range(-numberofbins,numberofbins+1):
#		binning.append( qobs + i*spacing)
#	qsb_hist = plt.hist( qsb, bins=binning, density = True, label = 'S+B', log=True, alpha=0.5 ,color='r' )
#	qb_hist = plt.hist( qb, bins=binning, density = True,  label = 'B', log=True   , alpha=0.5 ,color='b' )
#	# scale histo
#	#plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
#
#	#
#	# obtain CLs and q values: [-2sig, -sig, median, sig, 2sig, observed]
#	#
#	CLs_values, q_values = getCLs_values( qsb_hist, qb_hist, qobs, spacing, numberofbins)
#
#	#
#	# plot q_values, make pretty and save plot
#	#
#	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
#	for q_value, tag in zip(q_values[:-1], [ r'$q_{exp}^{2 \sigma}$', r'$q_{exp}^{ \sigma}$', r'$q_{exp}^{ med}$', r'$q_{exp}^{-\sigma}$', r'$q_{exp}^{-2 \sigma}$' ]):
#		plt.axvline( q_value, color = 'black', linestyle='dashed', label= tag )
#	plt.xlabel(r'$q_{obs}$')
#	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}, \mathcal{L}=\mathcal{G}$')
#	plt.legend()
#
#	# asymptotic does not work out since i am drawing poisson, but testing gauss
#	##
#	## plot asymptotic
#	##
#	#def asymptotic_b(x, qasimov):
#	#	# asimov evaluatat at testmu
#	#	sig = np.sqrt(1./ abs(qasimov) ) 
#
#	#	loc = 1./sig**2 # mu of gauss
#	#	scale = 2./sig # sigma of gauss
#	#	return norm.pdf(x, loc=loc, scale=scale)
#
#	#def asymptotic_sb(x, qasimov):
#	#	# asimov evaluatat at testmu
#	#	sig = np.sqrt(1./ abs(qasimov) )
#
#	#	loc = -1./sig**2 # mu of gauss
#	#	scale = 2./sig # sigma of gauss
#	#	return norm.pdf(x, loc=loc, scale=scale)
#
#	#qlist = np.linspace( binning[0], binning[-1],1000)
#	#plt.plot( qlist, [ asymptotic_b( q, qasimov_sb) for q in qlist], label ='Asym. B',   color='b')
#	#plt.plot( qlist, [ asymptotic_sb(q, qasimov_b ) for q in qlist], label ='Asym. S+B', color='r')
#	#plt.legend()
#	CLs_obs_str = "%.3f" % CLs_values[-1]
#	CLs_exp_str = "%.3f" % CLs_values[2]
#	#plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
#	plt.title( 'CLs_obs=' + CLs_obs_str + ' / CLs_exp=' + CLs_exp_str + ' / N ' + str(N) )
#	plt.savefig( filename )
#	print 'Teststatistik saved as ', filename
#	plt.clf()
#
#	return CLs_values

if __name__ == "__main__":
	from ZPEEDmod.Zpeedcounts import getZpmodel_sep
	import argparse
	import time
	from directories.directories import plotdir 
	import os

	argParser = argparse.ArgumentParser(description = "Argument parser")
	#argParser.add_argument('--tag',       		default='Test' ,  help='Tag for files', )
	argParser.add_argument('--powN',      type=int, default=2,  help='power of 10 for toydata', )
	# defaultvalues s.t CLs approx 0.05 - asympotics works perfect
	argParser.add_argument('--ge',      type=float, default=0.1,  help='ge coupling', )
	argParser.add_argument('--gm',      type=float, default=1.0,  help='gm coupling', )
	argParser.add_argument('--M',       type=float, default=1000.,  help='Resonance mass', )
	args = argParser.parse_args()

	plotdirectory = os.path.join(plotdir, 'Teststatistik')
	# implement a flavor universal Zp model, to compare with ZPEED/RunMe.py
	# it workes out!
	ge= args.ge
	gm= args.gm
	MZp = args.M
	model = 'VV'
	Zp_model = getZpmodel_sep(ge ,gm , MZp, model = model,  WZp = 'auto')
	# just for comparison purpuse
	#Zp_model['gtv']=Zp_model['gev'] 
	#Zp_model['gta']=Zp_model['gea'] 
	#Zp_model['gntv']=Zp_model['gnev'] 
	#Zp_model['gnta']=Zp_model['gnea'] 
	#Zp_model['Gamma']=myDecayWidth(Zp_model) 
	print Zp_model

	#
	# Uncert
	#
	#print 'CLsZpeed_uncert: ', CLsZpeed_uncert( Zp_model,  searchwindow=[-3.,3.], withint = False, plotname= None, plotdirectory=None, N=10**args.powN, Lunc = 0.03 )
	#print 'CLsZpeed_sampling: ', CLsZpeed_sampling( Zp_model,  searchwindow=[-3.,3.], withint = False, plotname= None, plotdirectory=None, 	N=10**args.powN)
	#print 'CLsZpeed_withexpected: ', CLsZpeed_withexpected( Zp_model,  searchwindow=[-3.,3.], withint = False )

	#
	# Searches in mll
	#
	##print 'Compare CLs values: rembember CLsPython without weight --> therefore differences (especially if smaller 3 widht)'
	#print 'CLsZpeed: ', CLsZpeed( Zp_model,  searchwindow = [-3.,3.], withint = False )
	#print 'CLsZpeed_withexpected: ', CLsZpeed_withexpected( Zp_model,  searchwindow=[-3.,3.], withint = False )
	## ZPEED Sampling
	##start_time = time.clock()
	print 'CLsZpeed_sampling:', CLsZpeed_sampling( Zp_model,  searchwindow= [-3.,3.], withint = False, plotname= None , plotdirectory=plotdirectory, N =10**args.powN )
	##end_time = time.clock()
	##print 'Duration CLsZpeed_sampling in min for 10^x ' + str(args.powN) + ': ' + str((end_time-start_time)/60.) 

	#
	# Searches in ratio
	#
	variant = None # 'rev' or 'alb'
	#print 'CLsRatio_withexpected: ', CLsRatio_withexpected( Zp_model,  searchwindow=[-3.,3.], withint = False, variant = variant )
	# Ratio Sampling
	#start_time = time.clock()
	#print 'CLsRatio_sampling: ', CLsRatio_sampling( Zp_model, searchwindow= [-3.,3.], withint = False, plotname= None , plotdirectory=plotdirectory, N =10**args.powN, variant = variant )
	#end_time = time.clock()
	#print 'Duration CLsRatio_sampling in min for 10^x ' + str(args.powN) + ': ' + str((end_time-start_time)/60.) 

	#
	# just one bin
	#
	#print 'CLsZpeed_withexpected: ', CLsZpeed_withexpected( Zp_model,  searchwindow=[0.,0.2], withint = False )
	#print 'CLsZpeed_sampling: ', CLsZpeed_sampling( Zp_model,  searchwindow=[0.,0.2], withint = False, plotname= None, plotdirectory=None, 	N=10**args.powN)

	#
	#old implementations:
	#
	#	Tevatron teststatistic in mll search
	#	Tevatron teststatistic in mll search with gauss Likelihood
	#	Tevatron teststatistic in mll search with chi2 (actuatlly not a Likelihood)
	#	Tevatron teststatistic in ratio search with gauss Likelihood but Poisson sampling (WRONG!!)
