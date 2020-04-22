# helper functions for Zpeed plotting
import ROOT
from array import array
import math
import matplotlib.pyplot as plt
import os

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

def CLspython( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2 ):
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
	# extract p values from plot
	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
	#print qsb_hist[0][numberofbins:] #10bins above qobs 
	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print qb_hist[0][:numberofbins] #10bins above qobs 
	#print 'Integral qb:', sum(qb_hist[0])*spacing
	# finish plot
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
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
	#
	# Evaluate CLs vals
	#
	# asymptotic
	psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(chi2_Asimov(1.)) )
	oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(chi2_Asimov(1.)) )
	CLs_asym = psb_asym[0]/oneminuspb_asym[0]
	# python sampling
	CLs = psb_val/oneminuspb_val
	#print 'psb_val:' , psb_val
	#print 'oneminuspb_val:' , oneminuspb_val
	#print 'CLs: ', CLs
	CLs_str = "%.3f" % CLs
	# Zpeed
	CLs_asym_str = "%.3f" % CLs_asym
	Zpeedresult_str = "%.3f" % Zpeedresult
	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str + ' / ' + 'ZPEED=' + Zpeedresult_str)
	if plotname != None:
		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
	plt.clf()

	return {'CLspython:':CLs, 'CLspython_asym:':CLs_asym, 'Zpeedresult(noweights)':Zpeedresult}

def CLspython_Tevatron( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2 ):
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
	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2 ( ln L (mu=1) - ln L (mu=0)) 
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
	#
	# observed test statistic
	#
	qobs = sum(getloglikelihood( 1., ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin,	mm_signal_bin)) - sum(getloglikelihood( 0., ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin))

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

			q.append( sum(getloglikelihood( 1. , toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin)) -
			sum(getloglikelihood( 0. , toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin)) )
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
	# scale histo
	#plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
	# extract p values from plot
	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
	#print qsb_hist[0][numberofbins:] #10bins above qobs 
	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print qb_hist[0][:numberofbins] #10bins above qobs 
	#print 'Integral qb:', sum(qb_hist[0])*spacing
	# finish plot
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
	plt.xlabel(r'$q_{obs}$')
	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}$')
	#
	# plot asymptotic
	#
	def asymptotic_b(x, asimov, testmu=1.):
		# asimov evaluatat at testmu
		sig = sqrt_safe( testmu**2/asimov )

		loc = 1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)

	def asymptotic_sb(x, asimov, testmu=1.):
		# asimov evaluatat at testmu
		sig = sqrt_safe( testmu**2/asimov )

		loc = -1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)

	qlist = np.linspace( binning[0], binning[-1],1000)
	plt.plot( qlist, [ asymptotic_b(q, chi2_Asimov(1.) ) for q in qlist], label ='Asym. B', color='b')
	plt.plot( qlist, [ asymptotic_sb(q,chi2_Asimov(1.) ) for q in qlist], label ='Asym. S+B', color='r')
	plt.legend()
	#
	# Evaluate CLs vals
	#
	# asymptotic
	psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(chi2_Asimov(1.)) )
	oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(chi2_Asimov(1.)) )
	CLs_asym = psb_asym[0]/oneminuspb_asym[0]
	# python sampling
	CLs = psb_val/oneminuspb_val
	#print 'psb_val:' , psb_val
	#print 'oneminuspb_val:' , oneminuspb_val
	#print 'CLs: ', CLs
	CLs_str = "%.3f" % CLs
	# Zpeed
	CLs_asym_str = "%.3f" % CLs_asym
	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
	if plotname != None:
		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
	plt.clf()

	return {'CLspython:':CLs, 'CLspython_asym:':CLs_asym}

def CLspython_Tevatron_gauss( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2 ):
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
	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2* (loglik (s+b) -loglik (b) ) 
	#
	# some helpers
	def getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin):
		splusb = 0.
		b = 0.
		for i in range(len(ee_observed_bin)):
			y_data = ee_observed_bin[i]
			y_pred_splusb = ee_expected_bin[i] + 1.*ee_signal_bin[i]
			uncert_splusb = np.sqrt((ee_expected_bin[i]+1.*ee_signal_bin[i]))
			splusb += norm.logpdf( y_data , loc = y_pred_splusb, scale =uncert_splusb)	

			y_pred_b = ee_expected_bin[i] + 0*ee_signal_bin[i]
			uncert_b = np.sqrt((ee_expected_bin[i]+0*ee_signal_bin[i]))
			b += norm.logpdf( y_data , loc = y_pred_b, scale =uncert_b)	
		for i in range(len(mm_observed_bin)):
			y_data = mm_observed_bin[i]
			y_pred_splusb = mm_expected_bin[i] + 1.*mm_signal_bin[i]
			uncert_splusb = np.sqrt((mm_expected_bin[i]+1.*mm_signal_bin[i]))
			splusb += norm.logpdf( y_data , loc = y_pred_splusb, scale =uncert_splusb)	

			y_pred_b = mm_expected_bin[i] + 0*mm_signal_bin[i]
			uncert_b = np.sqrt((mm_expected_bin[i]+0*mm_signal_bin[i]))
			b += norm.logpdf( y_data , loc = y_pred_b, scale =uncert_b)	

		return -2*(splusb - b)

	#
	# observed test statistic
	#
	qobs = 		getteststatistik(  ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin,	mm_signal_bin)
	qasimov_b = 	getteststatistik(  ee_expected_bin, ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_expected_bin,	mm_signal_bin)
	qasimov_sb = 	getteststatistik(  [ b+s for b,s in zip(ee_expected_bin,ee_signal_bin) ], ee_expected_bin, ee_signal_bin,  [ b+s for b,s in zip(mm_expected_bin,mm_signal_bin) ], mm_expected_bin,	mm_signal_bin)

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

			q.append( getteststatistik( toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin))
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
	# scale histo
	#plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
	# extract p values from plot
	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
	#print qsb_hist[0][numberofbins:] #10bins above qobs 
	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print qb_hist[0][:numberofbins] #10bins above qobs 
	#print 'Integral qb:', sum(qb_hist[0])*spacing
	# finish plot
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
	plt.xlabel(r'$q_{obs}$')
	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}, \mathcal{L} = \mathcal{G}$')

	#
	# plot asymptotic
	#
	def asymptotic_b(x, qasimov):
		# asimov evaluatat at testmu
		sig = np.sqrt( 1./ abs(qasimov) )

		loc = 1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)

	def asymptotic_sb(x, qasimov):
		# asimov evaluatat at testmu
		sig = np.sqrt( 1./ abs(qasimov) )

		loc = -1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)

	qlist = np.linspace( binning[0], binning[-1],1000)
	#plt.plot( qlist, [ asymptotic_b(q, qasimov_b)  for q in qlist], label ='Asym. B', color='b')
	plt.plot( qlist, [ asymptotic_b(q, qasimov_sb)  for q in qlist], label ='Asym. B', color='b') # these are the correct ones..determined by trying
	#plt.plot( qlist, [ asymptotic_sb(q, qasimov_sb)  for q in qlist], label ='Asym. S+B', color='r')
	plt.plot( qlist, [ asymptotic_sb(q, qasimov_b)  for q in qlist], label ='Asym. S+B', color='r') # these are the correct ones..determined by trying
	plt.legend()
	#
	# Evaluate CLs vals
	#
	# asymptotic
	psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(qasimov_b) )
	oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(qasimov_sb) )
	CLs_asym = psb_asym[0]/oneminuspb_asym[0]
	# python sampling
	CLs = psb_val/oneminuspb_val
	#print 'psb_val:' , psb_val
	#print 'oneminuspb_val:' , oneminuspb_val
	#print 'CLs: ', CLs
	CLs_str = "%.3f" % CLs
	# Zpeed
	CLs_asym_str = "%.3f" % CLs_asym
	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
	if plotname != None:
		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
	plt.clf()

	return {'CLspython:':CLs, 'CLspython_asym:':CLs_asym}

def CLspython_Tevatron_chi2( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2 ):
	# Note: this is basecally the same as gauss, with the backdraw that the asymtotic does not work out

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
	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = Now use chi2 
	#
	# some helpers
	def getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin):
		splusb = 0.
		b = 0.
		for i in range(len(ee_observed_bin)):
			y_data = ee_observed_bin[i]
			y_pred_splusb = ee_expected_bin[i] + 1.*ee_signal_bin[i]
			uncert_splusb = np.sqrt((ee_expected_bin[i]+1.*ee_signal_bin[i]))
			splusb += (y_data - y_pred_splusb)**2/uncert_splusb**2	

			y_pred_b = ee_expected_bin[i] + 0*ee_signal_bin[i]
			uncert_b = np.sqrt((ee_expected_bin[i]+0*ee_signal_bin[i]))
			b += (y_data - y_pred_b)**2/uncert_b**2	
		for i in range(len(mm_observed_bin)):
			y_data = mm_observed_bin[i]
			y_pred_splusb = mm_expected_bin[i] + 1.*mm_signal_bin[i]
			uncert_splusb = np.sqrt((mm_expected_bin[i]+1.*mm_signal_bin[i]))
			splusb += (y_data - y_pred_splusb)**2/uncert_splusb**2	

			y_pred_b = mm_expected_bin[i] + 0*mm_signal_bin[i]
			uncert_b = np.sqrt((mm_expected_bin[i]+0*mm_signal_bin[i]))
			b += (y_data - y_pred_b)**2/uncert_b**2	

		return splusb - b

	#
	# observed test statistic
	#
	qobs = 		getteststatistik(  ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin,	mm_signal_bin)
	# Asmivos: Note: qasimov_sb is negative! 
	qasimov_b = 	getteststatistik(  ee_expected_bin, ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_expected_bin,	mm_signal_bin)
	qasimov_sb = 	getteststatistik(  [ b+s for b,s in zip(ee_expected_bin,ee_signal_bin) ], ee_expected_bin, ee_signal_bin,  [ b+s for b,s in zip(mm_expected_bin,mm_signal_bin) ], mm_expected_bin,	mm_signal_bin)

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

			q.append( getteststatistik( toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin))
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
	# scale histo
	# plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
	# extract p values from plot
	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
	#print qsb_hist[0][numberofbins:] #10bins above qobs 
	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print qb_hist[0][:numberofbins] #10bins above qobs 
	#print 'Integral qb:', sum(qb_hist[0])*spacing
	# finish plot
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
	plt.xlabel(r'$q_{obs}$')
	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}, -2 \ln \mathcal{L} \approx \chi^2$')
	#
	# plot asymptotic
	#
#
	def asymptotic_b(x, qasimov):
		# asimov evaluatat at testmu
		sig = np.sqrt( 1./ abs(qasimov) )

		loc = 1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)

	def asymptotic_sb(x, qasimov):
		# asimov evaluatat at testmu
		sig = np.sqrt( 1./ abs(qasimov) )

		loc = -1./sig**2 # mu of gauss
		scale = 2./sig # sigma of gauss
		return norm.pdf(x, loc=loc, scale=scale)
	qlist = np.linspace( binning[0], binning[-1],1000)
	plt.plot( qlist, [ asymptotic_b(q, qasimov_sb)  for q in qlist], label ='Asym. B', color='b')
	plt.plot( qlist, [ asymptotic_sb(q,qasimov_b)  for q in qlist], label ='Asym. S+B', color='r')
	plt.legend()
	#
	# Evaluate CLs vals
	#
	# asymptotic
	psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(qasimov_b)) 
	oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(qasimov_sb) )
	CLs_asym = psb_asym[0]/oneminuspb_asym[0]
	# python sampling
	CLs = psb_val/oneminuspb_val
	#print 'psb_val:' , psb_val
	#print 'oneminuspb_val:' , oneminuspb_val
	#print 'CLs: ', CLs
	CLs_str = "%.3f" % CLs
	# Zpeed
	CLs_asym_str = "%.3f" % CLs_asym
	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
	#plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=?')
	if plotname != None:
		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
	plt.clf()

	return {'CLspython:':CLs, 'CLspython_asym:':CLs_asym}

def CLspython_ratio_Tevatron_gauss( Zp_model,  searchwindow=[-3.,3.], withint = True, plotname= None, N=10**2, hilumi=False ):
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
	# Tevatron test statistik q=-2ln( Likelihood(mu=1) / Likelihood(mu=0)) = -2  ln L (mu=1) + 2 ln L (mu=0)) = chi2(s+b) - chi2(b) 
	# Recall: -2 loglikelihood of gaus -> chi2
	#
	# some helpers
	def getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, hilumi=False):
		splusb = 0.
		b = 0.
		for i in range(len(ee_observed_bin)):
			if mm_observed_bin[i] != 0:
				fakeval = 1. # scales all ee counts
				y_data = 1./(fakeval*(float(ee_observed_bin[i])/float(mm_observed_bin[i])))
				y_pred_splusb = 1./(fakeval*float(ee_expected_bin[i] + 1.*ee_signal_bin[i])/float(mm_expected_bin[i] + 1.*mm_signal_bin[i]))
				if not hilumi:
					uncert_splusb = y_pred_splusb * np.sqrt( 1./(ee_expected_bin[i]+1.*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+1.*mm_signal_bin[i])) 
					#uncert_splusb = 1.
				else:
					uncert_splusb = y_pred_splusb * np.sqrt( 1./(21.9*(ee_expected_bin[i]+1.*ee_signal_bin[i])) + 1./(21.9*(mm_expected_bin[i]+1*mm_signal_bin[i]))) 
				splusb += norm.logpdf( y_data , loc = y_pred_splusb, scale =uncert_splusb)	

				y_pred_b = 1./(fakeval*float( ee_expected_bin[i] + 0*ee_signal_bin[i])/float(mm_expected_bin[i] + 0*mm_signal_bin[i]))
				if not hilumi:
					uncert_b = y_pred_b * np.sqrt( 1./(ee_expected_bin[i]+0*ee_signal_bin[i]) + 1./(mm_expected_bin[i]+0*mm_signal_bin[i])) 
					#uncert_b = 1.
				else:
					uncert_b = y_pred_b * np.sqrt( 1./(21.9*(ee_expected_bin[i]+0*ee_signal_bin[i])) + 1./(21.9*(mm_expected_bin[i]+0*mm_signal_bin[i]))) 
				b += norm.logpdf( y_data , loc = y_pred_b, scale =uncert_b)	
			else:
				print 'WARNING mm_observed_bin[%i]=0'%i
				splusb += 0.
				b += 0.
			#print '%.2f %.2f %.2f %.2f %.2f'%(y_data, y_pred_b, uncert_b, y_pred_splusb, uncert_splusb )
		# just to check wheather the script is correct
		#for i in range(len(ee_observed_bin)):
		#	if mm_observed_bin[i] != 0:
		#		fakeval = 9. # scales all ee counts
		#		y_data = ee_observed_bin[i]
		#		y_pred_splusb = ee_expected_bin[i] + 1*ee_signal_bin[i]
		#		uncert_splusb = np.sqrt( y_pred_splusb )

		#		splusb += norm.logpdf( y_data , loc = y_pred_splusb, scale =uncert_splusb)	

		#		y_pred_b = ee_expected_bin[i]
		#		uncert_b = np.sqrt( y_pred_b )
		#		b += norm.logpdf( y_data , loc = y_pred_b, scale =uncert_b)	
		#	else:
		#		print 'WARNING mm_observed_bin[%i]=0'%i
		#		splusb += 0.
		#		b += 0.
		#	#print '%.2f %.2f %.2f'%(y_data, y_pred_b, y_pred_splusb)

		return -2*(splusb - b)

	#
	# observed test statistic
	#
	qobs = getteststatistik( ee_observed_bin, ee_expected_bin, ee_signal_bin, mm_observed_bin, mm_expected_bin, mm_signal_bin, hilumi=hilumi) 
	qasimov_b = getteststatistik( ee_expected_bin, ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_expected_bin, mm_signal_bin, hilumi=hilumi) 
	qasimov_sb= getteststatistik(  [ b+s for b,s in zip(ee_expected_bin,ee_signal_bin) ], ee_expected_bin, ee_signal_bin,  [ b+s for b,s in zip(mm_expected_bin,mm_signal_bin) ], mm_expected_bin,	mm_signal_bin, hilumi=hilumi)

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

	def get_qlist( ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin, mu=0, N=100, hilumi=hilumi):
		q = []
		for i in range(N):
			if (np.log10(i+1) % 1) == 0:
				print i+1   
			toydata_ee, toydata_mm = gettoydata( [b + mu *s for b,s in zip(ee_expected_bin, ee_signal_bin) ] , [b + mu *s for b,s in zip(mm_expected_bin, mm_signal_bin) ] )

			q.append( getteststatistik( toydata_ee, ee_expected_bin, ee_signal_bin, toydata_mm, mm_expected_bin, mm_signal_bin, hilumi=hilumi) )
		return q

	# sampled test statistic
	qsb = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=1., N=N, hilumi=hilumi)
	qb  = get_qlist(ee_expected_bin, ee_signal_bin, mm_expected_bin, mm_signal_bin,  mu=0., N=N, hilumi=hilumi)

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
	# scale histo
	plt.ylim(( plt.ylim()[1] - 10**5, plt.ylim()[1])) 
	# extract p values from plot
	psb_val = sum(qsb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print 'Integral qsb:', sum(qsb_hist[0])*spacing
	#print qsb_hist[0][numberofbins:] #10bins above qobs 
	oneminuspb_val = sum(qb_hist[0][numberofbins:])*spacing #50bins above qobs 
	#print qb_hist[0][:numberofbins] #10bins above qobs 
	#print 'Integral qb:', sum(qb_hist[0])*spacing
	# finish plot
	plt.axvline( qobs, color = 'black', label=r'$q_{obs}$')
	plt.xlabel(r'$q_{obs}$')
	plt.xlabel(r'$-2*\ln \frac{ \mathcal{L}(\mu=1)}{ \mathcal{L}(\mu=0)}, \mathcal{L}=\mathcal{G}$')
	#
	# plot asymptotic
	#
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
	plt.plot( qlist, [ asymptotic_b( q, qasimov_sb) for q in qlist], label ='Asym. B',   color='b')
	plt.plot( qlist, [ asymptotic_sb(q, qasimov_b ) for q in qlist], label ='Asym. S+B', color='r')
	plt.legend()
	#
	# Evaluate CLs vals
	#
	# asymptotic
	#psb_asym = quad(asymptotic_sb, qobs, binning[-1], args=(qasimov) )
	#print psb_asym
	#oneminuspb_asym = quad(asymptotic_b, qobs,binning[-1], args=(qasimov) )
	#print oneminuspb_asym
	#CLs_asym = psb_asym[0]/oneminuspb_asym[0]
	# python sampling
	CLs = psb_val/oneminuspb_val
	#print 'psb_val:' , psb_val
	#print 'oneminuspb_val:' , oneminuspb_val
	#print 'CLs: ', CLs
	CLs_str = "%.3f" % CLs
	# Zpeed
	#CLs_asym_str = "%.3f" % CLs_asym
	#plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) + ' / CLs_asym=' + CLs_asym_str )
	plt.title( 'CLs=' + CLs_str + ' - N ' + str(N) )
	if plotname != None:
		plt.savefig( os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' ) )
		print 'Teststatistik saved as ', os.path.join( plot_directory, Zp_model['name'] + '_' + plotname + '_' + str(N) + '.pdf' )
	plt.clf()

	return CLs

if __name__ == "__main__":
	from ZPEEDmod.Zpeedcounts import getZpmodel_sep
	import argparse
	import time

	argParser = argparse.ArgumentParser(description = "Argument parser")
	argParser.add_argument('--tag',       		default='Test' ,  help='Tag for files', )
	argParser.add_argument('--powN',      type=int, default=2,  help='power of 10 for toydata', )
	argParser.add_argument('--ge',      type=float, default=0.5,  help='ge coupling', )
	argParser.add_argument('--gm',      type=float, default=0.5,  help='gm coupling', )
	argParser.add_argument('--M',       type=float, default=1000,  help='Resonance mass', )
	args = argParser.parse_args()

	# implement a flavor universal Zp model, to compare with ZPEED/RunMe.py
	# it workes out!
	ge= args.ge
	gm= args.gm
	MZp = args.M
	model = 'VV'
	Zp_model = getZpmodel_sep(ge ,gm , MZp, model = 'VV',  WZp = 'auto')
	Zp_model['gtv']=Zp_model['gev'] 
	Zp_model['gta']=Zp_model['gea'] 
	Zp_model['gntv']=Zp_model['gnev'] 
	Zp_model['gnta']=Zp_model['gnea'] 
	Zp_model['Gamma']=myDecayWidth(Zp_model) 
	print Zp_model

	#
	# searches in mll bins
	#
	#print 'Compare CLs values: rembember CLsPython without weight --> therefore differences (especially if smaller 3 widht)'
	#print 'Note: 10**5 toys works, but for 10**6 nothing happens'
	## standard +- 3 Gamma
	#print '----------- standard, noint-----------------'
	#print 'CLsZPEED: ', CLsZpeed( Zp_model,  searchwindow = [-3.,3.], withint = False )
	#
	## ZPEED Sampling
	###start_time = time.clock()
	#print 'CLsPython:', CLspython( Zp_model,  searchwindow= [-3.,3.], withint = False, plotname= modelname , N =10**args.powN )
	###end_time = time.clock()
	##print 'Duration in min for 10^x ' + str(args.powN) + ': ' + str((end_time-start_time)/60.) 

	## Tevatron 
	#print 'CLsPython_Tevatron:', CLspython_Tevatron( Zp_model,  searchwindow= [-3.,3.], withint = False, plotname= modelname + '_Tevatron' , N =10**args.powN )

	## Tevatron with gauss/chi2 (mainly to check how to implement it in ratios)
	#print 'CLsPython_Tevatron_gauss:', CLspython_Tevatron_gauss( Zp_model,  searchwindow= [-3.,3.], withint = False, plotname= modelname + '_Tevatron_gauss' , N =10**args.powN )
	## chi2 -> asymptotic does not fit well
	#print 'CLsPython_Tevatron_chi2:', CLspython_Tevatron_chi2( Zp_model,  searchwindow= [-3.,3.], withint = False, plotname= modelname + '_Tevatron_chi2' , N =10**args.powN )

	#
	# searches in ratio of mll bins
	#
	# RATIO STUFF
	#start_time = time.clock()
	print 'CLsPython_ratio_Tevatron_gauss:', CLspython_ratio_Tevatron_gauss( Zp_model,  searchwindow= [-3.,3.], withint = False, plotname= 'Tevatron_ratio' , N =10**args.powN, hilumi=False )
	#end_time = time.clock()
	#print 'Duration in min for 10^x ' + str(args.powN) + ': ' + str((end_time-start_time)/60.) 
	#print 'CLsPython_ratio_Tevatron_hilumi:', CLspython_ratio_Tevatron_gauss( Zp_model,  searchwindow= [-3.,3.], withint = False, plotname= 'Tevatron_ratio_hilumi' , N =10**args.powN, hilumi=True )

	## capture one bin
	#print '-----------one bin, noint-----------------'
	#print 'CLsZPEED: ', CLsZpeed( Zp_model,  searchwindow = [-0.,0.1], withint = False)
	#print 'CLsPython:', CLspython( Zp_model,  searchwindow= [-0.,0.1], withint = False, plotname= modelname + '_onebin', N =10**args.powN )

	## capture three bins
	#print '-----------three bins, noint-----------------'
	#print 'CLsZPEED: ', CLsZpeed( Zp_model,  searchwindow = [-1,1], withint = False )
	#print 'CLsPython:', CLspython( Zp_model,  searchwindow= [-1,1], withint = False, plotname= modelname + '_threebin', N =10**args.powN )

	#print 'withint'
	#print CLsZpeed( Zp_model,  searchwindow = [-3.,3.], withint = True )
	#print CLspython( Zp_model,  searchwindow=[-3.,3.], withint = True , plotname=modelname + '_withint', N =10**args.powN )

