import ROOT
import os
import sys
import numpy as np
from array import array

from directories.directories import plotdir

#
# directories
#
plotname = 'ATLAS_data_to_model.pdf'
plotdirname = 'RooFit'
plot_directory = os.path.join( plotdir, plotdirname)
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory) 
print 'plot_directory: ', plot_directory

#
# Import data
#
analysis_name = '/users/gerhard.ungersbaeck/masterscripts/lib/ZPEED/ATLAS_13TeV'
lumi = 139.           # in fb^1-

ee_data = np.loadtxt(os.path.join( analysis_name+'/ee_data.dat'),delimiter='\t')
mm_data = np.loadtxt(os.path.join( analysis_name+'/mm_data.dat'),delimiter='\t')

ee_bin_low = ee_data[:,0]
ee_bin_high = ee_data[:,1]
ee_observed = ee_data[:,2]

mm_bin_low = mm_data[:,0]
mm_bin_high = mm_data[:,1]
mm_observed = mm_data[:,2]

#
# fill data in histogram
#
binning_low = array('d')
for i in ee_bin_low:
	binning_low.append( i )
binning_low.append( ee_bin_high[-1] )

histo_ee   = ROOT.TH1D( 'histo_ee', 'histo_ee' , len( ee_observed ) , binning_low)
for i in range(len( ee_observed )):
	# ROOT bins start with 1
	histo_ee.SetBinContent( i+1, ee_observed[i] )
	#print ee_bin_low[i], ee_observed[i]

#
# perform fit in ROOT
#

#integral = histo_ee.Integral(1, int(len(ee_observed)-1) )
#histo_ee.Scale(1./integral)
# define fit function: x...variable, []...parameters
A1 = "(1/pi)*(2.4952/2)/((2.4952/2)*(2.4952/2)+(13000-91.1876)*(13000-91.1876))" 
A2 = "(1-(x/13000))^[1]" 
A3 = "(x/13000)^([2]+[3]*log(x/13000)+[4]*log(x/13000)^2+[5]*log(x/13000)^3)" 
A =  "[0]*(%s)*(%s)*(%s)"%(A1, A2, A3)
# define range of fit and start values
myfit = ROOT.TF1("myfit", A , binning_low[0] , binning_low[-1])
myfit.SetParameter(0,15000) #a
myfit.SetParameter(1,1) #b
myfit.SetParameter(2,0) #p0
myfit.SetParameter(3,0) #p1
myfit.SetParameter(4,0) #p2
myfit.SetParameter(5,0) #p3
# perform fit
histo_ee.Fit(myfit,"E")

# get fit results
chi2  = myfit.GetChisquare()
ndof  = myfit.GetNDF()
a  = myfit.GetParameter(0)
b  = myfit.GetParameter(1)
p0 = myfit.GetParameter(2)
p1 = myfit.GetParameter(3)
p2 = myfit.GetParameter(4)
p3 = myfit.GetParameter(5)

# plot and add result to plot
c = ROOT.TCanvas()
c.SetLogy()
c.SetLogx()
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.03)

histo_ee.Draw("pe")
latex.DrawText(0.5 ,0.80  , "a = %.1f "%(a ))
latex.DrawText(0.5 ,0.75 , "b = %.2f "%(b ))
latex.DrawText(0.5 ,0.70 ,"p0 = %.2f "%(p0 ))
latex.DrawText(0.5 ,0.65 ,"p1 = %.2f "%(p1 ))
latex.DrawText(0.5 ,0.60 ,"p2 = %.2f "%(p2 ))
latex.DrawText(0.5 ,0.55 ,"p3 = %.2f "%(p3 ))
latex.DrawText(0.5 ,0.50 ,"chi2/ndof = %.1f/%d = %.1f"%(chi2 ,ndof ,chi2/ndof))
c.Draw()
c.SaveAs( os.path.join( plot_directory, 'ATLAS_data_hist.pdf' ))
sys.exit()

#
# fit with ROOFit: Note: I lleave it, since it is not so easy to implement a pdf (C) for Roofit
#
#
##
## convert TH1D to RooDataHist - needs to be associated with a RooRealVar
##
#x = ROOT.RooRealVar("x","x", ee_bin_low[1], ee_bin_high[-1])
#l = ROOT.RooArgList(x) # acc to docs
#data = ROOT.RooDataHist("data", "data set with x1", l, histo_ee)
#
##
## plot data
##
##c = ROOT.TCanvas()
##c.SetLogy()
##c.SetLogx()
##pl = x.frame(ROOT.RooFit.Title("ATLAS data"))
##data.plotOn(pl)
##pl.Draw()
##c.Draw()
##c.SaveAs( os.path.join( plot_directory, 'ATLAS_RooDataHist.pdf' ))
#
##
## Create Workspace and a Model
##
#w = ROOT.RooWorkspace("w")
##w.factory("Gaussian:pdf(x[232.51,5806.19],mu[1,-10,10000],sigma[2,0,10000])")
#w.factory("BreitWigner:pdf(x[232.51,5806.19],mean[91.1876,50,150],width[2.4952,2,3])")
#w.factory("BreitWigner:pdf(x[232.51,5806.19],mean[91.1876,50,150],width[2.4952,2,3])")
#
## generate some data
#pdf = w.pdf("pdf")
#x = w.var("x")
##n = 100
##data = pdf.generate(ROOT.RooArgSet(x),n) # <class 'ROOT.RooDataSet'>
## or generate binned data (make a histo out of pdf, then draw - small difference but still)
##data = pdf.generateBinned(ROOT.RooArgSet(x),n)
#data.SetName("data")
#
## perform a fit to the data, and save the fitted model to display uncertainties
#fittedmodel = pdf.fitTo( data , ROOT.RooFit.Minimizer("Minutit2","Migrad"), ROOT.RooFit.Save())
#
## print data and fit 
#c = ROOT.TCanvas()
#pl = x.frame(ROOT.RooFit.Title("Gaussian Fit"))
#data.plotOn(pl)
#pdf.plotOn(pl)
#pdf.paramOn(pl, ROOT.RooFit.Layout(0.6,0.9,0.85))
#
## uncert on hole model
#pdf.plotOn(pl, ROOT.RooFit.VisualizeError( fittedmodel,1), ROOT.RooFit.FillColor(ROOT.kOrange) )
## uncert on single parameters
##mu = w.var("mu")
##pdf.plotOn(pl, ROOT.RooFit.VisualizeError( fittedmodel, ROOT.RooArgSet( mu)), ROOT.RooFit.FillColor(ROOT.kGreen) )
##sigma = w.var("sigma")
##pdf.plotOn(pl, ROOT.RooFit.VisualizeError( fittedmodel, ROOT.RooArgSet( sigma)), ROOT.RooFit.FillColor(ROOT.kCyan) )
##
#pl.Draw()
##c.Draw()
#c.SaveAs( os.path.join( plot_directory, plotname ))
#
## save data to workspace
#getattr(w,'import')(data)
#
