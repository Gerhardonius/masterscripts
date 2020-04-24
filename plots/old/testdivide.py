import ROOT
from math import sqrt
from array import array

num = ROOT.TGraphAsymmErrors(  ) # copy histo
den = ROOT.TGraphAsymmErrors(  ) # copy histo

#graph.SetPoint( i, center, counts ) # point number, centervalue and counts
#graph.SetPointError( i, 0, 0, errDown, errUp) # location of low and up error in counts

for i in range(1,11):
	num.SetPoint( i, i, i ) # point number, centervalue and counts
	num.SetPointError( i, 0, 0, 2, 3) # location of low and up error in counts

for i in range(1,11):
	den.SetPoint( i, i, i ) # point number, centervalue and counts
	den.SetPointError( i, 0, 0, 2, 3) # location of low and up error in counts

res = ROOT.TGraphAsymmErrors(  ) # copy histo
for i in range(1,11):
	#res.SetPoint( i, i, num.GetBinContent(i)/den.GetBinContent(i) ) # point number, centervalue and counts
	errorup = sqrt( ( num.GetErrorXhigh( i ) / den.GetBinContent(i) )**2 + ( den.GetBinContent(i) * den.GetErrorXhigh( i ) / den.GetBinContent(i) )**2) 
	errordo = sqrt( ( num.GetErrorXlow( i ) / den.GetBinContent(i) )**2 + ( den.GetBinContent(i) * den.GetErrorXlow( i ) / den.GetBinContent(i) )**2  )
	res.SetPointError( i, 0, 0, errordo, errorup ) # point number, centervalue and counts
