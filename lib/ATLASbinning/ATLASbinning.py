# define binning, by lower edges + the upper edge of the final bin
atlasbinning = 	[602.522, 622.634, 643.417, 664.894, 687.087, 710.022, 733.722, 758.213, 783.521, 809.674, 836.7, 864.629, 893.489, 923.313, 954.133, 985.981, 1018.89, 1052.9, 1088.05, 1124.36, 1161.9, 1200.68, 1240.76, 1282.17, 1324.97, 1369.2, 1414.9, 1462.13, 1510.93, 1561.36, 1613.48, 1667.34, 1722.99, 1780.5, 1839.94, 1901.35, 1964.82, 2030.4, 2098.17, 2168.21, 2240.58, 2315.37, 2392.65, 2472.52, 2555.05, 2640.34, 2728.47, 2819.54, 2913.66, 3010.91, 3111.41, 3215.27, 3322.59, 3433.5, 3548.1, 3666.54, 3788.92, 3915.39, 4046.09, 4181.14, 4320.7,]

#other implemenation used
# define kfactors for each bin (list has one entry less than binning list)
#kfactors = [1.] * (len(atlasbinning) -1 )
#
#def applykfactor( plot, kfactors = kfactors):
#	''' applies k factors to plot 
#	plot: filled RooTools Plot, with some binning 
#	kfactors: list with kfactors, or list of lists (for every histo sep) 
#	'''
#	if not isinstance(kfactors[0],list):
#		for histo in plot.histos:
#			histo = histo[0]
#			if len(kfactors)==histo.GetNbinsX():
#				for i in range(histo.GetNbinsX()): # 1 to Nbins-1
#					histo.SetBinContent(i, kfactors[i-1]* histo.GetBinContent(i))
#			else:
#            			raise ValueError( "kfactors do not match binning" )
	
