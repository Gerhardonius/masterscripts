from ZPEEDmod.ATLAS_13TeV import ee_resolution, mm_resolution
from directories.directories import plotdir

import os
import numpy as np
import matplotlib.pyplot as plt

plotdirectory = os.path.join( plotdir, 'Resoloution')
if not os.path.exists( plotdirectory ):
	os.makedirs(   plotdirectory )


mll = np.linspace( 500 , 2000, 201 ) #smallest 0, largest 1, separation 0.1

ee_res = [ ee_resolution(m) for m in mll]
mm_res = [ mm_resolution(m) for m in mll]

plt.plot( mll , ee_res, label='dielectron', color ='r')
plt.plot( mll , mm_res, label='dimuon', color ='b')
plt.title('Resolution data (ATLAS) used in  ZPEED')
plt.xlabel(r'$M_{ll}$ [GeV]')
plt.ylabel('Resolution [GeV]')
plt.grid()
plt.legend()
plt.savefig( os.path.join( plotdirectory, 'Resolution.pdf') )
plt.clf()
