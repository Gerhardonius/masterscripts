##########################################################################
# check generic Zp model:
# conclusion: 
#	1) counts depend on Gamma: the higher Gamma the broader the res, the more dilluted the peak, hence less count difference
#	2) if I change the coupling in ee channel, then Gamma changes, which affects mm channel
#	3) so for unfixed Gamma - no separated exclusion possible since we have Gamme dependence
##########################################################################
from ZPEEDmod.Zpeedcounts import getZpmodel_gen, getSMcounts, getBSMcounts


print 'Check generic Zp_model'
gp = 0.01
MZp = 1000.
WZp = 'auto'
#WZp = 0.03

#
# vary ge
#
ges = [0., 0.25, 0.5, 0.75, 1.]
gm = 1.
print 'vary ge'
for ge in ges:
	#Zp_model
	Zp_model = getZpmodel_gen(ge ,gm , MZp, gp ,  WZp = WZp)
	#deltaM = 1*Zp_model['Gamma']	
	#mllrange = [ MZp - deltaM, MZp + deltaM]
	mllrange = [ MZp - 25., MZp + 25.]
	wint = False
	# Get counts
	sm_counts_ee = getSMcounts( 'ee', counttype='expected', mllrange = mllrange , lumi =139.)
	bsm_counts_ee = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange = mllrange , withinterference = wint )
	sm_counts_mm = getSMcounts( 'mm', counttype='expected', mllrange = mllrange , lumi =139.)
	bsm_counts_mm = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange = mllrange , withinterference = wint )
	print 'ge/gm/Gamma ', ge, gm, Zp_model['Gamma']
	e_signal = []
	for i in range(len(sm_counts_ee)):
		e_signal.append(bsm_counts_ee[i][2]-sm_counts_ee[i][2])
	m_signal = []
	for i in range(len(sm_counts_mm)):
		m_signal.append(bsm_counts_mm[i][2]-sm_counts_mm[i][2])
	print [ int(e) for e in e_signal], [ int(m) for m in m_signal]

#
# vary gm
#
print 'vary gm'
gms = [0., 0.25, 0.5, 0.75, 1.]
ge = 1.
for gm in gms:
	#Zp_model
	Zp_model = getZpmodel_gen(ge ,gm , MZp, gp ,  WZp = WZp)
	#deltaM = 1*Zp_model['Gamma']	
	#mllrange = [ MZp - deltaM, MZp + deltaM]
	mllrange = [ MZp - 25., MZp + 25.]
	wint = False
	# Get counts
	sm_counts_ee = getSMcounts( 'ee', counttype='expected', mllrange = mllrange , lumi =139.)
	bsm_counts_ee = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange = mllrange , withinterference = wint )
	sm_counts_mm = getSMcounts( 'mm', counttype='expected', mllrange = mllrange , lumi =139.)
	bsm_counts_mm = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange = mllrange , withinterference = wint )
	print 'ge/gm/Gamma ', ge, gm, Zp_model['Gamma']
	e_signal = []
	for i in range(len(sm_counts_ee)):
		e_signal.append(bsm_counts_ee[i][2]-sm_counts_ee[i][2])
	m_signal = []
	for i in range(len(sm_counts_mm)):
		m_signal.append(bsm_counts_mm[i][2]-sm_counts_mm[i][2])
	print [ int(e) for e in e_signal], [ int(m) for m in m_signal]

#
# check sum of counts in SR: +/- 3 Gamma'
#
gms = [0., 0.25, 0.5, 0.75, 1.]
ge = 1.
print 'check sum of counts in SR: +/- 3 Gamma'
for gm in gms:
	#Zp_model
	Zp_model = getZpmodel_gen(ge ,gm , MZp, gp ,  WZp = WZp)
	deltaM = 5*Zp_model['Gamma']	
	mllrange = [ MZp - deltaM, MZp + deltaM]
	#mllrange = [ MZp - 25., MZp + 25.]
	wint = False
	# Get counts
	sm_counts_ee = getSMcounts( 'ee', counttype='expected', mllrange = mllrange , lumi =139.)
	bsm_counts_ee = getBSMcounts( 'ee', Zp_model, lumi =139., mllrange = mllrange , withinterference = wint )
	sm_counts_mm = getSMcounts( 'mm', counttype='expected', mllrange = mllrange , lumi =139.)
	bsm_counts_mm = getBSMcounts( 'mm', Zp_model, lumi =139., mllrange = mllrange , withinterference = wint )
	print 'ge/gm/Gamma ', ge, gm, Zp_model['Gamma']
	e_signal = []
	for i in range(len(sm_counts_ee)):
		e_signal.append(bsm_counts_ee[i][2]-sm_counts_ee[i][2])
	m_signal = []
	for i in range(len(sm_counts_mm)):
		m_signal.append(bsm_counts_mm[i][2]-sm_counts_mm[i][2])
	print [ int(e) for e in e_signal], [ int(m) for m in m_signal]
	print sum(e_signal), sum(m_signal)


