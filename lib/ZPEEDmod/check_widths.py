##########################################################################
# check generic Zp model:
# conclusion: 
#	1) counts depend on Gamma: the higher Gamma the broader the res, the more dilluted the peak, hence less count difference
#	2) if I change the coupling in ee channel, then Gamma changes, which affects mm channel
#	3) so for unfixed Gamma - no separated exclusion possible since we have Gamme dependence
##########################################################################
from ZPEEDmod.Zpeedcounts import getZpmodel_sep, getZpmodel_semi
import argparse

argParser = argparse.ArgumentParser(description = "Argument parser")
#argParser.add_argument('--tag',       		default='Test' ,  help='Tag for files', )
argParser.add_argument('--ge',      type=float, default=0.5,  help='ge coupling', )
argParser.add_argument('--gm',      type=float, default=0.5,  help='gm coupling', )
argParser.add_argument('--M',       type=float, default=1000.,  help='Resonance mass', )
args = argParser.parse_args()

ge= args.ge
gm= args.gm
MZp = args.M
model = 'VV'
Zp_model_sep  = getZpmodel_sep(ge ,gm , MZp, model = model,  WZp = 'auto')
Zp_model_semi = getZpmodel_semi(ge ,gm , MZp, model = model,  WZp = 'auto')

#widths
#print 'Zp_model_sep width: ', Zp_model_sep['Gamma'], (Zp_model_sep['Gamma']/Zp_model_sep['MZp'])*100
#print 'Zp_model_sem width: ', Zp_model_semi['Gamma'], (Zp_model_semi['Gamma']/Zp_model_semi['MZp'])*100

#production
print 'Zp_model_sep: ', Zp_model_sep['guv'], Zp_model_sep['gua'], Zp_model_sep['gdv'], Zp_model_sep['gda']
print 'Zp_model_sem: ', Zp_model_semi['guv'], Zp_model_semi['gua'], Zp_model_semi['gdv'], Zp_model_semi['gda']

print 'Ratio: ', Zp_model_sep['guv']/Zp_model_semi['guv']

##
## vary ge
##
#ges = [0., 0.25, 0.5, 0.75, 1.]
#gm = 1.
#print 'vary ge'
#for ge in ges:
#	#Zp_model
#	Zp_model = getZpmodel_gen(ge ,gm , MZp, gp ,  WZp = WZp)

