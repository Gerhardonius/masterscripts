'''
'''
import ROOT
ROOT.gROOT.SetBatch(True)
import argparse
import sys
import os
import itertools
import glob
#Delphes
import uuid
import shutil

#RootTools
from RootTools.core.Sample import Sample
from RootTools.core.DelphesReader import DelphesReader
from RootTools.core.TreeVariable import TreeVariable
from RootTools.core.TreeMaker import TreeMaker
import RootTools.core.logger as logger_rt

#custom stuff
from directories.directories import sampledir, flattreedir  
from helpers import bannerfiletoModelName, createModeldict, deltaPhi

#
# argarser
#
argParser = argparse.ArgumentParser(description='Transform Delphes trees to FlatTrees')
argParser.add_argument('--dir',type=str,		help='relative path to Madresults dir, in which Madraph output dirs are')
argParser.add_argument('--tag',	type=str, default='PP1',help='new subdir name for flattree')
argParser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
argParser.add_argument('--small',    action='store_true' ,   help='work with 1000 events')
argParser.add_argument('--test',    action='store_true' ,   help='check if file structure works')
argParser.add_argument('--sysweight', type=int, default=153,  help='Number of Weight.Weight in Madgraph root files')
args = argParser.parse_args()

#
# logger
#
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

#
# directories
#

# samples
samplepath = os.path.join( sampledir, args.dir)
if not os.path.exists(samplepath):
	print '--path does not exits'	
	sys.exit()

samplesubdirlist = args.dir.split('/')
if not samplesubdirlist[-1]: samplesubdirlist.pop() #pop last enty if empty
samplesubdir = os.path.join( samplesubdirlist[-3],samplesubdirlist[-2], samplesubdirlist[-1]) # example: singularity/MG1/SM_DYjets_dielec
rootfiles = []
for name in glob.glob( os.path.join( samplepath, '*/Events/*/*.root')):
	rootfiles.append(name)
[logger_rt.info("Root files to postprocess: " + rootfile.split('/')[-1]) for rootfile in rootfiles]

#
# split root files in low and high mlls
#
rootfiles_lo = [ rootfile for rootfile in rootfiles if 'lo' in rootfile.split('/')[-1]] 
rootfiles_mi = [ rootfile for rootfile in rootfiles if 'mi' in rootfile.split('/')[-1]] 
rootfiles_hi = [ rootfile for rootfile in rootfiles if 'hi' in rootfile.split('/')[-1]] 

# flattree
outdir = os.path.join( flattreedir, samplesubdir, args.tag )
if not os.path.exists(outdir):
    os.makedirs(outdir) 
logger_rt.info("Flattree directory: " + outdir)
outfile_lo = os.path.join( outdir, samplesubdirlist[-1] + '_lo_small.root' if args.small else samplesubdirlist[-1] + '_lo.root')
outfile_mi = os.path.join( outdir, samplesubdirlist[-1] + '_mi_small.root' if args.small else samplesubdirlist[-1] + '_mi.root')
outfile_hi = os.path.join( outdir, samplesubdirlist[-1] + '_hi_small.root' if args.small else samplesubdirlist[-1] + '_hi.root')
logger_rt.info("Flattree file: " + outfile_lo)
logger_rt.info("Flattree file: " + outfile_mi)
logger_rt.info("Flattree file: " + outfile_hi)

# determine flavor (later used in filler)
if 'dielec' in samplesubdirlist[-1]: flavor = 'dielec'
if 'dimuon' in samplesubdirlist[-1]: flavor = 'dimuon'
if flavor not in ['dielec','dimuon']: sys.exit()

if args.test: sys.exit()
#
# Tree Maker
#

# standard variables
variables_strings = [   'l1_pt/F', 'l1_phi/F', 'l1_eta/F', #leading lepton1
			'l2_pt/F', 'l2_phi/F', 'l2_eta/F', #leading lepton2
			'nLep/I',
			'met_pt/F','met_phi/F', #MET 
			'ht/F', 'nJet/I', # hadronic activity
			'dl_mass/F', 'dl_pt/F', 'dl_phi/F', 'dl_eta/F', 'dPhi_ll/F', #dilepton system
			'weight/F',
			'gen_dl_mass/F', 'gen_dl_pt/F', 'gen_dl_phi/F', 'gen_dl_eta/F', 'gen_dPhi_ll/F', # generator level: dilepton system
			]
# sysweight
for i in range(1,args.sysweight +1):
	variables_strings.append( 'sysweight_' + str(i).zfill(3) + '/F')
# create TreeVariables
new_variables = [ TreeVariable.fromString(x) for x in variables_strings ]

# Filler
def filler(event):
    '''
    Example from roottools: event.MyJet_pt2[i] = reader.event.Jet_pt[i]**2
    my example:             event.met_pt = reader.myelec()
    or directly:            event.l1_pt = reader.event.Electron_PT[0]
    Caution: even though Collection_size = x, Collection_PT[y] gives an unphys. value for y>x
    thats the reason for read_collection
    '''
    #
    # RecoLevel
    #

    #
    # 1:1 translation
    #

    # weight, reads from Event.Weight - sum over all weights = crosssection
    event.weight = reader.weight()[0]['weight']
    if reader.weight()[0]['weight'] > 1.: logger_rt.warning("weight > 1 found: " + str(reader.weight()[0]['weight']) )

    # sysweigth
    listofdirweights = reader.sysweight()
    listofweights = [ dic['sysweight'] for dic in listofdirweights ]
    maxsysweight = max( listofweights ) 
    #if maxsysweight > 1.: logger_rt.warning("sysweight > 1 found: " + str(maxsysweight) )
    for i in range(1, len(listofdirweights) + 1):
    	setattr( event, 'sysweight_' + str(i).zfill(3), listofdirweights[i-1]['sysweight'])

    # Jet
    event.nJet = len( reader.jets() )
    for recojet, rank in zip(reader.jets(), ['j1','j2']):
        for var in ['pt','eta','phi']:
            setattr( event, rank+'_'+var, recojet[var])     

    # hadron activity
    event.ht = reader.scalarHT()[0]['ht']

    # MET
    for var in ['pt','phi']:
        setattr( event, 'met_'+var, reader.met()[0][var])     


    # get leptons based on flavor
    if flavor=='dielec': leptonlist = reader.electrons() 
    if flavor=='dimuon': leptonlist = reader.muons() 

    # leptons
    event.nLep = len( leptonlist )
    for recoelec, rank in zip( leptonlist, ['l1','l2']):
        for var in ['pt','eta','phi']:
            setattr( event, rank+'_'+var, recoelec[var])     

    #
    # new branches
    #

    # dilepton system
    if event.nLep == 2 and leptonlist[0]['charge']*leptonlist[1]['charge']<0:
        l1 = ROOT.TLorentzVector()
        # last argument is mass
        l1.SetPtEtaPhiM(leptonlist[0]['pt'], leptonlist[0]['eta'], leptonlist[0]['phi'], 0)
        l2 = ROOT.TLorentzVector()
        l2.SetPtEtaPhiM(leptonlist[1]['pt'], leptonlist[1]['eta'], leptonlist[1]['phi'], 0)
        dl = l1 + l2

        event.dl_mass   = dl.M()
        event.dl_pt     = dl.Pt() 
        event.dl_phi    = dl.Phi() 
        event.dl_eta    = dl.Eta() 
        event.dPhi_ll   = deltaPhi( leptonlist[0]['phi'], leptonlist[1]['phi']) 

	logger_rt.debug( '*********************************************************************')
	logger_rt.debug( 'reco_Dilepton (dlM, pt, pt):     '+str(dl.M())+', '+str(leptonlist[0]['pt'])+', '+str(leptonlist[1]['pt']))
    else:
	event.dl_mass   = -1.
        event.dl_pt     = -1. 
        event.dl_phi    = -10.
        event.dl_eta    = -10. 
        event.dPhi_ll   = -10.

    #
    # GenLevel
    #

    if flavor=='dielec': pdgId = 11 
    if flavor=='dimuon': pdgId = 13 
    gen_leptonlist = []
    genparticles = reader.genParticles()
    for genparticle in genparticles:
        if abs(genparticle['pdgId']) == pdgId:
		if genparticle['status'] == 1:
			genparticle['charge']=genparticle['pdgId']/(-pdgId)
                	gen_leptonlist.append(genparticle)  

    if len(gen_leptonlist)>1:
    	inds =range(len(gen_leptonlist))
	vecs = [ ROOT.TLorentzVector() for i in inds ]                                                    
	for i, v in enumerate(vecs):
		v.SetPtEtaPhiM(gen_leptonlist[i]['pt'], gen_leptonlist[i]['eta'], gen_leptonlist[i]['phi'], 0,)
	# all index combinations of OC particles
	dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if gen_leptonlist[comb[0]]['charge']*gen_leptonlist[comb[1]]['charge'] < 0 ]
	# sort the candidates, by mass of dilepton system
	dlMasses = sorted(dlMasses, key=lambda (m,i1,i2): -m)
        # sort indizes for pt
        for k,dlMass in enumerate(dlMasses): 
		if gen_leptonlist[dlMass[1]]['pt'] < gen_leptonlist[dlMass[2]]['pt']:
	        	dlMasses[k] = (dlMass[0],dlMass[2],dlMass[1],)
	gen_leading_index = dlMasses[0][1]
	gen_subleading_index = dlMasses[0][2]

        if event.nLep == 2 and leptonlist[0]['charge']*leptonlist[1]['charge']<0:
        	for j, dlMass in enumerate(dlMasses): 
			logger_rt.debug( 'gen_Dileptons (dlM, pt, pt) ' +str(j+1)+'/'+str(len(dlMasses))+ ': '+str(dlMass[0])+', '+str(gen_leptonlist[dlMass[1]]['pt'])+', '+str(gen_leptonlist[dlMass[2]]['pt']))
		logger_rt.debug( '*********************************************************************')
	# do it again, to have same synthax as for reco
        l1 = ROOT.TLorentzVector()
        # last argument is mass
        l1.SetPtEtaPhiM(gen_leptonlist[gen_leading_index]['pt'], gen_leptonlist[gen_leading_index]['eta'], gen_leptonlist[gen_leading_index]['phi'], 0)
        l2 = ROOT.TLorentzVector()
        l2.SetPtEtaPhiM(gen_leptonlist[gen_subleading_index]['pt'], gen_leptonlist[gen_subleading_index]['eta'], gen_leptonlist[gen_subleading_index]['phi'], 0)
        dl = l1 + l2

        event.gen_dl_mass   = dl.M()
        event.gen_dl_pt     = dl.Pt() 
        event.gen_dl_phi    = dl.Phi() 
        event.gen_dl_eta    = dl.Eta() 
        event.gen_dPhi_ll   = deltaPhi( gen_leptonlist[gen_leading_index]['phi'], gen_leptonlist[gen_subleading_index]['phi']) 

    else:
        event.gen_dl_mass   = -1.
        event.gen_dl_pt     = -1. 
        event.gen_dl_phi    = -10.
        event.gen_dl_eta    = -10. 
        event.gen_dPhi_ll   = -10.


    return

#
# postprocess
#

for rootfiles, outfile in zip([rootfiles_lo,rootfiles_mi,rootfiles_hi],[outfile_lo,outfile_mi,outfile_hi]):
	maker  = TreeMaker( sequence = [filler], variables = new_variables, treeName ="Events" )
	maker.start()
	if rootfiles:
		if args.small: rootfiles=[rootfiles[0]]
		for rootfile in rootfiles:
			logger_rt.info("Working on: " + rootfile)
			reader = DelphesReader( Sample.fromFiles( '', rootfile, treeName = "Delphes" ) )
			if args.small: reader.setEventRange( (0,1000) )
			reader.start()
			# Loop over events
			while reader.run():
			    maker.run()

		outputfile = ROOT.TFile.Open( outfile , 'recreate')
		outputfile.cd()
		maker.tree.Write()
		outputfile.Close()
		# Destroy the TTree
		maker.clear()

logger_rt.info("Success!")
