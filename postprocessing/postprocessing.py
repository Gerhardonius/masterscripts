'''
TO DO:  - scan over model dir and banner files is very slow
        - weights need to be implemented

'''
import ROOT
import argparse
import sys
import os
import itertools
import logging
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
argParser.add_argument('--subdir',type=str, default='vtest', help='Subdirectory of sampledir')
argParser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
argParser.add_argument('--small',    action='store_true' ,   help='work with 1000 events')
args = argParser.parse_args()

#
# logger
#
logger_rt = logger_rt.get_logger(args.logLevel, logFile = None)

#
# directories
#
maddir = os.path.join( sampledir, args.subdir )

outdir = os.path.join( flattreedir, args.subdir)
if not os.path.exists(outdir):
    os.makedirs(outdir) 

# This is way too slow!
#logger_rt.info( 'Create model list' )
#modeldict = createModeldict( maddir ) #{modelname:[rootfile1, rootfile2,...]}, modelnames:  LL02_2000_dimu or SMLO_0000_diel
#if args.small:
#    modeldict = {modeldict.keys()[0]:modeldict[modeldict.keys()[0]]}
#logger_rt.info('Models and rootfiles:')
#for key in modeldict.keys():
#    logger_rt.info(key, [ a.strip( sampledir ) for a in modeldict[key] ])

modeldict = {'SMtest':  ['/mnt/hephy/pheno/gerhard/Madresults/v3/dielec/SM/SM_DYjets_dielec_v3/Events/run_01/tag_1_delphes_events.root',\
                        '/mnt/hephy/pheno/gerhard/Madresults/v3/dielec/SM/SM_DYjets_dielec_v3/Events/run_02/tag_1_delphes_events.root'] }

#
# Tree Maker
#

# better synthax: [ TreeVariable.fromString('Jet[pt/F,eta/F,phi/F]' ) ] \
variables_strings = ['l1_pt/F', 'l1_phi/F', 'l1_eta/F', #leading lepton1
                     'l2_pt/F', 'l2_phi/F', 'l2_eta/F', #leading lepton2
                    'nLep/I',
                    'met_pt/F','met_phi/F', #MET 
                    'ht/F', 'nJet/I', # hadronic activity
                    'dl_mass/F', 'dl_pt/F', 'dl_phi/F', 'dl_eta/F', 'dPhi_ll/F', #dilepton system
                    ]

#Examples for Gen variables: nGenJet/I, GenMet_pt/F, GenMet_phi/F, GenLep_pt/F, GenLep_pt[1]/F
new_variables = [ TreeVariable.fromString(x) for x in variables_strings ]

# Filler
# maybe here 2 fillers: electron and muon
def filler(event):
    '''
    Example from roottools: event.MyJet_pt2[i] = reader.event.Jet_pt[i]**2
    my example:             event.met_pt = reader.myelec()
    or directly:            event.l1_pt = reader.event.Electron_PT[0]
    Caution: even though Collection_size = x, Collection_PT[y] gives an unphys. value for y>x
    thats the reason for read_collection
    '''
    #
    # 1:1 translation
    #

    # leptons
    event.nLep = len( reader.electrons() )
    for recoelec, rank in zip(reader.electrons(), ['l1','l2']):
        for var in ['pt','eta','phi']:
            setattr( event, rank+'_'+var, recoelec[var])     

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

    #
    # new branches
    #

    # dilepton system
    # very simple implementation for now
    leptonlist = reader.electrons() 
    if event.nLep == 2:
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

    return

#
# loop over models
#
for modelname in modeldict.keys():
    
    logger_rt.info("Working on: " + modelname)
    # selection on electons/muons
    if 'diel' in modelname:
	flavor = 'electrons'
    else:
	flavor = 'muons'

    delphes_files = modeldict[modelname]
    reader = DelphesReader( Sample.fromFiles( delphes_files[0], delphes_files, treeName = "Delphes" ) )
    if args.small:
        reader.setEventRange( (0,1000) )
    maker  = TreeMaker( sequence = [filler], variables = new_variables, treeName ="Events" )

    reader.start()
    maker.start()
    # Loop over events
    while reader.run():
        maker.run()

    outputfile = ROOT.TFile.Open( os.path.join( outdir, modelname +'.root') , 'recreate')
    maker.tree.Write()
    outputfile.Close()
    # Destroy the TTree
    maker.clear()

#logger.info("Success!")

#    # rec stuff
#    Jet.PT
#    Jet.Eta
#    Jet.Phi
#    Jet.Mass
#    Jet.T
#    Muon/Electron/Photon
#    Muon.PT
#    Muon.Eta
#    Muon.Phi
#    Muon.T
#    FatJet.PT
#    FatJet.Eta
#    FatJet.Phi
#    FatJet.T
#    FatJet.Mass
#    FatJet.DeltaEta
#    FatJet.DeltaPhi
#    MissingET.MET
#    MissingET.Eta
#    MissingET.Phi
#    ScalarHT.HT
#    # gen stuff
#    GenJet
#    GenMissingET
#    Particle
