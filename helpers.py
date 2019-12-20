#Standard imports
import ROOT
import os
import itertools
import math

# Logging
import logging
logger = logging.getLogger(__name__)

def bannerfiletoModelName( bannerfilepath ):
    ''' takes the path of the bannerfile and extracts the modelinformation
        returns: LL0_2000_dimu or SM0_0000_diel
    '''
    bannerfile = open( bannerfilepath )
    modelname = ''
    #initialize some vals
    mzpval=666
    gal1x1=10
    gvl1x1=10
    gal2x2=10
    gvl2x2=10
    gal3x3=10
    gvl3x3=10
 
    # check UFO model
    # get values from banner file
    for line in bannerfile:
        line = line.strip()
        if 'generate p p' in line:
            if 'mu' in line.split()[-1]:
                flavor = 'dimu'
            if 'e' in line.split()[-1]:
                flavor = 'diel'
        if 'mzp' in line:
            if '9000001' in line:
                mzpval = int(float((line.lstrip('9000001').strip().split('#')[0])))
        # electron
        if 'gal1x1' in line:
            gal1x1 =float(line.lstrip('1').strip().lstrip('1').strip().split('#')[0])
        if 'gvl1x1' in line:
            gvl1x1 =float(line.lstrip('1').strip().lstrip('1').strip().split('#')[0])
        # muons
        if 'gal2x2' in line:
            gal2x2 =float(line.lstrip('2').strip().lstrip('2').strip().split('#')[0])
        if 'gvl2x2' in line:
            gvl2x2 =float(line.lstrip('2').strip().lstrip('2').strip().split('#')[0])
        # tau
        if 'gal3x3' in line:
            gal3x3 =float(line.lstrip('3').strip().lstrip('3').strip().split('#')[0])
        if 'gvl3x3' in line:
            gvl3x3 =float(line.lstrip('3').strip().lstrip('3').strip().split('#')[0])
    # determine model from values
    # electron model letter
    if gvl1x1 == -gal1x1:
        modelname += 'L'
    elif gvl1x1 == gal1x1:
        modelname += 'R'
    elif gal1x1 == 0:
        modelname += 'V'
    else:
        modelname += 'X'
    # muon model letter
    if gvl2x2 == -gal2x2:
        modelname += 'L'
    elif gvl2x2 == gal2x2:
        modelname += 'R'
    elif gal2x2 == 0:
        modelname += 'V'
    else:
        modelname += 'X'
    # tau model letter
    if gvl3x3 == 0 and gal3x3 == 0:
        modelname += '0'
    else: 
        modelname += 'X'
    # ratio
    ratio10 = gvl1x1/gvl2x2 *10
    if ratio10 in [1,3,5]:
        modelname += str(int(ratio10))
    else:
        modelname += 'X'
    modelname = modelname + '_' + str(mzpval).zfill(4) + '_' + flavor
    if 'RRX_0666' in modelname:
	modelname = 'SM0_0000_' + flavor
    return modelname

def createModellist2( maddir ):
    models = []
    masslist = []
    # Modellist for maddir
    for root, dirs, files in os.walk( os.path.join( maddir, 'Events')):
        if root.split('/')[-1].startswith('run_'):
            for filename in files:
                if filename.endswith('banner.txt'):
                    bannerfilepath = os.path.join( root,filename)
                    modelname, mass = createModelName( bannerfilepath )
                if filename.endswith('.root'):
                    rootfilepath = os.path.join( root,filename)
            models.append( (modelname, mass, rootfilepath) )
            if mass not in masslist:
                masslist.append(mass)
    # get flavor from last modelname
    if 'dimu' in modelname:
        flavor = 'dimu'
    if 'diel' in modelname:
        flavor = 'diel'

    # Modellist for SMindir, based on BSMindir (we need the masses from there)
    if SMindir:
        models = []
        for root, dirs, files in os.walk( os.path.join( SMindir, 'Events')):
            if root.split('/')[-1].startswith('run_'):
                for filename in files:
                    if root.split('/')[-1].startswith('run_'):
                        for filename in files:
                            if filename.endswith('.root'):
                                rootfilepath =  os.path.join( root,filename) 
        for mass in masslist:
            models.append( ('SMLO_' + str(mass).zfill(4) + '_' + flavor, mass, rootfilepath )) 

    models = sorted(models, key=lambda (modelname,mass,rootfilepath): modelname)
    return models


def setdihistoprop( hist1, hist2, xlabel = 'm_{ll}/[GeV]'):
    # Disable Stats box
    hist2.SetStats(0)
    # Set the line color to red for gen level and black for reconstruction level
    hist1.SetLineColor(ROOT.kBlack)
    hist2.SetLineColor(ROOT.kRed)
    # Set line style 
    hist1.SetLineStyle(1)
    hist2.SetLineStyle(1)
    # Set the line width to 2
    hist1.SetLineWidth(2)
    hist2.SetLineWidth(2)
    # Set axis labels
    hist1.GetXaxis().SetTitle( xlabel )
    hist1.GetYaxis().SetTitle("Number of events")
    return

#def drawdihisto( canvas, hist1, hist2, hist1label, hist2label, plotFileName, log=True, lumi=137):
#    #
#    # Draw the normal plots (not the ratio)
#    #
#    pad1 = ROOT.TPad("pad1","pad1",0,0.3,1,1)
#    pad1.SetLogy(True)                      # Set the y-axis of the top plot to be logarithmic
#    pad1.SetBottomMargin(0)                 # Upper and lower pads are joined
#    pad1.Draw()                             # Draw the upper pad in the canvas
#    pad1.cd()                               # pad1 becomes the current pad
#    # Draw histos
#    hist1.SetTitle('')                      # Remove the plot title
#    hist1.GetXaxis().SetLabelSize(0)        # Remove x-axis labels for the top pad
#    hist1.GetXaxis().SetTitleSize(0)        # Remove x-axis title for the top pad
#    hist1.GetYaxis().SetTitleSize(0.05)     # Increase y-axis title size (pad is not full page)
#    hist1.Draw('h')
#    hist2.Draw('pe,sames')
#    # Add a legend to the top pad
#    legend = ROOT.TLegend(0.7,0.6,0.85,0.75)    # Add a legend near the top right corner
#    legend.AddEntry(hist1, hist1label)
#    legend.AddEntry(hist2, hist2label)
#    legend.SetLineWidth(0)                      # Remove the boundary on the legend
#    legend.Draw("same")                         # Draw the legend on the plot
#    # Lumi text
#    Lumitex = ROOT.TLatex()
#    Lumitex.SetNDC()
#    Lumitex.SetTextSize(0.04)
#    Lumitex.DrawLatex( 0.15,0.95, 'L=%3.1f fb{}^{-1} (13 TeV)'% ( lumi ) ) 
#
#    #
#    # Draw the ratio
#    #
#    # First clone the current data points
#    histratio = hist1.Clone()
#    # Divide histos1 by histos2
#    histratio.Divide(hist2)
#    histratio.SetLineColor(ROOT.kRed)
#
#    # Now draw the ratio
#    canvas.cd()                             # Go back to the main canvas before defining pad2
#    pad2 = ROOT.TPad("pad2","pad2",0,0.05,1,0.3)
#    pad2.SetTopMargin(0)                    # Upper and lower pads are joined
#    pad2.SetBottomMargin(0.25)              # Expand the bottom margin for extra label space
#    pad2.Draw()                             # Draw the lower pad in the canvas
#    pad2.cd()                               # pad2 becomes the current pad
#    histratio.SetStats(0)
#    histratio.SetTitle("")                      # Turn off the title to avoid overlap
#    histratio.GetXaxis().SetLabelSize(0.12)     # Larger x-axis labels (pad is not full page)
#    histratio.GetXaxis().SetTitleSize(0.12)     # Larger x-axis title (pad is not full page)
#    histratio.GetYaxis().SetLabelSize(0.1)      # Larger y-axis labels (pad is not full page)
#    histratio.GetYaxis().SetTitleSize(0.15)     # Larger y-axis title (pad is not full page)
#    histratio.GetYaxis().SetTitle(hist1label+"/"+hist2label)    # Change the y-axis title (this is the ratio)
#    histratio.GetYaxis().SetTitleOffset(0.3)    # Reduce the y-axis title spacing
#    histratio.GetYaxis().SetRangeUser(0.5,1.5)  # Set the y-axis ratio range from 0.5 to 1.5
#    histratio.GetYaxis().SetNdivisions(207)     # Change the y-axis tick-marks to work better
#    histratio.Draw("pe")                        # Draw the ratio in the current pad
#    
#    ## Add a line at 1 to the ratio plot
#    #line = ROOT.TLine(50.e3,1,200.e3,1) # Draw a line at 1 from 50 GeV to 200 GeV (full plot)
#    #line.SetLineColor(ROOT.kBlack)      # Set the line colour to black
#    #line.SetLineWidth(2)                # Set the line width to 2
#    #line.Draw("same")                   # Draw the line on the same plot as the ratio
#
#    canvas.SaveAs( plotFileName.split('.')[0] + '.png' )
#    canvas.Close( )
#    return

def createModellist( BSMindir, SMindir=False):
    models = []
    masslist = []
    # Modellist for BSMindir
    for root, dirs, files in os.walk( os.path.join( BSMindir, 'Events')):
        if root.split('/')[-1].startswith('run_'):
            for filename in files:
                if filename.endswith('banner.txt'):
                    bannerfilepath = os.path.join( root,filename)
                    modelname, mass = createModelName( bannerfilepath )
                if filename.endswith('.root'):
                    rootfilepath = os.path.join( root,filename)
            models.append( (modelname, mass, rootfilepath) )
            if mass not in masslist:
                masslist.append(mass)
    # get flavor from last modelname
    if 'dimu' in modelname:
        flavor = 'dimu'
    if 'diel' in modelname:
        flavor = 'diel'

    # Modellist for SMindir, based on BSMindir (we need the masses from there)
    if SMindir:
        models = []
        for root, dirs, files in os.walk( os.path.join( SMindir, 'Events')):
            if root.split('/')[-1].startswith('run_'):
                for filename in files:
                    if root.split('/')[-1].startswith('run_'):
                        for filename in files:
                            if filename.endswith('.root'):
                                rootfilepath =  os.path.join( root,filename) 
        for mass in masslist:
            models.append( ('SMLO_' + str(mass).zfill(4) + '_' + flavor, mass, rootfilepath )) 

    models = sorted(models, key=lambda (modelname,mass,rootfilepath): modelname)
    return models


def labelhistoaxis(histo, xlabel, ylabel):
    histo.GetYaxis().SetTitle( ylabel )
    histo.GetXaxis().SetTitle( xlabel )
    histo.SetLineWidth(2)

def createModelName( bannerfilepath ):
    ''' takes the path of the bannerfile and extracts the modelinformation
        note that import model statement in the bannerfile is crucial!
        returns: LL0_2000 or SMNLO
    '''
    bannerfile = open( bannerfilepath )
    modelname = ''
    # check UFO model
    # get values from banner file
    for line in bannerfile:
        line = line.strip()
        if 'generate p p' in line:
            if 'mu' in line.split()[-1]:
                flavor = 'dimu'
            if 'e' in line.split()[-1]:
                flavor = 'diel'
        if 'mzp' in line:
            if '9000001' in line:
                mzpval = int(float((line.lstrip('9000001').strip().split('#')[0])))
        # electron
        if 'gal1x1' in line:
            gal1x1 =float(line.lstrip('1').strip().lstrip('1').strip().split('#')[0])
        if 'gvl1x1' in line:
            gvl1x1 =float(line.lstrip('1').strip().lstrip('1').strip().split('#')[0])
        # muons
        if 'gal2x2' in line:
            gal2x2 =float(line.lstrip('2').strip().lstrip('2').strip().split('#')[0])
        if 'gvl2x2' in line:
            gvl2x2 =float(line.lstrip('2').strip().lstrip('2').strip().split('#')[0])
        # tau
        if 'gal3x3' in line:
            gal3x3 =float(line.lstrip('3').strip().lstrip('3').strip().split('#')[0])
        if 'gvl3x3' in line:
            gvl3x3 =float(line.lstrip('3').strip().lstrip('3').strip().split('#')[0])
    # determine model from values
    # electron model letter
    if gvl1x1 == -gal1x1:
        modelname += 'L'
    elif gvl1x1 == gal1x1:
        modelname += 'R'
    elif gal1x1 == 0:
        modelname += 'V'
    else:
        modelname += 'X'
    # muon model letter
    if gvl2x2 == -gal2x2:
        modelname += 'L'
    elif gvl2x2 == gal2x2:
        modelname += 'R'
    elif gal2x2 == 0:
        modelname += 'V'
    else:
        modelname += 'X'
    # tau model letter
    if gvl3x3 == 0 and gal3x3 == 0:
        modelname += '0'
    else: 
        modelname += 'X'
    # ratio
    ratio10 = gvl1x1/gvl2x2 *10
    if ratio10 in [1,3,5]:
        modelname += str(int(ratio10))
    else:
        modelname += 'X'
    modelname = modelname + '_' + str(mzpval).zfill(4) + '_' + flavor
    return (modelname, mzpval)

# takes list, dictonary with pt, eta, phi, charge keys
def includeOUflowbins( hist ):
    # counts of uf and of bins
    nbins = hist.GetNbinsX()
    cnt_uf= hist.GetBinContent(0)
    err_uf= hist.GetBinError(0)
    cnt_of= hist.GetBinContent(nbins+1)
    err_of= hist.GetBinError(nbins+1)

    # set bin content of first and last bin to include overflows
    hist.SetBinContent(      1, hist.GetBinContent(1) + cnt_uf )
    hist.SetBinError(        1, math.sqrt( err_uf**2 +  hist.GetBinContent(1)**2))
    hist.SetBinContent(nbins  ,  hist.GetBinContent(nbins) + cnt_of )
    hist.SetBinError(        1, math.sqrt( err_of**2 + hist.GetBinContent(nbins)**2))
    hist.SetBinContent(      0, 0)
    hist.SetBinError(        0, 0) 
    hist.SetBinContent(nbins+1, 0)
    hist.SetBinError(  nbins+1, 0) 

def getSortedZCandidates_negfirst(leptons, targetmass=91.1876):
    ''' needs list of leptons, each lepton is a dict with pt, eta, phi, E or M, charge keys.
        returns:    list of tuples (mll, neg charged lep dict, pos charged lep dic) sorted via mll closest 
                    to targetmass. If no targetmass is given (in GeV), mZ is assumed
    '''
    inds = range(len(leptons))
    vecs = [ ROOT.TLorentzVector() for i in inds ]
    if leptons:
        if 'E' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiE(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['E'],)
        if 'M' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['M'],)
    # all index combinations of OC particles
    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if leptons[comb[0]]['charge']*leptons[comb[1]]['charge'] < 0 ]
    # sort the candidates, only keep the best ones
    dlMasses = sorted(dlMasses, key=lambda (m,i1,i2):abs(m-targetmass))
    usedIndices = []
    bestCandidates = []
    for m in dlMasses:
        if m[1] not in usedIndices and m[2] not in usedIndices:
            usedIndices += m[1:3]
            if leptons[m[1]]['charge'] < 0:
                bestCandidates.append( (m[0], leptons[m[1]], leptons[m[2]]) )
            else:
                bestCandidates.append( (m[0], leptons[m[2]], leptons[m[1]]) )
    return bestCandidates

def getSortedZCandidates(leptons, targetmass=91.1876):
    ''' needs list of leptons, each lepton is a dict with pt, eta, phi, E or M, charge keys.
        returns:    list of tuples (mll, leading_pt, subleading_pt) sorted via mll closest 
                    to targetmass. If no targetmass is given (in GeV), mZ is assumed
    '''
    inds = range(len(leptons))
    vecs = [ ROOT.TLorentzVector() for i in inds ]
    if leptons:
        if 'E' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiE(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['E'],)
        if 'M' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['M'],)
    # all index combinations of OC particles
    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if leptons[comb[0]]['charge']*leptons[comb[1]]['charge'] < 0 ]
    # sort the candidates, only keep the best ones
    dlMasses = sorted(dlMasses, key=lambda (m,i1,i2):abs(m-targetmass))
    usedIndices = []
    bestCandidates = []
    for m in dlMasses:
        if m[1] not in usedIndices and m[2] not in usedIndices:
            usedIndices += m[1:3]
            if leptons[m[1]]['pt'] > leptons[m[2]]['pt']:
                bestCandidates.append( (m[0], leptons[m[1]]['pt'], leptons[m[2]]['pt']) )
            else:
                bestCandidates.append( (m[0], leptons[m[2]]['pt'], leptons[m[1]]['pt']) )
    return bestCandidates

def getmaxsumPtZCandidates(leptons):
    ''' needs list of leptons, each lepton is a dict with pt, eta, phi, E or M, charge keys.
        returns:    list of tuples (mll, leading_pt, subleading_pt) sorted via sum pt`s

    '''
    inds = range(len(leptons))
    vecs = [ ROOT.TLorentzVector() for i in inds ]
    if leptons:
        if 'E' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiE(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['E'],)
        if 'M' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['M'],)
    # all index combinations of OC particles
    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if leptons[comb[0]]['charge']*leptons[comb[1]]['charge'] < 0 ]
    # sort the candidates, only keep the best ones
    dlMasses = sorted(dlMasses, key=lambda (m,i1,i2): leptons[i1]['pt'] + leptons[i2]['pt'] )
    usedIndices = []
    bestCandidates = []
    for m in dlMasses:
        if m[1] not in usedIndices and m[2] not in usedIndices:
            usedIndices += m[1:3]
            if leptons[m[1]]['pt'] > leptons[m[2]]['pt']:
                bestCandidates.append( (m[0], leptons[m[1]]['pt'], leptons[m[2]]['pt']) )
            else:
                bestCandidates.append( (m[0], leptons[m[2]]['pt'], leptons[m[1]]['pt']) )
    return bestCandidates

def getmaxPtZCandidates(leptons):
    ''' needs list of leptons, each lepton is a dict with pt, eta, phi, E or M, charge keys.
        returns:    list of tuples (mll, leading_pt, subleading_pt) sorted via 
                    highest and second highest pt in event.
    '''
    inds = range(len(leptons))
    vecs = [ ROOT.TLorentzVector() for i in inds ]
    if leptons:
        if 'E' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiE(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['E'],)
        if 'M' in leptons[0].keys():
            for i, v in enumerate(vecs):
                v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], leptons[i]['M'],)
    # all index combinations of OC particles
    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if leptons[comb[0]]['charge']*leptons[comb[1]]['charge'] < 0 ]
    # sort the candidates, only keep the best ones
    dlMasses = sorted(dlMasses, key=lambda (m,i1,i2): max(leptons[i1]['pt'],leptons[i2]['pt']) )
    usedIndices = []
    bestCandidates = []
    for m in dlMasses:
        if m[1] not in usedIndices and m[2] not in usedIndices:
            usedIndices += m[1:3]
            if leptons[m[1]]['pt'] > leptons[m[2]]['pt']:
                bestCandidates.append( (m[0], leptons[m[1]]['pt'], leptons[m[2]]['pt']) )
            else:
                bestCandidates.append( (m[0], leptons[m[2]]['pt'], leptons[m[1]]['pt']) )
    return bestCandidates

# import histos?
def getFileList(dir, histname='histo', maxN=-1):
    filelist = os.listdir(os.path.expanduser(dir))
    filelist = [dir+'/'+f for f in filelist if histname in f and f.endswith(".root")]
    if maxN>=0:
        filelist = filelist[:maxN]
    return filelist
#
#def getMinDLMass(leptons):
#    inds = range(len(leptons))
#    vecs = [ ROOT.TLorentzVector() for i in inds ]
#    for i, v in enumerate(vecs):
#        v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], 0.)
#    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) ]
#    return min(dlMasses), dlMasses
#
## Returns (closest mass, index1, index2)
#def closestOSDLMassToMZ(leptons):
#    inds = [i for i in range(len(leptons))]
#    vecs = [ROOT.TLorentzVector() for i in range(len(leptons))]
#    for i, v in enumerate(vecs):
#        v.SetPtEtaPhiM(leptons[i]['pt'], leptons[i]['eta'], leptons[i]['phi'], 0.)
#    dlMasses = [((vecs[comb[0]] + vecs[comb[1]]).M(), comb[0], comb[1])  for comb in itertools.combinations(inds, 2) if leptons[comb[0]]['pdgId']*leptons[comb[1]]['pdgId'] < 0 and abs(leptons[comb[0]]['pdgId']) == abs(leptons[comb[1]]['pdgId']) ]
#    return min(dlMasses, key=lambda (m,i1,i2):abs(m-mZ)) if len(dlMasses)>0 else (float('nan'), -1, -1)
#
#def getChain(sampleList, histname='', maxN=-1, treeName="Events"):
#    if not type(sampleList)==type([]):
#        sampleList_ = [sampleList]
#    else:
#        sampleList_= sampleList
#    c = ROOT.TChain(treeName)
#    i=0
#    for s in sampleList_:
#        if type(s)==type(""):
#            for f in getFileList(s, histname, maxN):
#                if checkRootFile(f, checkForObjects=[treeName]):
#                    i+=1
#                    c.Add(f)
#                else:
#                    print "File %s looks broken."%f
#            print "Added ",i,'files from samples %s' %(", ".join([s['name'] for s in sampleList_]))
#        elif type(s)==type({}):
#            if s.has_key('file'):
#                c.Add(s['file'])
##        print "Added file %s"%s['file']
#                i+=1
#            if s.has_key('bins'):
#                for b in s['bins']:
#                    dir = s['dirname'] if s.has_key('dirname') else s['dir']
#                    for f in getFileList(dir+'/'+b, histname, maxN):
#                        if checkRootFile(f, checkForObjects=[treeName]):
#                            i+=1
#                            c.Add(f)
#                        else:
#                            print "File %s looks broken."%f
##      print 'Added %i files from %i elements' %(i, len(sampleList))
#        else:
##      print sampleList
#            print "Could not load chain from sampleList %s"%repr(sampleList)
#    return c
#
#def checkRootFile(f, checkForObjects=[]):
#    rf = ROOT.TFile.Open(f)
#    if not rf: return False
#    try:
#        good = (not rf.IsZombie()) and (not rf.TestBit(ROOT.TFile.kRecovered))
#    except:
#        if rf: rf.Close()
#        return False
#    for o in checkForObjects:
#        if not rf.GetListOfKeys().Contains(o):
#            print "[checkRootFile] Failed to find object %s in file %s"%(o, f)
#            rf.Close()
#            return False
##    print "Keys recoveredd %i zombie %i tb %i"%(rf.Recover(), rf.IsZombie(), rf.TestBit(ROOT.TFile.kRecovered))
#    rf.Close()
#    return good
#
#def getMC(inputdir):
#    inputfile = []
#    for dirpath, dirnames, filenames in os.walk(inputdir):
#        for filename in [f for f in filenames if f.endswith(".root")]:
#            inputfile.append( os.path.join(dirpath, filename) )
#
#    chain = ROOT.TChain("Delphes")
#    
#    for f in inputfile:
#       print 'file is ', f
#       chain.Add(f)
#    
#    nrevents = chain.GetEntries()
#    print('Number of events: ',  nrevents)
#    # return object associated with the chain
#    return ROOT.Delphes(chain)
#
