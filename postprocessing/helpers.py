#Standard imports
import os
from math import pi

# Logging
import logging
logger = logging.getLogger(__name__)

def deltaPhi(phi1, phi2):
    dphi = phi2-phi1
    if  dphi > pi:
        dphi -= 2.0*pi
    if dphi <= -pi:
        dphi += 2.0*pi
    return abs(dphi)

def bannerfiletoModelName( bannerfilepath ):
    ''' takes the path of the bannerfile and extracts the modelinformation
        returns: LL02_2000_dimu or SMLO_0000_diel
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
            if 'mu' in line:
                flavor = 'dimu'
            if 'e' in line:
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
    print bannerfilepath
    modelname = modelname + '_' + str(mzpval).zfill(4) + '_' + flavor
    if 'RRXX_0666' in modelname:
	modelname = 'SMLO_0000_' + flavor
    return modelname

def createModeldict( maddir ):
    ''' looks for structure Events/run_XXX recursevly in maddir.
        from these dirs files are organized in dict
            XXXXbanner.txt  -> modelname
            XXXXXXXXXX.root -> rootfile 
        {modelname:[rootfile1, rootfile2,...]}
    '''
    modeldict = {}
    # Modellist for maddir
    for root, dirs, files in os.walk( maddir):
        # find Events/run_ subdirectories
        if root.split('/')[-2].startswith('Events'):
            if root.split('/')[-1].startswith('run_'):
                # loop over files in this subdirectories
                for filename in files:
                    if filename.endswith('banner.txt'):
                        bannerfilepath = os.path.join( root,filename)
                        modelname = bannerfiletoModelName( bannerfilepath )
                    if filename.endswith('.root'):
                        rootfilepath = os.path.join( root,filename)
                if modelname not in modeldict.keys():
                    modeldict[modelname]= [rootfilepath]
                else:
                    modeldict[modelname].append(rootfilepath)
    return modeldict
