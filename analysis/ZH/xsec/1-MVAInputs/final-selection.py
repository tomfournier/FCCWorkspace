##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Analysis configuration and paths
from package.userConfig import (
    loc, get_loc, get_params,
    frac, nb
)

cat, ecm, lumi = get_params(os.environ.copy(), '1-run.json', is_final=True)
if cat not in ['ee', 'mumu']:
    raise ValueError(f'Invalid channel: {cat}. Must be "ee" or "mumu"')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Input directory for pre-selection outputs
inputDir  = get_loc(loc.EVENTS_TRAINING, cat, ecm, '')

# Output directory for final-selection histograms
outputDir = get_loc(loc.HIST_MVA,   cat, ecm, '')

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

# If procDict is incomplete, can use procDictAdd to add information on the missing samples

# Parallel processing configuration (default nCPUS=4)
nCPUS = 10

# Produce ROOT TTrees in addition to histograms (default is False)
doTree = True

# Scale yields to integrated luminosity
doScale = True
intLumi = lumi * 1e6 # in pb-1



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Process samples for BDT (signal + backgrounds)
samples_BDT = [
    # Signal: ZH production
    f'wzp6_ee_{cat}H_ecm{ecm}',

    # Main backgrounds: diboson and Z+jets
    f'p8_ee_ZZ_ecm{ecm}', 
    f'p8_ee_WW_{cat}_ecm{ecm}', 
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee' else f'wzp6_ee_mumu_ecm{ecm}',

    # Rare backgrounds: radiative and diphton
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}', 
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gaga_{cat}_60_ecm{ecm}'
]

# Process list with per-sample parameters (from `param`)
processList = {i:{'fraction': frac, 'chunks': nb} for i in samples_BDT}



#######################
### DEFINE CUT LIST ###
#######################

# Define range for CoM dependent variables
p_up = 70 if ecm==240 else (150 if ecm==365 else 240)
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)

# Define baseline selection cuts
m_cut, p_cut = 'zll_m > 86 && zll_m < 96', f'zll_p > {p_dw} && zll_p < {p_up}'
rec_cut = ' && zll_recoil_m > 100 && zll_recoil_m < 150' if ecm==365 else ''
Baseline_Cut = m_cut + ' && ' + p_cut + rec_cut

# Selection cuts dictionary (key = selection name used in outputs)
cutList = { 
    # 'sel0':     'return true;',
    'Baseline': Baseline_Cut
}



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Output histogram definitions (name, title, binning)
histoList = {

    # Lepton kinematics: leading lepton
    'leading_p':        {'name':'leading_p',
                         'title':'p_{l,leading} [GeV]',
                         'bin':400,'xmin':0,'xmax':200},

    'leading_pT':       {'name':'leading_pT',
                         'title':'p_{T,l,leading} [GeV]',
                         'bin':400,'xmin':0,'xmax':200},

    'leading_theta':    {'name':'leading_theta',
                         'title':'#theta_{l,leading}',
                         'bin':128, 'xmin':0,  'xmax':3.2},

    'leading_phi':      {'name':'leading_phi',
                         'title':'#phi_{l,leading}',
                         'bin':64,'xmin':-3.2,'xmax':3.2},

    # Lepton kinematics: subleading lepton
    'subleading_p':     {'name':'subleading_p',
                         'title':'p_{l,subleading} [GeV]',
                         'bin':400,'xmin':0,'xmax':200},

    'subleading_pT':    {'name':'subleading_pT',
                         'title':'p_{T,l,subleading} [GeV]',
                         'bin':400,'xmin':0,'xmax':200},

    'subleading_theta': {'name':'subleading_theta',
                         'title':'#theta_{l,subleading}',
                         'bin':128, 'xmin':0,  'xmax':3.2},
    
    'subleading_phi':   {'name':'subleading_phi',
                         'title':'#phi_{l,subleading}',
                         'bin':64,'xmin':-3.2,'xmax':3.2},

    # Angular separation between leptons
    'acolinearity':     {'name':'acolinearity',
                         'title':'#Delta#theta_{l^{+}l^{-}}',
                         'bin':120,'xmin':0,'xmax':3},

    'acoplanarity':     {'name':'acoplanarity',
                         'title':'#Delta#phi_{l^{+}l^{-}}',
                         'bin':128,'xmin':0,'xmax':3.2},
    
    'deltaR':           {'name':'deltaR',
                         'title':'#DeltaR',
                         'bin':100,'xmin':1,'xmax':7},
    
    # Z boson properties
    'zll_m':            {'name':'zll_m',
                         'title':'m_{l^{+}l^{-}} [GeV]',
                         'bin':100,'xmin':86,'xmax':96},

    'zll_p':            {'name':'zll_p',
                         'title':'p_{l^{+}l^{-}} [GeV]',
                         'bin':500,'xmin':0,'xmax':250},

    'zll_pT':           {'name':'zll_pT',
                         'title':'p_{T,l^{+}l^{-}} [GeV]',
                         'bin':500,'xmin':0,'xmax':250},

    'zll_theta':        {'name':'zll_theta',
                         'title':'#theta_{l^{+}l^{-}}',
                         'bin':128,'xmin':0,'xmax':3.2},

    'zll_phi':          {'name':'zll_phi',
                         'title':'#phi_{l^{+}l^{-}}',
                         'bin':64,'xmin':-3.2,'xmax':3.2},
    
    # Recoil mass (Higgs candidate)
    'zll_recoil_m':     {'name':'zll_recoil_m',
                         'title':'m_{recoil} [GeV]',
                         'bin':100,'xmin':100,'xmax':150},
    
    # Visible and invisible information
    'cosTheta_miss':    {'name':'cosTheta_miss',
                         'title':'|cos#theta_{miss}|',
                         'bin':1000,'xmin':0,'xmax':1},
    
    'visibleEnergy':    {'name':'visibleEnergy',
                         'title':'E_{vis} [GeV]',
                         'bin':700,'xmin':0,'xmax':350},

    'missingMass':      {'name':'missingMass',
                         'title':'m_{miss} [GeV]',
                         'bin':700,'xmin':0,'xmax':350},
    
    # Higgsstrahlungness
    'H':                {'name':'H',
                         'title':'Higgsstrahlungness [GeV^{2}]',
                         'bin':110,'xmin':0,'xmax':110},

}
