##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Analysis configuration and paths
from package.userConfig import (
    loc, get_loc, event, 
    ecm, lumi, frac, nb
)
from package.func.bdt import def_bdt, make_high_low
from package.config import (
    z_decays, 
    H_decays, 
    input_vars
)
# Select Z decay
cat = input('Select a channel [ee, mumu]: ')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Input directory for pre-selection outputs
inputDir  = get_loc(loc.EVENTS, cat, ecm, '')

# Output directory for final-selection histograms
outputDir = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# If procDict is incomplete, can use procDictAdd to add information on the missing samples

# Parallel processing configuration (default nCPUS=4)
nCPUS = 10

# Produces ROOT TTrees, default is False
# doTree = True

# Scale yields to integrated luminosity
doScale = True
intLumi = lumi * 1e6 # in pb-1



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Background samples:
samples_bkg = [
    # Diboson:  ee -> VV
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_ecm{ecm}',
    f'p8_ee_WW_ee_ecm{ecm}', 
    f'p8_ee_WW_mumu_ecm{ecm}',

    # ee -> Z+jets
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', 
    f'wzp6_ee_mumu_ecm{ecm}', 
    f'wzp6_ee_tautau_ecm{ecm}',

    # Radiative: ey -> eZ(ll)
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', 
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
    f'wzp6_gammae_eZ_Zee_ecm{ecm}',

    # Diphoton: yy -> ll
    f'wzp6_gaga_ee_60_ecm{ecm}', 
    f'wzp6_gaga_mumu_60_ecm{ecm}', 
    f'wzp6_gaga_tautau_60_ecm{ecm}', 
    
    # Invisible: ee -> nunuZ
    f'wzp6_ee_nuenueZ_ecm{ecm}'
]

# Signal samples: ee -> Z(ll)H with all Higgs decay modes
samples_sig = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in H_decays + ('ZZ_noInv',)]
samples_sig.extend((f'wzp6_ee_eeH_ecm{ecm}', f'wzp6_ee_mumuH_ecm{ecm}', f'wzp6_ee_ZH_Hinv_ecm{ecm}'))

# Load event samples with events TTree
samples = event(samples_sig + samples_bkg, inputDir)

# Large samples requiring chunked processing
big_sample = (
    f'p8_ee_ZZ_ecm{ecm}', 
    f'p8_ee_WW_ecm{ecm}', 
    f'p8_ee_WW_{cat}_ecm{ecm}',

    f'wzp6_ee_mumu_ecm{ecm}' if cat=='mumu' else f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',

    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}', 
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gaga_{cat}_60_ecm{ecm}'
)

# Configure processing parameters for each sample
processList = {i:{'fraction': frac, 'chunks': nb if i in big_sample else 1}  for i in samples}

# Define BDT score from trained model and apply BDT cut
sel_BDT = 'Baseline'
loc_BDT = get_loc(loc.BDT, cat, ecm, sel_BDT)
defineList, bdt_cut = def_bdt(input_vars, loc_BDT)



#######################
### DEFINE CUT LIST ###
#######################

# Define range for CoM dependent variables
p_up = 70 if ecm==240 else (150 if ecm==365 else 200)
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)

# Define baseline selection cuts
vis_cut = 100 if ecm==240 else (171 if ecm==365 else 0)
m_cut, p_cut = 'zll_m > 86 && zll_m < 96', f'zll_p > {p_dw} && zll_p < {p_up}'
Baseline_Cut = m_cut + ' && ' + p_cut
vis, inv = Baseline_Cut + f' && visibleEnergy > {vis_cut}', Baseline_Cut + f' && visibleEnergy < {vis_cut}'

# Selection cut dictionary (key = selection name used in outputs)
cutList = { 
    # 'sel0':              'return true;',
    # 'Baseline':          Baseline_Cut,
    # 'Baseline_vis':      vis,
    # 'Baseline_inv':      inv,
    # 'Baseline_miss':     Baseline_Cut + ' && cosTheta_miss < 0.98',
    'Baseline_sep':      '(('+vis+') || ('+inv+' && cosTheta_miss < 0.99))',
    'Baseline_sep1':     '(('+vis+') || ('+inv+' && cosTheta_miss < 0.995))',
    'Baseline_sep2':     '(('+vis+') || ('+inv+' && cosTheta_miss < 0.985))',
    'Baseline_sep3':     '(('+vis+') || ('+inv+' && cosTheta_miss < 0.98))',
}

# List of selections to split into high/low BDT score regions
sels = [
    'Baseline',
    'Baseline_miss', 
    'Baseline_sep',
    'Baseline_sep1', 'Baseline_sep2', 'Baseline_sep3'
]
# Split each selection into high and low BDT score regions
cutList = make_high_low(cutList, bdt_cut, sels)



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

P_up = 90 if ecm==240 else (200 if ecm==365 else 300)
P_dw = 20 if ecm==240 else (0 if ecm==365 else 0)

lead_up = 100 if ecm==240 else (250 if ecm==365 else 300)
lead_dw = 40 if ecm==240 else (0 if ecm==365 else 0)

sub_up = 60 if ecm==240 else (200 if ecm==365 else 300)
sub_dw = 20 if ecm==240 else (0 if ecm==365 else 0)

vis_up = 160 if ecm==240 else (350 if ecm==365 else 400)
vis_dw = 0

mis_up = 160 if ecm==240 else (365 if ecm==365 else 400)
mis_dw = 0

# Output histogram definitions (name, title, binning)
histoList = {

    # Lepton kinematics: leading lepton
    'leading_p':        {'name':'leading_p',
                         'title':'p_{l,leading} [GeV]',
                         'bin':int((lead_up-lead_dw)/0.5), 
                         'xmin':lead_dw, 'xmax':lead_up},

    'leading_theta':    {'name':'leading_theta',
                         'title':'#theta_{l,leading}',
                         'bin':128, 'xmin':0,  'xmax':3.2},

    'leading_phi':      {'name':'leading_phi',
                         'title':'#phi_{l,leading}',
                         'bin':64,'xmin':-3.2,'xmax':3.2},

    # Lepton kinematics: subleading lepton
    'subleading_p':     {'name':'subleading_p',
                         'title':'p_{l,subleading} [GeV]',
                         'bin':int((sub_up-sub_dw)/0.5), 
                         'xmin':sub_dw, 'xmax':sub_up},

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
                         'bin':100,'xmin':0,'xmax':10},
    
    # Z boson properties
    'zll_m':            {'name':'zll_m',
                         'title':'m_{l^{+}l^{-}} [GeV]',
                         'bin':100,'xmin':86,'xmax':96},

    'zll_p':            {'name':'zll_p',
                         'title':'p_{l^{+}l^{-}} [GeV]',
                         'bin':int((P_up-P_dw)/0.5),
                         'xmin':P_dw,'xmax':P_up},

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
                         'bin':int((vis_up-vis_dw)/0.5),
                         'xmin':vis_dw,'xmax':vis_up},

    'missingMass':      {'name':'missingMass',
                         'title':'m_{miss} [GeV]',
                         'bin':int((mis_up-mis_dw)/0.5),
                         'xmin':mis_dw,'xmax':mis_up},
    
    # Higgsstrahlungness
    'H':                {'name':'H',
                         'title':'Higgsstrahlungness [GeV^{2}]',
                         'bin':110,'xmin':0,'xmax':110},

    # BDT score
    'BDTscore':         {'name':'BDTscore',
                         'title':'BDT score',
                         'bin':500,'xmin':0,'xmax':1}
                         
}
