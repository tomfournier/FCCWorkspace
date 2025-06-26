import os 

######################
##### PARAMETERS #####
######################

plot_file = 'png'
frac, nb  = 1, 10
treemaker = True

ecm, lumi = 240, 10.8
recoil120, miss, bdt = False, False, False
vis = True

_120, _miss, _bdt = '_120' if recoil120 else '', '_miss' if miss else '', '_missBDT' if bdt else ''
_vis = '_vis' if vis else ''
sel = 'Baseline'+_vis+_120+_miss+_bdt

#############################
##### LOCATION OF FILES #####
#############################

eos, ZH = True, False
if eos:
    repo = os.path.abspath(".")
    # example: "/eos/user/t/tofourni/public/FCC/FCCWorkspace/"
else:
    repo = os.getenv('PWD')
    # example: "/afs/cern.ch/user/t/tofourni/eos/public/FCC/FCCWorkspace"

if ZH:
    repo = repo+"/analysis/ZH/xsec"

class loc : pass
# Location of the root folder
loc.ROOT               = repo

# Location of primary folders
loc.OUT                = f"{loc.ROOT}/output"
loc.DATA               = f"{loc.OUT}/data"
loc.PLOTS              = f"{loc.OUT}/plots"

# Location of MVA related files
loc.MVA                = f"{loc.DATA}/MVA"
loc.MVA_INPUTS         = f"{loc.MVA}/ecm/final_state/selection/MVAInputs"

loc.MVA_PROCESSED      = f"{loc.MVA}/ecm/final_state/selection/MVAProcessed"
loc.BDT                = f"{loc.MVA}/ecm/final_state/selection/BDT"

# Location of histograms
loc.HIST               = f"{loc.DATA}/histograms"

loc.HIST_MVAINPUTS     = f"{loc.HIST}/MVAInputs/ecm"
loc.HIST_MVAFINAL      = f"{loc.HIST}/MVAFinal/ecm/final_state/"

loc.HIST_PREPROCESSED  = f"{loc.HIST}/preprocessed/ecm/selection"
loc.HIST_PROCESSED     = f"{loc.HIST}/processed/ecm/selection"

loc.HIST_PSEUDO        = f"{loc.HIST}/pseudo-data/ecm/selection"

# Location of plots
loc.PLOTS_MVA          = f"{loc.PLOTS}/ecm/final_state/MVAInputs"
loc.PLOTS_MVAINPUTS    = f"{loc.PLOTS}/ecm/final_state/MVA/selection"
loc.PLOTS_BDT          = f"{loc.PLOTS}/ecm/final_state/evaluation/selection"
loc.PLOTS_MEASUREMENT  = f"{loc.PLOTS}/ecm/final_state/measurement/selection"
loc.PLOTS_BIAS         = f"{loc.PLOTS}/ecm/final_state/bias/selection"

# Location of combine files
loc.COMBINE            = f"{loc.DATA}/combine/ecm/selection"

loc.COMBINE_CHANNEL    = f"{loc.COMBINE}/final_state"
loc.COMBINE_NOMINAL    = f"{loc.COMBINE_CHANNEL}/nominal"
loc.COMBINE_BIAS       = f"{loc.COMBINE_CHANNEL}/bias"

# Location of combine files when doing nominal fit
loc.NOMINAL_LOG        = f"{loc.COMBINE_NOMINAL}/log" 
loc.NOMINAL_RESULT     = f"{loc.COMBINE_NOMINAL}/results"
loc.NOMINAL_DATACARD   = f"{loc.COMBINE_NOMINAL}/datacard"
loc.NOMINAL_WS         = f"{loc.COMBINE_NOMINAL}/WS"

# Location of combine files when doing bias test
loc.BIAS_LOG           = f"{loc.COMBINE_BIAS}/log" 
loc.BIAS_FIT_RESULT    = f"{loc.COMBINE_BIAS}/results/fit"
loc.BIAS_RESULT        = f"{loc.COMBINE_BIAS}/results/bias"
loc.BIAS_DATACARD      = f"{loc.COMBINE_BIAS}/datacard"
loc.BIAS_WS            = f"{loc.COMBINE_BIAS}/WS"



#########################################
##### FUNCTIONS TO EXTRACT THE PATH #####
#########################################

def get_loc(path: str, cat: str , ecm: int, sel: str) -> str:
    path = path.replace('final_state', cat)
    path = path.replace('ecm', str(ecm))
    path = path.replace('selection', sel)
    return path

def select(recoil120: bool = False, miss: bool = False, bdt: bool = False, vis: bool = False) -> str:
    sel = 'Baseline'
    if recoil120: sel += '_120'
    elif miss:    sel += '_miss'
    elif bdt:     sel += '_missBDT'
    elif vis:     sel += '_vis'
    return sel



#############################
##### VARIABLES FOR BDT #####
#############################

#First stage BDT including event-level vars
train_vars = [
    "leading_p", "leading_theta", "subleading_p", "subleading_theta",
    "acolinearity", "acoplanarity", "zll_m", "zll_p", "zll_theta"
]

# Latex mapping for importance plot
latex_mapping = {
    'leading_p':        r'$p_{\ell,leading}$',
    'leading_theta':    r'$\theta_{\ell,leading}$',
    'subleading_p':     r'$p_{\ell,subleading}$',
    'subleading_theta': r'$\theta_{\ell,subleading}$',
    'acolinearity':     r'$|\Delta\theta_{\ell\ell}|$',
    'acoplanarity':     r'$|\Delta\phi_{\ell\ell}|$',
    'zll_m':            r'$m_{\ell\ell}$',
    'zll_p':            r'$p_{\ell\ell}$',
    'zll_theta':        r'$\theta_{\ell\ell}$',
    'cosTheta_miss':    r'$cos\theta_{miss}$',
    'H':                r'$H$'
}

Label = {}
Label = {
    f"mumuH":       r"$e^+e^-\rightarrow Z(\mu^+\mu^-)H$",
    f"ZZ":          r"$e^+e^-\rightarrow ZZ$", 
    f"Zmumu":       r"$e^+e^-\rightarrow\mu^+\mu^-$",
    f"WWmumu":      r"$e^+e^-\rightarrow WW[\nu_{\mu}\mu]$",
    f"egamma_mumu": r"$e^-\gamma\rightarrow e^-Z(\mu^+\mu^-)$",
    f"gammae_mumu": r"$e^+\gamma\rightarrow e^+Z(\mu^+\mu^-)$",
    f"gaga_mumu":   r"$\gamma\gamma\rightarrow\mu^+\mu^-$",
    
    f"eeH":         r"$e^+e^-\rightarrow Z(e^+e^-)H$",
    f"Zee":         r"$e^+e^-\rightarrow e^+e^-$",
    f"WWee":        r"$e^+e^-\rightarrow WW[\nu_{e}e]$",
    f"egamma_ee":   r"$e^-\gamma\rightarrow e^-Z(e^+e^-)$",
    f"gammae_ee":   r"$e^+\gamma\rightarrow e^+Z(e^+e^-)$",
    f"gaga_ee":     r"$\gamma\gamma\rightarrow e^+e^-$"
}



###########################
##### OTHER VARIABLES #####
###########################

# Decays of the Z and the Higgs
z_decays = ['bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu']
h_decays = ['bb', 'cc', 'gg', 'ss', 'mumu', 'tautau', 'ZZ', 'WW', 'Za', 'aa', 'inv']

param = {'fraction': frac} 
# param = {'chunks':   nb}
