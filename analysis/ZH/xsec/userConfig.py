import os, sys, uproot
from glob import glob



######################
##### PARAMETERS #####
######################

plot_file, fitting = ['png'], False
frac, nb  = 1, 10
inv, ww   = False, True

# Precaution to not have the two variables set to True
if inv and ww: ww = False 

ecm, lumi = 240, 10.8



#############################
##### LOCATION OF FILES #####
#############################

eos = True
if eos: repo = os.path.abspath(".")
# example: "/eos/user/t/tofourni/public/FCC/FCCWorkspace/"
else: repo = os.getenv('PWD')
# example: "/afs/cern.ch/user/t/tofourni/eos/public/FCC/FCCWorkspace"

if   "xsec"             not in repo: repo = repo+"/xsec"
elif "ZH/xsec"          not in repo: repo = repo+"/ZH/xsec"
elif "analysis/ZH/xsec" not in repo: repo = repo+"/analysis/ZH/xsec"

class loc : pass
# Location of the root folder
loc.ROOT               = repo

# Location of primary folders
loc.PACKAGE            = f"{loc.ROOT}/package"
loc.OUT                = f"{loc.ROOT}/output"
loc.PLOTS              = f"{loc.OUT}/plots"
loc.DATA               = f"{loc.OUT}/data"
loc.TMP                = f"{loc.OUT}/tmp"

# Location of files needed for configuration
loc.JSON               = f"{loc.TMP}/config_json"

# Location of events
loc.EVENTS             = f"{loc.DATA}/events/ecm/cat/analysis"
loc.EVENTS_TRAINING    = f"{loc.DATA}/events/ecm/cat/training"

# Location of MVA related files
loc.MVA                = f"{loc.DATA}/MVA"
loc.MVA_INPUTS         = f"{loc.MVA}/ecm/cat/sel/MVAInputs"
loc.BDT                = f"{loc.MVA}/ecm/cat/sel/BDT"

# Location of histograms
loc.HIST               = f"{loc.DATA}/histograms"

loc.HIST_MVA           = f"{loc.HIST}/MVAInputs/ecm/cat/"
loc.HIST_PREPROCESSED  = f"{loc.HIST}/preprocessed/ecm/cat"
loc.HIST_PROCESSED     = f"{loc.HIST}/processed/ecm/cat/sel"

# Location of plots
loc.PLOTS_MVA          = f"{loc.PLOTS}/MVAInputs/ecm/cat"
loc.PLOTS_BDT          = f"{loc.PLOTS}/evaluation/ecm/cat/sel"
loc.PLOTS_MEASUREMENT  = f"{loc.PLOTS}/measurement/ecm/cat"

# Location of combine files
loc.COMBINE            = f"{loc.DATA}/combine/sel/ecm/cat"

loc.COMBINE_NOMINAL    = f"{loc.COMBINE}/nominal"
loc.COMBINE_BIAS       = f"{loc.COMBINE}/bias"

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
    path = path.replace('cat', cat)
    path = path.replace('ecm', str(ecm))
    path = path.replace('sel', sel)
    return path

def event(procs: list, path: str = '', end='.root') -> list:
    newprocs = []
    for proc in procs:
        if os.path.exists(f'{path}{proc}{end}'):
            filenames = [f'{path}{proc}{end}']
        else:
            filenames = glob(f'{path}{proc}/*')
        isTTree = []
        for i, filename in enumerate(filenames):
            file = uproot.open(filename)
            if 'events' in file:
                isTTree.append(i)
        if len(isTTree)==len(filenames):
            newprocs.append(proc)
    return newprocs

def add(names: list = []) -> None:
    sys.path.append(loc.PACKAGE)
    if names!=[]:
        for name in names:
            sys.path.append(name)



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

param = {'fraction': frac, 'chunks': nb} 
