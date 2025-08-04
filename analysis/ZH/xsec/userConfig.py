import os, time, json, argparse
import numpy as np



######################
##### PARAMETERS #####
######################

plot_file = 'png'
frac, nb  = 1, 10
treemaker = True

ecm, lumi = 240, 10.8
recoil120, miss, bdt = False, False, False
leading,   vis,  sep = False, False, True

_120, _miss, _bdt = '_120' if recoil120 else '', '_miss' if miss else '', '_missBDT' if bdt else ''
_vis, _leading, _sep = '_vis' if vis else '', '_leading' if leading else '', '_sep' if sep else ''
sel = 'Baseline'+_sep+_leading+_vis+_120+_miss+_bdt



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
loc.TMP                = f"{loc.OUT}/tmp"

# Location of files needed for configuration
loc.JSON               = f"{loc.TMP}/config_json"

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

def select(recoil120: bool = False, miss: bool = False, bdt: bool = False, 
           leading: bool = False, vis: bool = False, sep: bool = False) -> str:
    sel = 'Baseline'
    if sep:         sel += '_sep'
    elif leading:   sel += '_leading'
    elif vis:       sel += '_vis'
    elif recoil120: sel += '_120'
    elif miss:      sel += '_miss'
    elif bdt:       sel += '_missBDT'
    return sel

def extra_arg(args: list) -> str:
    cmd = ''
    for arg in args: cmd += f' --{arg}'
    return cmd

def argument(cat: bool = True, sel: bool = True, lumi: bool = False, comb: bool = False, 
             extra: bool = False, run: bool = False, ILC: bool = False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365, -1], type=int, default=240)
    
    if cat:
        parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')

    if lumi:
        parser.add_argument('--lumi', help='Integrated luminosity in attobarns', type=float, default=-1)

    if sel:
        parser.add_argument('--recoil', help='Cut with 120 GeV < recoil mass < 140 GeV',    action='store_true')
        parser.add_argument('--miss',   help='Add the cos(theta_miss) < 0.98 cut',          action='store_true')
        parser.add_argument('--bdt',    help='Add cos(theta_miss) cut as input to the BDT', action='store_true')
        parser.add_argument('--lead',   help='Add the p_leading and p_subleading cuts',     action='store_true')
        parser.add_argument('--vis',    help='Add E_vis > 100 GeV cut',                     action='store_true')
        parser.add_argument('--sep',    help='Separate events by using E_vis',              action='store_true')

    if comb:
        parser.add_argument("--combine", help='Combine the channel to do the fit', action='store_true')

    if extra:
        parser.add_argument("--target", type=str,   help="Target pseudodata",      default="bb")
        parser.add_argument("--pert",   type=float, help="Target pseudodata size", default=1.0)
        parser.add_argument("--run",                help="Run combine",            action='store_true')
        parser.add_argument("--freezeBackgrounds",  help="Freeze backgrounds",     action='store_true')
        parser.add_argument("--floatBackgrounds",   help="Float backgrounds",      action='store_true')
        parser.add_argument("--plot_dc",            help="Plot datacard",          action='store_true')

    if ILC:
        parser.add_argument("--polL", help="Scale to left polarization",  action='store_true')
        parser.add_argument("--polR", help="Scale to right polarization", action='store_true')
        parser.add_argument("--ILC",  help="Scale to ILC luminosity",     action='store_true')

    if run:
        parser.add_argument('--presel',  help='Run only pre-selection.py',     action='store_true')
        parser.add_argument('--final',   help='Run only final-selection.py',   action='store_true')
        parser.add_argument('--plots',   help='Run only plots.py',             action='store_true')

        parser.add_argument('--input',   help='Run only process_input.py',     action='store_true')
        parser.add_argument('--train',   help='Run only train_bdt.py',         action='store_true')
        parser.add_argument('--eval',    help='Run only evaluation.py',        action='store_true')

        parser.add_argument('--sel',     help='Run only selection.py',         action='store_true')
        parser.add_argument('--process', help='Run only process_histogram.py', action='store_true')
        parser.add_argument('--comb',    help='Run only combine.py',           action='store_true')

        parser.add_argument('--full',    help='Run all files in 2-BDT',        action='store_true')

        parser.add_argument('--multi',   help='Run combined selection, rely combined selection with "-"'
                            ' and separate the combined selections with "_"', type=list, default='')
            
    arg = parser.parse_args()
    args = [arg]
    
    sel_list   =          [arg.recoil, arg.miss, arg.bdt, arg.lead, arg.vis, arg.sep]
    extra_list = np.array([   'recoil',   'miss',   'bdt',   'lead',   'vis',   'sep'])
    if sel:   args.append(sel_list)
    if extra: args.append(extra_arg(extra_list[np.where(sel_list)]))
    return args

def dump_json(arg, file, indent=4):
    with open(file, mode='w', encoding='utf-8') as fOut:
        json.dump(arg, fOut, indent=indent)

def load_json(file):
    with open(file, mode='r', encoding='utf-8') as fIn:
        arg = json.load(fIn)
    return arg

def warning(cat, comb: bool = False, is_there_comb: bool = False):
    if not is_there_comb and cat=='':
        print('\n-----------------------------------------\n')
        print('Final state was not selected. Aborting...')
        print('\n-----------------------------------------\n')
        exit(0)

    if is_there_comb and cat=='' and not comb:
        print('\n-----------------------------------------------------\n')
        print('Final state or combine were not selected. Aborting...')
        print('\n-----------------------------------------------------\n')
        exit(0)

def timer(t):
    dt = time.time() - t
    h, m, s = dt//3600, dt//60, dt%60 

    print('\n\n-----------------------------------------\n')
    print(f'Time taken to run the code: {h:.0f} h {m:.0f} min {s:.2f} s')
    print('\n-----------------------------------------\n\n')



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
