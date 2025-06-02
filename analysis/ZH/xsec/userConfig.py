import os 

######################
##### PARAMETERS #####
######################

final_state, plot_file              = 'ee', 'png'
ecm, intLumi, pert, fraction        = 240, 10.8, 1.05, 1
miss_BDT, miss, combine, recoil_120 = False, False, True, False

if miss_BDT and miss:
    miss = False

missing   = '_miss' if miss else ''
_120      = '_120' if recoil_120 else ''
_missBDT  = '_missBDT' if miss_BDT else ''

selection = 'Baseline'+_120+missing+_missBDT



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
loc.MVA_INPUTS         = f"{loc.MVA}/{ecm}/{final_state}/{selection}/MVAInputs".replace('_missBDT','')

loc.MVA_PROCESSED      = f"{loc.MVA}/{ecm}/{final_state}/{selection}/MVAProcessed"
loc.BDT                = f"{loc.MVA}/{ecm}/{final_state}/{selection}/BDT"

# Location of histograms
loc.HIST               = f"{loc.DATA}/histograms"

loc.HIST_MVA           = f"{loc.HIST}/MVAFinal/{ecm}/{final_state}/"
loc.HIST_PREPROCESSED  = f"{loc.HIST}/preprocessed/{ecm}/{selection}"
loc.HIST_PROCESSED     = f"{loc.HIST}/processed/{ecm}/{selection}/{final_state}"
loc.HIST_PSEUDO        = f"{loc.HIST}/pseudo-data/{ecm}/{selection}"

# Location of plots
loc.PLOTS_MVA          = f"{loc.PLOTS}/{ecm}/{final_state}/MVAInputs"
loc.PLOTS_BDT          = f"{loc.PLOTS}/{ecm}/{final_state}/evaluation/{selection}"
loc.PLOTS_MEASUREMENT  = f"{loc.PLOTS}/{ecm}/final_state/measurement/{selection}"

# Location of combine files
loc.COMBINE            = f"{loc.DATA}/combine/{ecm}/{selection}"

loc.COMBINE_CHANNEL    = f"{loc.COMBINE}/{final_state}" if not combine else f"{loc.COMBINE}/combined"
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



###################
##### SAMPLES #####
###################

ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if final_state=='ee' else f"wzp6_ee_mumu_ecm{ecm}"

# Process samples for BDT
samples_BDT = [
    #signal
    f"wzp6_ee_{final_state}H_ecm{ecm}",

    #background: 
    f"p8_ee_ZZ_ecm{ecm}", f"p8_ee_WW_{final_state}_ecm{ecm}", ee_ll,

    #rare backgrounds:
    f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
    f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
    f"wzp6_gaga_{final_state}_60_ecm{ecm}"
]

# Background samples for measurement
samples_bkg = [
    f"p8_ee_WW_ecm{ecm}", f"p8_ee_ZZ_ecm{ecm}",
    f"wzp6_ee_ee_Mee_30_150_ecm{ecm}", f"wzp6_ee_mumu_ecm{ecm}", f"wzp6_ee_tautau_ecm{ecm}",
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}', f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f"wzp6_gaga_ee_60_ecm{ecm}", f"wzp6_gaga_mumu_60_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}", 
    f"wzp6_ee_nuenueZ_ecm{ecm}"
]

# Signal samples for measurement
z_decays = ['bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu']
h_decays = ['bb', 'cc', 'ss', 'gg', 'mumu', 'tautau', 'WW', 'ZZ', 'Za', 'aa'] #, 'inv']

samples_sig = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]
for i in ['ee', 'mumu']:
    samples_sig.append(f"wzp6_ee_{i}H_ecm{ecm}")
samples_sig.append(f'wzp6_ee_ZH_Hinv_ecm{ecm}')

# Signal samples for measurement
sig_combine = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]

# For Beam Energy spread uncertainty
syst = False
if syst:
    samples_sig.append(f"wzp6_ee_{final_state}H_BES-higher-1pc_ecm{ecm}")
    samples_sig.append(f"wzp6_ee_{final_state}H_BES-lower-1pc_ecm{ecm}")

# Total process samples for measurement
samples = samples_sig + samples_bkg

# Parameter of processList
# paramList = {'chunks':20} 
paramList = {'fraction':fraction}

# Process list for BDT
List_bdt = {i:paramList for i in samples_BDT}

# Process list for measurement
List_meas = {j:paramList for j in samples}



###################################
##### OTHER VARIABLES FOR BDT #####
###################################

#First stage BDT including event-level vars
train_vars = [
    #leptons
    "leading_p", "leading_theta", "subleading_p", "subleading_theta",
    "acolinearity", "acoplanarity",
    #Z
    "zll_m", "zll_p", "zll_theta"
]
if miss_BDT:
    train_vars.append("cosTheta_miss")

# Latex mapping for importance plot
latex_mapping = {
    'leading_p': r'$p_{\ell,leading}$',
    'leading_theta': r'$\theta_{\ell,leading}$',
    'subleading_p': r'$p_{\ell,subleading}$',
    'subleading_theta': r'$\theta_{\ell,subleading}$',
    'acolinearity': r'$|\Delta\theta_{\ell\ell}|$',
    'acoplanarity': r'$|\Delta\phi_{\ell\ell}|$',
    'zll_m': r'$m_{\ell\ell}$',
    'zll_p': r'$p_{\ell\ell}$',
    'zll_theta': r'$\theta_{\ell\ell}$',
    'cosTheta_miss': r'$cos\theta_{miss}$',
    'H': r'$H$'
}

# Decay modes used in first stage training and their respective file names
mode_names = {f"{final_state}H": f"wzp6_ee_{final_state}H_ecm{ecm}",
              f"ZZ": f"p8_ee_ZZ_ecm{ecm}", 'Zll':ee_ll,
              f"WW{final_state}": f"p8_ee_WW_{final_state}_ecm{ecm}",
              f"egamma": f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
              f"gammae": f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
              f"gaga_{final_state}": f"wzp6_gaga_{final_state}_60_ecm{ecm}"}

Label = {}
if final_state == "mumu":
    Label = {f"{final_state}H": r"$e^+e^-\rightarrow Z(\mu^+\mu^-)H$",
            f"ZZ": r"$e^+e^-\rightarrow ZZ$", f"Zll": r"$e^+e^-\rightarrow\mu^+\mu^-$",
            f"WW{final_state}": r"$e^+e^-\rightarrow WW[\nu_{\mu}\mu]$",
            f"egamma": r"$e^-\gamma\rightarrow e^-Z(\mu^+\mu^-)$",
            f"gammae": r"$e^+\gamma\rightarrow e^+Z(\mu^+\mu^-)$",
            f"gaga_{final_state}": r"$\gamma\gamma\rightarrow\mu^+\mu^-$"}

elif final_state == "ee":
    Label = {f"{final_state}H": r"$e^+e^-\rightarrow Z(e^+e^-)H$",
            f"ZZ": r"$e^+e^-\rightarrow ZZ$", f"Zll": r"$e^+e^-\rightarrow e^+e^-$",
            f"WW{final_state}": r"$e^+e^-\rightarrow WW[\nu_{e}e]$",
            f"egamma": r"$e^-\gamma\rightarrow e^-Z(e^+e^-)$",
            f"gammae": r"$e^+\gamma\rightarrow e^+Z(e^+e^-)$",
            f"gaga_{final_state}": r"$\gamma\gamma\rightarrow e^+e^-$"}

###########################
##### OTHER VARIABLES #####
###########################

procs_cfg = {
"ZH"        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays],
"ZmumuH"    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
"ZeeH"      : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in ["ee"] for y in h_decays],
"WW"        : [f'p8_ee_WW_ecm{ecm}'],
"ZZ"        : [f'p8_ee_ZZ_ecm{ecm}'],
"Zgamma"    : [f'wzp6_ee_tautau_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
                f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],
"Rare"      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
                f'wzp6_gaga_mumu_60_ecm{ecm}', f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
                f'wzp6_gammae_eZ_Zee_ecm{ecm}', f'wzp6_gaga_ee_60_ecm{ecm}', 
                f'wzp6_gaga_tautau_60_ecm{ecm}', f'wzp6_ee_nuenueZ_ecm{ecm}'],
}
