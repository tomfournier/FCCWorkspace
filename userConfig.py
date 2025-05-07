import os 
import numpy as np

eos = True
ZH =True

final_state, ecm = "mumu", 240
intLumi = 10.8 # in ab-1

plot_file = "png"

if eos:
    repo = os.path.abspath(".")
    # repo = "/eos/user/t/tofourni/public/FCC/FCCWorkspace/"
else:
    repo = os.getenv('PWD')
    # repo = "/afs/cern.ch/user/t/tofourni/eos/public/FCC/FCCWorkspace"

batch = repo

if ZH:
    repo = repo+"/analysis/ZH"

class loc : pass
loc.ROOT = repo
loc.OUT = f"{loc.ROOT}/output"
loc.DATA = f"{loc.OUT}/data/{final_state}"
loc.CSV = f"{loc.DATA}/csv"
loc.PKL = f"{loc.DATA}/pkl"
loc.PKL_Val = f"{loc.DATA}/pkl_val"
loc.ROOTFILES = f"{loc.DATA}/ROOT"
loc.PLOTS = f"{repo}/plots/{final_state}"
loc.PLOTS_Val = f"{repo}/plots_val/{final_state}/"
loc.METRIC = f"{loc.PLOTS}/metrics"
loc.TEX = f"{loc.OUT}/tex"
loc.JSON = f"{loc.OUT}/json"
loc.BATCH = f"{batch}/userBatch.Config"

#Location of preselection events
loc.PRESEL = f"{loc.OUT}/preselection/{final_state}"

#Location of final selection
loc.FINAL = f"{loc.OUT}/final/{final_state}/"

#Output BDT model location - used in official sample production to assign MVA weights
loc.BDT = f"{loc.OUT}/BDT/{final_state}"

#Samples for first stage BDT training
loc.TRAIN = f"{loc.OUT}/MVAInputs/{final_state}"

#Location of BDT evaluation plots
loc.PLOTS_BDT = f"{loc.PLOTS}/evaluation"

#Samples for final analysis pre-selection
loc.ANALYSIS = f"{loc.OUT}/BDT_analysis/{final_state}"

#Samples for final analysis final selection
loc.ANALYSIS_FINAL = f"{loc.OUT}/BDT_final/{final_state}/"

# Process samples that should match the produced files.
samples = {
    #signal
    f"wzp6_ee_{final_state}H_ecm{ecm}",
    #background: 
    f"p8_ee_ZZ_ecm{ecm}",
    #rare backgrounds:
    f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
    f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
    f"wzp6_gaga_{final_state}_60_ecm{ecm}",
}

# Parameter of processList
# paramList = {'chunks':20} 
paramList = {'frac':0.1}

# Process list that should match the produced files.
processList = {i:paramList for i in samples}

if final_state=="mumu":
    processList[f"wzp6_ee_{final_state}_ecm{ecm}"] = paramList
elif final_state=="ee":
    processList[f"wzp6_ee_{final_state}_Mee_30_150_ecm{ecm}"] = paramList

processList1 = processList.copy()
processList[f"p8_ee_WW_{final_state}_ecm{ecm}"] = paramList

processList1[f"p8_ee_WW_ecm{ecm}"] = paramList
processList1[f"wzp6_ee_tautau_ecm{ecm}"] = paramList
processList1[f"wzp6_gaga_tautau_60_ecm{ecm}"] = paramList
processList1[f"wzp6_ee_nuenueZ_ecm{ecm}"] = paramList

mass = False
if mass:
    processList1[f"wzp6_ee_{final_state}H_mH-higher-100MeV_ecm{ecm}"] = paramList
    processList1[f"wzp6_ee_{final_state}H_mH-higher-50MeV_ecm{ecm}"] = paramList
    processList1[f"wzp6_ee_{final_state}H_mH-lower-100MeV_ecm{ecm}"] = paramList
    processList1[f"wzp6_ee_{final_state}H_mH-lower-50MeV_ecm{ecm}"] = paramList

syst = False
if syst:
    processList1[f"wzp6_ee_{final_state}H_BES-higher-1pc_ecm{ecm}"] = paramList
    processList1[f"wzp6_ee_{final_state}H_BES-lower-1pc_ecm{ecm}"] = paramList

#First stage BDT including event-level vars
train_vars = [
    #leptons
    "leading_p", "leading_theta",
    "subleading_p", "subleading_theta",
    "acolinearity", "acoplanarity",
    #Zed
    "zll_m", "zll_p", "zll_theta"
    #Higgsstrahlungness
    #"H"
]

train_vars_scaleup = [
    # leptons
    "leading_p_scaleup",
    "leading_theta_scaleup",
    "subleading_p_scaleup",
    "subleading_theta_scaleup",
    "zll_leptons_acolinearity_scaleup",
    "zll_leptons_acoplanarity_scaleup",
    # Zed
    "zll_m_scaleup",
    "zll_p_scaleup",
    "zll_theta_scaleup"
    # Higgsstrahlungness
    # "H"
]

train_vars_scaledw = [
    # leptons
    "leading_p_scaledw",
    "leading_theta_scaledw",
    "subleading_p_scaledw",
    "subleading_theta_scaledw",
    "zll_leptons_acolinearity_scaledw",
    "zll_leptons_acoplanarity_scaledw",
    # Zed
    "zll_m_scaledw",
    "zll_p_scaledw",
    "zll_theta_scaledw"
    # Higgsstrahlungness
    # "H"
]

latex_mapping = {
    'leading_p': r'$p_{\ell_1}$',
    'leading_theta': r'$\theta_{\ell_1}$',
    'subleading_p': r'$p_{\ell_2}$',
    'subleading_theta': r'$\theta_{\ell_2}$',
    'acolinearity': r'$|\Delta\theta_{\ell\ell}|$',
    'acoplanarity': r'$|\Delta\phi_{\ell\ell}|$',
    'zll_m': r'$m_{\ell\ell}$',
    'zll_p': r'$p_{\ell\ell}$',
    'zll_theta': r'$\theta_{\ell\ell}$',
    'H': r'$H$'
}

#Decay modes used in first stage training and their respective file names

mode_names = {f"{final_state}H": f"wzp6_ee_{final_state}H_ecm{ecm}",
              f"ZZ": f"p8_ee_ZZ_ecm{ecm}",
              f"WW{final_state}": f"p8_ee_WW_{final_state}_ecm{ecm}",
              f"egamma": f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
              f"gammae": f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
              f"gaga_{final_state}": f"wzp6_gaga_{final_state}_60_ecm{ecm}"}

if final_state == "mumu":
    mode_names['Zll'] = f"wzp6_ee_{final_state}_ecm{ecm}"
elif final_state == "ee":
    mode_names['Zll'] = f"wzp6_ee_{final_state}_Mee_30_150_ecm{ecm}"

Label = {}
if final_state == "mumu":
    Label = {f"{final_state}H": r"$e^+e^-\rightarrow Z(\mu^+\mu^-)H$",
            f"ZZ": r"$e^+e^-\rightarrow ZZ$",
            f"WW{final_state}": r"$e^+e^-\rightarrow WW[\nu_{\mu}\mu]$",
            f"Zll": r"$e^+e^-\rightarrow\mu^+\mu^-$",
            f"egamma": r"$e^-\gamma\rightarrow e^-Z(\mu^+\mu^-)$",
            f"gammae": r"$e^+\gamma\rightarrow e^+Z(\mu^+\mu^-)$",
            f"gaga_{final_state}": r"$\gamma\gamma\rightarrow\mu^+\mu^-$"}

elif final_state == "ee":
    Label = {f"{final_state}H": r"$e^+e^-\rightarrow Z(e^+e^-)H$",
            f"ZZ": r"$e^+e^-\rightarrow ZZ$",
            f"WW{final_state}": r"$e^+e^-\rightarrow WW[\nu_{e}e]$",
            f"Zll": r"$e^+e^-\rightarrow e^+e^-$",
            f"egamma": r"$e^-\gamma\rightarrow e^-Z(e^+e^-)$",
            f"gammae": r"$e^+\gamma\rightarrow e^+Z(e^+e^-)$",
            f"gaga_{final_state}": r"$\gamma\gamma\rightarrow e^+e^-$"}
