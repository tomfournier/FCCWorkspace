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
loc.PLOTS_Val = f"{repo}/plots_val/{final_state}"
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

#Samples for final analysis pre-selection
loc.ANALYSIS = f"{loc.OUT}/BDT_analysis_samples/{final_state}"

#Samples for final analysis final selection
loc.ANALYSIS_FINAL = f"{loc.OUT}/BDT_analysis_samples_final/{final_state}"

# Process list that should match the produced files.
processList = {
  #signal
  f"wzp6_ee_{final_state}H_ecm{ecm}",
  #background: 
  f"p8_ee_WW_{final_state}_ecm{ecm}",
  f"p8_ee_ZZ_ecm240",
  f"wzp6_ee_{final_state}_ecm{ecm}",
  #rare backgrounds:
  f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
  f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
  f"wzp6_gaga_{final_state}_60_ecm{ecm}",
}

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
              f"Zll": f"wzp6_ee_{final_state}_ecm{ecm}",
              f"egamma": f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
              f"gammae": f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
              f"gaga_{final_state}": f"wzp6_gaga_{final_state}_60_ecm{ecm}"}
