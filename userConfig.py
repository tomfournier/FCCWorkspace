import os 
import numpy as np

eos = True
ZH =True

final_state, ecm = "mumuH", 240

if eos:
    repo = os.path.abspath(".")
    # repo = "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH"
else:
    repo = os.getenv('PWD')
    # repo = "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH"

if ZH:
    repo = repo+"/analysis/ZH"

class loc : pass
loc.ROOT = repo
loc.OUT = f"{loc.ROOT}/output"
loc.DATA = f"{loc.OUT}/data"
loc.CSV = f"{loc.DATA}/csv"
loc.PKL = f"{loc.DATA}/pkl"
loc.PKL_Val = f"{loc.DATA}/pkl_val"
loc.ROOTFILES = f"{loc.DATA}/ROOT"
loc.PLOTS = f"{loc.OUT}/plots"
loc.PLOTS_Val = f"{loc.OUT}/plots_val"
loc.TEX = f"{loc.OUT}/tex"
loc.JSON = f"{loc.OUT}/json"

#Location of preselection events
loc.PRESEL = f"{loc.OUT}/preselection"

#Location of final selection
loc.FINAL = f"{loc.OUT}/final"

#Output BDT model location - used in official sample production to assign MVA weights
loc.BDT = f"{loc.OUT}/BDT"

#Samples for first stage BDT training
loc.TRAIN = f"{loc.OUT}/MVAInputs"

#Samples for final analysis
loc.ANALYSIS = f"{loc.OUT}/BDT_analysis_samples/"

#First stage BDT including event-level vars
train_vars = [
              #leptons
              "leading_p",
              "leading_theta",
              "subleading_p",
              "subleading_theta",
              "acolinearity",
              "acoplanarity",
              #Zed
              "zll_m",
              "zll_p",
              "zll_theta"
              #Higgsstrahlungness
              #"H",
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
