import os 
import numpy as np

eos = True
ZH =True

final_state, ecm = "ee", 240
intLumi = 10.8 # in ab-1

plot_file, final = "png", False
combine, miss = True, False

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

#Location of the cutflow histmaker for MVAInputs
loc.CUTFLOW_MVA = f"{loc.OUT}/cutFlow/{final_state}/MVAInputs"

#Location of the cutflow plots for MVAInputs
loc.PLOTS_CUTFLOW_MVA = f"{loc.PLOTS}/cutFlow/MVAInputs"

#Location of the cutflow histmaker for MVAInputs
loc.CUTFLOW_FINAL = f"{loc.OUT}/cutFlow/{final_state}/final"

#Location of the cutflow plots for MVAInputs
loc.PLOTS_CUTFLOW_FINAL = f"{loc.PLOTS}/cutFlow/final"

#Location of the hists for the combine part
loc.COMBINE_HIST = f"{loc.OUT}/combine/hists"

#Location of the hists for the combine part
loc.COMBINE_PROC = f"{loc.OUT}/combine/hists_processed/{final_state}"

#Location of the combine output
if combine:
    loc.COMBINE = f"{loc.OUT}/combine"
else:
    loc.COMBINE = f"{loc.OUT}/combine/{final_state}"

#Location of the hists for the model-independence part
loc.MODEL = f"{loc.OUT}/model-independence/hists"

#Location of the plots for the model-independence part
loc.MODEL_PLOTS = loc.PLOTS_Val = f"{repo}/plots_independence"

#Location of the hists for the model-independence part
loc.BIAS = f"{loc.OUT}/model-independence/bias"

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
    #Missing energy information
    # cosTheta_miss
    #Higgsstrahlungness
    # "H"
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
    
if final_state=='mumu':
    procDictAdd = {"wzp6_ee_mumuH_ecm240": {"numberOfEvents": 1200000, "sumOfWeights": 1200000.0, "crossSection": 0.0067643, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_egamma_eZ_Zmumu_ecm240": {"numberOfEvents": 2500000, "sumOfWeights": 2500000.0, "crossSection": 0.10368, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_gammae_eZ_Zmumu_ecm240": {"numberOfEvents": 6000000, "sumOfWeights": 6000000.0, "crossSection": 0.10368, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_gaga_mumu_60_ecm240": {"numberOfEvents": 19500000, "sumOfWeights": 19500000.0, "crossSection": 0.873, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_ee_mumu_ecm240": {"numberOfEvents": 116400000, "sumOfWeights": 116400000.0, "crossSection": 5.288, "kfactor": 1.0, "matchingEfficiency": 1.0},
    }
elif final_state=='ee':
    procDictAdd = {"wzp6_ee_eeH_ecm240": {"numberOfEvents": 1200000, "sumOfWeights": 1200000.0, "crossSection": 0.0071611, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_egamma_eZ_Zee_ecm240": {"numberOfEvents": 5900000, "sumOfWeights": 5900000.0, "crossSection": 0.05198, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_gammae_eZ_Zee_ecm240": {"numberOfEvents": 6000000, "sumOfWeights": 6000000.0, "crossSection": 0.05198, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_gaga_ee_60_ecm240": {"numberOfEvents": 19500000, "sumOfWeights": 19500000.0, "crossSection": 0.873, "kfactor": 1.0, "matchingEfficiency": 1.0},
                "wzp6_ee_ee_Mee_30_150_ecm240": {"numberOfEvents": 119100000, "sumOfWeights": 119100000.0, "crossSection": 8.305, "kfactor": 1.0, "matchingEfficiency": 1.0},
    }

fraction = 1

processListBkg = {

    f'p8_ee_WW_ecm{ecm}':                  {'fraction':fraction},
    f'p8_ee_WW_mumu_ecm{ecm}':             {'fraction':fraction},
    f'p8_ee_WW_ee_ecm{ecm}':               {'fraction':fraction},
    f'p8_ee_ZZ_ecm{ecm}':                  {'fraction':fraction},
    f'wz3p6_ee_uu_ecm{ecm}':               {'fraction':fraction},
    f'wz3p6_ee_dd_ecm{ecm}':               {'fraction':fraction},
    f'wz3p6_ee_cc_ecm{ecm}':               {'fraction':fraction},
    f'wz3p6_ee_ss_ecm{ecm}':               {'fraction':fraction},
    f'wz3p6_ee_bb_ecm{ecm}':               {'fraction':fraction},
    f'wz3p6_ee_tautau_ecm{ecm}':           {'fraction':fraction},
    f'wz3p6_ee_mumu_ecm{ecm}':             {'fraction':fraction},
    f'wz3p6_ee_ee_Mee_30_150_ecm{ecm}':    {'fraction':fraction},
    f'wz3p6_ee_nunu_ecm{ecm}':             {'fraction':fraction},

    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}':      {'fraction':fraction},
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}':      {'fraction':fraction},
    f'wzp6_gaga_mumu_60_ecm{ecm}':         {'fraction':fraction},

    f'wzp6_egamma_eZ_Zee_ecm{ecm}':        {'fraction':fraction},
    f'wzp6_gammae_eZ_Zee_ecm{ecm}':        {'fraction':fraction},
    f'wzp6_gaga_ee_60_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_gaga_tautau_60_ecm{ecm}':       {'fraction':fraction},
    f'wzp6_ee_nuenueZ_ecm{ecm}':           {'fraction':fraction},
}

processListSignal = {

    # f'wzp6_ee_qqH_Hbb_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_Hcc_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_Hss_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_Hgg_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_Haa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_HZa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_HWW_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_HZZ_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_qqH_Hmumu_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_qqH_Htautau_ecm{ecm}':       {'fraction':fraction},
    # f'wz3p6_ee_qqH_Hinv_ecm{ecm}':         {'fraction':fraction},

    # f'wzp6_ee_ssH_Hbb_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_Hcc_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_Hss_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_Hgg_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_Haa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_HZa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_HWW_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_HZZ_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ssH_Hmumu_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_ssH_Htautau_ecm{ecm}':       {'fraction':fraction},
    # f'wz3p6_ee_ssH_Hinv_ecm{ecm}':         {'fraction':fraction},

    # f'wzp6_ee_ccH_Hbb_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_Hcc_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_Hss_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_Hgg_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_Haa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_HZa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_HWW_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_HZZ_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_ccH_Hmumu_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_ccH_Htautau_ecm{ecm}':       {'fraction':fraction},
    # f'wz3p6_ee_ccH_Hinv_ecm{ecm}':         {'fraction':fraction},


    # f'wzp6_ee_bbH_Hbb_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_Hcc_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_Hss_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_Hgg_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_Haa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_HZa_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_HWW_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_HZZ_ecm{ecm}':           {'fraction':fraction},
    # f'wzp6_ee_bbH_Hmumu_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_bbH_Htautau_ecm{ecm}':       {'fraction':fraction},
    # f'wz3p6_ee_bbH_Hinv_ecm{ecm}':         {'fraction':fraction},


    f'wzp6_ee_eeH_Hbb_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hcc_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hss_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hgg_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Haa_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_HZa_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_HWW_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_HZZ_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hmumu_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_eeH_Htautau_ecm{ecm}':       {'fraction':fraction},
    f'wz3p6_ee_eeH_Hinv_ecm{ecm}':         {'fraction':fraction},

    f'wzp6_ee_mumuH_Hbb_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hcc_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hss_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hgg_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Haa_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_HZa_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_HWW_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_HZZ_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hmumu_ecm{ecm}':       {'fraction':fraction},
    f'wzp6_ee_mumuH_Htautau_ecm{ecm}':     {'fraction':fraction},
    f'wz3p6_ee_mumuH_Hinv_ecm{ecm}':       {'fraction':fraction},

    # f'wzp6_ee_tautauH_Hbb_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_Hcc_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_Hss_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_Hgg_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_Haa_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_HZa_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_HWW_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_HZZ_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_tautauH_Hmumu_ecm{ecm}':     {'fraction':fraction},
    # f'wzp6_ee_tautauH_Htautau_ecm{ecm}':   {'fraction':fraction},
    # f'wz3p6_ee_tautauH_Hinv_ecm{ecm}':     {'fraction':fraction},


    # f'wzp6_ee_nunuH_Hbb_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_Hcc_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_Hss_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_Hgg_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_Haa_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_HZa_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_HWW_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_HZZ_ecm{ecm}':         {'fraction':fraction},
    # f'wzp6_ee_nunuH_Hmumu_ecm{ecm}':       {'fraction':fraction},
    # f'wzp6_ee_nunuH_Htautau_ecm{ecm}':     {'fraction':fraction},
    # f'wz3p6_ee_nunuH_Hinv_ecm{ecm}':       {'fraction':fraction},

}

processCutFlow = processListSignal | processListBkg