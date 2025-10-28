import ROOT, importlib

# Load userConfig 
userConfig = importlib.import_module("userConfig")
from userConfig import loc, get_loc, event, ecm, lumi, param, z_decays, h_decays, train_vars, frac, nb

cat = input('Select a channel [ee, mumu]: ')

# Input directory where the files produced at the pre-selection level are
inputDir  = get_loc(loc.EVENTS,            cat, ecm, '')
# Output directory where the files produced at the final selection level will be put
outputDir = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = "FCCee_procDict_winter2023_IDEA.json"
# procDictAdd = userConfig.procDictAdd

# Number of CPUs to use
nCPUS = 10

# produces ROOT TTrees, default is False
# doTree = True
doTree = False

# Scale to integrated luminosity
doScale = True
intLumi = lumi * 1e6 # in pb-1

# Process samples
samples_bkg = [
    f"p8_ee_ZZ_ecm{ecm}",
    f"p8_ee_WW_ecm{ecm}",              f"p8_ee_WW_ee_ecm{ecm}",       f"p8_ee_WW_mumu_ecm{ecm}",
    f"wzp6_ee_ee_Mee_30_150_ecm{ecm}", f"wzp6_ee_mumu_ecm{ecm}",      f"wzp6_ee_tautau_ecm{ecm}",
    f"wzp6_gaga_ee_60_ecm{ecm}",       f"wzp6_gaga_mumu_60_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}", 
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',  f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}',    f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f"wzp6_ee_nuenueZ_ecm{ecm}"
]
samples_sig = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]
samples_sig.extend([f"wzp6_ee_eeH_ecm{ecm}", f"wzp6_ee_mumuH_ecm{ecm}", f'wzp6_ee_ZH_Hinv_ecm{ecm}'])

samples = event(samples_sig + samples_bkg, inputDir+'/')
big_sample = [
    f'p8_ee_ZZ_ecm{ecm}', f'p8_ee_WW_ecm{ecm}', f'p8_ee_WW_{cat}_ecm{ecm}',
    f'wzp6_ee_mumu_ecm{ecm}' if cat=='mumu' else f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}', f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gaga_{cat}_60_ecm{ecm}'
]
processList = {i:{'fraction': frac, 'chunks': nb if i in big_sample else 1}  for i in samples}
processList = {i:param for i in samples}



sel_BDT = 'Baseline'
ROOT.gInterpreter.ProcessLine(f'''
  TMVA::Experimental::RBDT<> tmva("ZH_Recoil_BDT", "{get_loc(loc.BDT, cat, ecm, sel_BDT)}/xgb_bdt.root");
''')
var_list = ', (float)'.join(train_vars)
defineList = {
    'MVAVec':    f'ROOT::VecOps::RVec<float>{{{var_list}}}',
    'mva_score': 'tmva.Compute(MVAVec)', 'BDTscore':  'mva_score.at(0)'
}


bdt = 0.91 if cat=='mumu' else 0.92
Baseline_Cut = 'zll_m > 86 && zll_m < 96 && zll_p > 20 && zll_p < 70 && zll_recoil_m > 100 && zll_recoil_m < 150'
cutList = { 
    # 'sel0':              'return true;',
    'Baseline':          Baseline_Cut,
    'Baseline_vis':      Baseline_Cut +  ' && visibleEnergy > 100',
    'Baseline_inv':      Baseline_Cut +  ' && visibleEnergy < 100',
    'Baseline_high':     Baseline_Cut + f' && BDTscore > {bdt}',
    'Baseline_low':      Baseline_Cut + f' && BDTscore < {bdt}'
}



# Dictionary for the ouput variable/histograms. 
histoList = {

    # Leptons information
    "leading_p":        {"name":"leading_p",
                         "title":"p_{l,leading} [GeV]",
                         "bin":120, "xmin":40, "xmax":100},

    "leading_theta":    {"name":"leading_theta",
                         "title":"#theta_{l,leading}",
                         "bin":128, "xmin":0,  "xmax":3.2},

    "leading_phi":      {"name":"leading_phi",
                         "title":"#phi_{l,leading}",
                         "bin":64,"xmin":-3.2,"xmax":3.2},

    "subleading_p":     {"name":"subleading_p",
                         "title":"p_{l,subleading} [GeV]",
                         "bin":80, "xmin":20, "xmax":60},

    "subleading_theta": {"name":"subleading_theta",
                         "title":"#theta_{l,subleading}",
                         "bin":128, "xmin":0,  "xmax":3.2},

    "subleading_phi":   {"name":"subleading_phi",
                         "title":"#phi_{l,subleading}",
                         "bin":64,"xmin":-3.2,"xmax":3.2},

    # Angular separation information
    "acolinearity":     {"name":"acolinearity",
                         "title":"#Delta#theta_{l^{+}l^{-}}",
                         "bin":120,"xmin":0,"xmax":3},

    "acoplanarity":     {"name":"acoplanarity",
                         "title":"#Delta#phi_{l^{+}l^{-}}",
                         "bin":128,"xmin":0,"xmax":3.2},
    
    "zll_deltaR":       {"name":"zll_deltaR",
                         "title":"#DeltaR",
                         "bin":100,"xmin":0,"xmax":10},
    
    # Zed information
    "zll_m":            {"name":"zll_m",
                         "title":"m_{l^{+}l^{-}} [GeV]",
                         "bin":100,"xmin":86,"xmax":96},

    "zll_p":            {"name":"zll_p",
                         "title":"p_{l^{+}l^{-}} [GeV]",
                         "bin":500,"xmin":20,"xmax":70},

    "zll_theta":        {"name":"zll_theta",
                         "title":"#theta_{l^{+}l^{-}}",
                         "bin":128,"xmin":0,"xmax":3.2},

    "zll_phi":          {"name":"zll_phi",
                         "title":"#phi_{l^{+}l^{-}}",
                         "bin":64,"xmin":-3.2,"xmax":3.2},
    
    "zll_recoil_m":     {"name":"zll_recoil_m",
                         "title":"m_{recoil} [GeV]",
                         "bin":100,"xmin":100,"xmax":150},
    
    # Visible and invisible information
    "cosTheta_miss":    {"name":"cosTheta_miss",
                         "title":"|cos#theta_{miss}|",
                         "bin":100,"xmin":0.9,"xmax":1},
    
    "visibleEnergy":    {"name":"visibleEnergy",
                         "title":"E_{vis} [GeV]",
                         "bin":320,"xmin":0,"xmax":160},

    "missingMass":      {"name":"missingMass",
                         "title":"m_{miss} [GeV]",
                         "bin":500,"xmin":0,"xmax":250},
    
    # Higgsstrahlungness
    "H":                {"name":"H",
                         "title":"Higgsstrahlungness",
                         "bin":110,"xmin":0,"xmax":110},

    # BDT score
    "BDTscore":         {"name":"BDTscore",
                         "title":"BDT score",
                         "bin":500,"xmin":0,"xmax":1}
                         
}
