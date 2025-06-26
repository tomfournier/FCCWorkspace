import importlib

# Load userConfig 
userConfig = importlib.import_module("userConfig")
from userConfig import loc, get_loc, ecm, sel, lumi, param

final_state = input('Select a channel [ee, mumu]: ')

# Input directory where the files produced at the pre-selection level are
inputDir  = get_loc(loc.MVA_INPUTS, final_state, ecm, sel)
# Output directory where the files produced at the final selection level will be put
outputDir = get_loc(loc.HIST_MVA,   final_state, ecm, sel)

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = "FCCee_procDict_winter2023_training_IDEA.json"
# procDictAdd = userConfig.procDictAdd

# Number of CPUs to use
nCPUS = 20

# produces ROOT TTrees, default is False
doTree = False

# Scale to integrated luminosity
doScale = True
intLumi = lumi * 1e6 # in pb-1

# Process list that should match the produced files.
ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if final_state=='ee' else f"wzp6_ee_mumu_ecm{ecm}"

# Process samples for BDT
samples_BDT = [
    #signal
    f"wzp6_ee_{final_state}H_ecm{ecm}",

    #background: 
    f"p8_ee_ZZ_ecm{ecm}", f"p8_ee_WW_{final_state}_ecm{ecm}", ee_ll,

    #rare backgrounds:
    f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}", f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
    f"wzp6_gaga_{final_state}_60_ecm{ecm}"
]

# Mandatory: List of processes
processList = {i:param for i in samples_BDT}

# Dictionnay of the list of cuts. The key is the name of the selection that will be added to the output file
if userConfig.recoil120:
    bin, xmin, xmax = 80, 120, 140
    recoil_dw, recoil_up = 120, 140
else:
    bin, xmin, xmax = 200, 100, 150
    recoil_dw, recoil_up = 100, 150

baselineCut = f"zll_p > 20 && zll_p < 70 && zll_m > 86 && zll_m < 96 && zll_recoil_m > {recoil_dw} && zll_recoil_m < {recoil_up}"
cosTheta_missCut = "cosTheta_miss < 0.98"
leading_pCut = "leading_p < 80 && leading_p > 50"

selection = baselineCut + ' && ' + cosTheta_missCut if userConfig.miss else baselineCut
if userConfig.leading:
    selection += '&&' + leading_pCut


cutList = { sel: selection }

# Dictionary for the ouput variable/hitograms. 
histoList = {
    "leading_p":        {"name":"leading_p",
                         "title":"p_{l,leading} [GeV]",
                         "bin":120, "xmin":40, "xmax":100},

    "leading_theta":    {"name":"leading_theta",
                         "title":"#theta_{l,leading}",
                         "bin":128, "xmin":0,  "xmax":3.2},

    "subleading_p":     {"name":"subleading_p",
                         "title":"p_{l,subleading} [GeV]",
                         "bin":80, "xmin":20, "xmax":60},

    "subleading_theta": {"name":"subleading_theta",
                         "title":"#theta_{l,subleading}",
                         "bin":128, "xmin":0,  "xmax":3.2},
    
    # Zed
    "zll_m":            {"name":"zll_m",
                         "title":"m_{l^{+}l^{-}} [GeV]",
                         "bin":100,"xmin":86,"xmax":96},

    "zll_p":            {"name":"zll_p",
                         "title":"p_{l^{+}l^{-}} [GeV]",
                         "bin":100,"xmin":20,"xmax":70},

    "zll_theta":        {"name":"zll_theta",
                         "title":"#theta_{l^{+}l^{-}}",
                         "bin":128,"xmin":0,"xmax":3.2},

    "zll_phi":          {"name":"zll_phi",
                         "title":"#phi_{l^{+}l^{-}}",
                         "bin":64,"xmin":-3.2,"xmax":3.2},
    
    "acolinearity":     {"name":"acolinearity",
                         "title":"#Delta#theta_{l^{+}l^{-}}",
                         "bin":120,"xmin":0,"xmax":3},

    "acoplanarity":     {"name":"acoplanarity",
                         "title":"#Delta#phi_{l^{+}l^{-}}",
                         "bin":128,"xmin":0,"xmax":3.2},
    
    "zll_recoil_m":     {"name":"zll_recoil_m",
                         "title":"m_{recoil} [GeV]",
                         "bin":bin,"xmin":xmin,"xmax":xmax},
    
    "cosTheta_miss":    {"name":"cosTheta_miss",
                         "title":"|cos#theta_{miss}|",
                         "bin":100,"xmin":0.9,"xmax":1},
    
    "H":                {"name":"H",
                         "title":"Higgsstrahlungness",
                         "bin":110,"xmin":0,"xmax":110},
}
