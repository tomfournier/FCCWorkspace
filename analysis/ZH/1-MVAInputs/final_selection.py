import importlib

# Load userConfig 
userConfig = importlib.import_module("userConfig")

# Input directory where the files produced at the pre-selection level are
inputDir = userConfig.loc.TRAIN

# Output directory where the files produced at the final selection level will be put
outputDir = userConfig.loc.FINAL

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = "FCCee_procDict_winter2023_IDEA.json"

# Process list that should match the produced files.
processList = userConfig.processList

# Number of CPUs to use
nCPUS = 2

# produces ROOT TTrees, default is False
doTree = False

# scale the histograms with the cross-section and integrated luminosity
# doScale = True
# intLumi = userConfig.intLumi # in ab-1

# saveTabular = True

# Dictionnay of the list of cuts. The key is the name of the selection that will be added to the output file
cutList = { 
  # baseline without costhetamiss 
  "sel_Baseline_no_costhetamiss":"zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 && zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  # baseline with costhetamiss
  "sel_Baseline_costhetamiss":"zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 && zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] < 0.995 && cosTheta_miss[0] > -0.995",
}

# Dictionary for the ouput variable/hitograms. 
# The key is the name of the variable in the output files. 
# "name" is the name of the variable in the input file, 
# "title" is the x-axis label of the histogram, 
# "bin" the number of bins of the histogram, 
# "xmin" the minimum x-axis value and 
# "xmax" the maximum x-axis value.
histoList = {
    # plot fundamental variables:
    "leading_p":{"name":"leading_p","title":"p_{l,leading} [GeV]","bin":100,"xmin":45,"xmax":85},
    "leading_theta":{"name":"leading_theta","title":"#theta_{l,leading}","bin":100,"xmin":0,"xmax":3.2},
    "subleading_p":{"name":"subleading_p","title":"p_{l,subleading} [GeV]","bin":100,"xmin":20,"xmax":60},
    "subleading_theta":{"name":"subleading_theta","title":"#theta_{l,subleading}","bin":100,"xmin":0,"xmax":3.2},
    
    # Zed
    "zll_m":{"name":"zll_m","title":"m_{l^{+}l^{-}} [GeV]","bin":100,"xmin":86,"xmax":96},
    "zll_p":{"name":"zll_p","title":"p_{l^{+}l^{-}} [GeV]","bin":100,"xmin":20,"xmax":70},
    "zll_theta":{"name":"zll_theta","title":"#theta_{l^{+}l^{-}}","bin":100,"xmin":0,"xmax":3.2},
    "zll_phi":{"name":"zll_phi","title":"#phi_{l^{+}l^{-}}","bin":100,"xmin":-3.2,"xmax":3.2},
    
    # more control variables
    "acolinearity":{"name":"acolinearity","title":"#Delta#theta_{l^{+}l^{-}}","bin":100,"xmin":0,"xmax":3.2},
    "acoplanarity":{"name":"acoplanarity","title":"#Delta#phi_{l^{+}l^{-}}","bin":100,"xmin":0,"xmax":3.2},
    
    # Recoil
    "zll_recoil_m":{"name":"zll_recoil_m","title":"m_{recoil} [GeV]","bin":100,"xmin":120,"xmax":140},
    
    # missing Information
    "cosTheta_miss":{"name":"cosTheta_miss","title":"|cos#theta_{miss}|","bin":100,"xmin":0,"xmax":1},
    
    # Higgsstrahlungness
    "H":{"name":"H","title":"Higgsstrahlungness","bin":110,"xmin":0,"xmax":110} 
}
