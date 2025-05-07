import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# Input directory where the files produced at the pre-selection level are
inputDir = userConfig.loc.ANALYSIS

# Input directory where the files produced at the pre-selection level are
outputDir = userConfig.loc.ANALYSIS_FINAL

# Link to the dictonary that contains all the cross section informations etc...
procDict = "FCCee_procDict_winter2023_IDEA.json"

# Process list that should match the produced files.
processList = userConfig.processList1

# Number of CPUs to use
nCPUS = 2

# produces ROOT TTrees, default is False
doTree = False

# Dictionnay of the list of cuts. The key is the name of the selection that will be added to the output file
cutList = { 
    # "sel0":"return true;",
    "sel_Baseline":"zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
    "sel_Baseline_no_costhetamiss":"zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 &&zll_recoil_m <140 && zll_p > 20 && zll_p < 70", 
    # "sel_Baseline_no_costhetamiss_scaleup":"zll_m_scaleup > 86 && zll_m_scaleup < 96 && zll_recoil_m_scaleup > 120 &&zll_recoil_m_scaleup < 140 && zll_p_scaleup > 20 && zll_p_scaleup < 70", 
    # "sel_Baseline_no_costhetamiss_scaledw":"zll_m_scaledw > 86 && zll_m_scaledw < 96 && zll_recoil_m_scaledw > 120 &&zll_recoil_m_scaledw < 140 && zll_p_scaledw > 20 && zll_p_scaledw < 70",
    # "sel_Baseline_no_costhetamiss_besup":"zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 &&zll_recoil_m < 140 && zll_p > 20 && zll_p < 70", 
    # "sel_Baseline_no_costhetamiss_besdw":"zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m <140 && zll_p > 20 && zll_p < 70",
    # "sel_Baseline_no_costhetamiss_sqrtsup":"zll_m > 86 && zll_m < 96 && zll_recoil_m_sqrtsup > 120 &&zll_recoil_m_sqrtsup <140 && zll_p > 20 && zll_p < 70",
    # "sel_Baseline_no_costhetamiss_sqrtsdw":"zll_m > 86 && zll_m < 96 && zll_recoil_m_sqrtsdw > 120 &&zll_recoil_m_sqrtsdw <140 && zll_p > 20 && zll_p < 70",
}

# Dictionary for the ouput variable/hitograms. 
# The key is the name of the variable in the output files. 
# "name" is the name of the variable in the input file, 
# "title" is the x-axis label of the histogram, 
# "bin" the number of bins of the histogram, 
# "xmin" the minimum x-axis value and 
# "xmax" the maximum x-axis value.
histoList = {
    "BDT_Score":{"name":"BDTscore","title":"BDT Score","bin":100,"xmin":0,"xmax":1},
    # "BDT_Score_scaleup":{"name":"BDTscore_scaleup","title":"BDT Score LEPSCALE UP","bin":100,"xmin":0,"xmax":1}, 
    # "BDT_Score_scaledw":{"name":"BDTscore_scaledw","title":"BDT Score LEPSCALE DOWN","bin":100,"xmin":0,"xmax":1}, 
}
