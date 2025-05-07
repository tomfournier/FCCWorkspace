import importlib

# Load userConfig 
userConfig = importlib.import_module("userConfig")

# Define final_state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# Input directory where the files produced at the pre-selection level will be put
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
  "sel_Baseline":"zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 &&zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA01":"BDTscore>0.1 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA02":"BDTscore>0.2 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA03":"BDTscore>0.3 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA04":"BDTscore>0.4 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA05":"BDTscore>0.5 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA06":"BDTscore>0.6 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA07":"BDTscore>0.7 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA08":"BDTscore>0.8 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  "sel_Baseline_MVA09":"BDTscore>0.9 && zll_m > 86 && zll_m < 96 && zll_recoil_m > 120 && zll_recoil_m < 140 && zll_p > 20 && zll_p < 70 && cosTheta_miss.size() >= 1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
            
  # APC3
  # "sel0_MRecoil_Mll_73_120_pll_05":"zll_m > 73 && zll_m  < 120 && zll_recoil_m > 120 &&zll_recoil_m <140 && zll_p > 5",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA01":"BDTscore>0.1 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5", 
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA02":"BDTscore>0.2 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5", 
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA03":"BDTscore>0.3 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5", 
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA04":"BDTscore>0.4 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5", 
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA05":"BDTscore>0.5 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA06":"BDTscore>0.6 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA07":"BDTscore>0.7 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA08":"BDTscore>0.8 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA09":"BDTscore>0.9 && zll_m>73 && zll_m<120 && zll_recoil_m>120 && zll_recoil_m<140 && zll_p>5",

  # APC3+costhetamissing
  # "sel0_MRecoil_Mll_73_120_pll_05_costhetamiss":"  zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m  <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA01_costhetamiss":"BDTscore>0.1 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA02_costhetamiss":"BDTscore>0.2 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA03_costhetamiss":"BDTscore>0.3 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA04_costhetamiss":"BDTscore>0.4 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;", 
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA05_costhetamiss":"BDTscore>0.5 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA06_costhetamiss":"BDTscore>0.6 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA07_costhetamiss":"BDTscore>0.7 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA08_costhetamiss":"BDTscore>0.8 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",
  # "sel0_MRecoil_Mll_73_120_pll_05_MVA09_costhetamiss":"BDTscore>0.9 &&   zll_m  > 73 &&  zll_m  < 120  && zll_recoil_m  > 120 &&zll_recoil_m <140 && zll_p  > 5&& cosTheta_miss.size() >=1 && cosTheta_miss[0]  > -0.98 && cosTheta_miss[0]  < 0.98;",

  # baseline without costhetamiss 
  "sel_Baseline_no_costhetamiss":"zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 && zll_recoil_m  < 140 && zll_p  > 20 && zll_p  <70",
  "sel_Baseline_no_costhetamiss_MVA01":"BDTscore > 0.1 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA02":"BDTscore > 0.2 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA03":"BDTscore > 0.3 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA04":"BDTscore > 0.4 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA05":"BDTscore > 0.5 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA06":"BDTscore > 0.6 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA07":"BDTscore > 0.7 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA08":"BDTscore > 0.8 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",
  "sel_Baseline_no_costhetamiss_MVA09":"BDTscore > 0.9 && zll_m  > 86 && zll_m  < 96  && zll_recoil_m > 120 &&zll_recoil_m  < 140 && zll_p  > 20 && zll_p  < 70",

  # Extrat
  # "sel_mll_86_96":"zll_m  > 86 && zll_m  < 96",

  # Figure 4 and 5
  # "sel_Baseline_without_mrec":"zll_m > 86 && zll_m < 96 && zll_p > 20 && zll_p <70 && cosTheta_miss.size() >=1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  # "sel_Baseline_without_mrec_1":"zll_m > 86 && zll_m < 96 && zll_p > 20 && zll_p <70",
  # "sel_Baseline_without_mrec_2":"zll_m > 86 && zll_m < 96",
  # "sel_Baseline_without_mrec_3":"zll_p > 20 && zll_p <70",
  # "sel_Baseline_without_mrec_4":"zll_m > 86 && zll_m < 96 && cosTheta_miss.size() >=1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98",
  # "sel_Baseline_without_mrec_5":"zll_m > 86 && zll_m < 96 && cosTheta_miss.size() >=1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98 && zll_p > 20",
  # "sel_Baseline_with_mrec":"zll_m > 86 && zll_m < 96 && zll_p > 20 && zll_p <70 && cosTheta_miss.size() >=1 && cosTheta_miss[0] > -0.98 && cosTheta_miss[0] < 0.98 && zll_recoil_m > 120 && zll_recoil_m  < 140",
             
}


# Dictionary for the ouput variable/hitograms. The key is the name of the variable in the output files. "name" is the name of the variable in the input file, "title" is the x-axis label of the histogram, "bin" the number of bins of the histogram, "xmin" the minimum x-axis value and "xmax" the maximum x-axis value.
histoList = {
    "mz":{"name":"zll_m","title":"m_{Z} [GeV]","bin":125,"xmin":86,"xmax":96},
    # "mz_zoom1":{"name":"zll_m","title":"m_{Z} [GeV]","bin":200,"xmin":80,"xmax":100},
    # "mz_zoom2":{"name":"zll_m","title":"m_{Z} [GeV]","bin":100,"xmin":86,"xmax":96},
    # "mz_zoom3":{"name":"zll_m","title":"m_{Z} [GeV]","bin":250,"xmin":75,"xmax":100},
    # "mz_zoom4":{"name":"zll_m","title":"m_{Z} [GeV]","bin":235,"xmin":73,"xmax":120},
    # "mz_zoom5":{"name":"zll_m","title":"m_{Z} [GeV]","bin":230,"xmin":40,"xmax":160},
    "recoil_m":{"name":"zll_recoil_m","title":"m_{recoil} [GeV]","bin":100,"xmin":120,"xmax":140},
    # "leptonic_recoil_m_zoom1":{"name":"zll_recoil_m","title":"Z leptonic recoil [GeV]","bin":200,"xmin":80,"xmax":160},
    # "leptonic_recoil_m_zoom2":{"name":"zll_recoil_m","title":"Z leptonic recoil [GeV]","bin":100,"xmin":120,"xmax":140},
    # "leptonic_recoil_m_zoom3":{"name":"zll_recoil_m","title":"Z leptonic recoil [GeV]","bin":200,"xmin":120,"xmax":140},
    # "leptonic_recoil_m_zoom4":{"name":"zll_recoil_m","title":"Z leptonic recoil [GeV]","bin":70,"xmin":123,"xmax":130},
    # "leptonic_recoil_m_zoom5":{"name":"zll_recoil_m","title":"Z leptonic recoil [GeV]","bin":20,"xmin":124,"xmax":126}, 
    # "leptonic_recoil_m_zoom6":{"name":"zll_recoil_m","title":"Z leptonic recoil [GeV]","bin":200,"xmin":110,"xmax":150}, 
    # "leptonic_recoil_m_zoom7":{"name":"zll_recoil_m","title":"Z leptonic recoil [GeV]","bin":240,"xmin":40,"xmax":160}, 
    "BDT_Score":{"name":"BDTscore","title":"BDT Score","bin":100,"xmin":0,"xmax":1},
    # "BDT_Score_scaleup":{"name":"BDTscore_scaleup","title":"BDT Score LEPSCALE UP","bin":100,"xmin":0,"xmax":1}, 
    # "BDT_Score_scaledw":{"name":"BDTscore_scaledw","title":"BDT Score LEPSCALE DOWN","bin":100,"xmin":0,"xmax":1}, 
    # more control variables
    "acolinearity":{"name":"acolinearity","title":"#Delta#theta_{l^{+}l^{-}}","bin":100,"xmin":0,"xmax":3.2},
    "acoplanarity":{"name":"acoplanarity","title":"#Delta#phi_{l^{+}l^{-}}","bin":100,"xmin":0,"xmax":3.2},
    "cosTheta_miss":{"name":"cosTheta_miss","title":"|cos#theta_{miss}|","bin":200,"xmin":0,"xmax":1},
    # plot fundamental variables:
    "leading_p":{"name":"leading_p","title":"p_{l,leading}","bin":200,"xmin":45,"xmax":85},
    "leading_theta":{"name":"leading_theta","title":"#theta_{l,leading}","bin":70,"xmin":0,"xmax":3.2},
    "leading_phi":{"name":"leading_phi","title":"#phi_{l,leading}","bin":70,"xmin":-3.2,"xmax":3.2},
    "subleading_p":{"name":"subleading_p","title":"p_{l,subleading}","bin":200,"xmin":20,"xmax":55},
    "subleading_theta":{"name":"subleading_theta","title":"#theta_{l,subleading}","bin":70,"xmin":0,"xmax":3.2},
    "subleading_phi":{"name":"subleading_phi","title":"#phi_{l,subleading}","bin":70,"xmin":-3.2,"xmax":3.2},
    # Zed
    "zll_p":{"name":"zll_p","title":"p_{l^{+}l^{-}} [GeV]","bin":200,"xmin":20,"xmax":65},
    "zll_theta":{"name":"zll_theta","title":"#theta_{l^{+}l^{-}}","bin":70,"xmin":0,"xmax":3.2},
    "zll_phi":{"name":"zll_phi","title":"#phi_{l^{+}l^{-}}","bin":70,"xmin":-3.2,"xmax":3.2},
    # Higgsstrahlungness
    # "H":{"name":"H","title":"Higgsstrahlungness","bin":200,"xmin":0,"xmax":200},
    # number of leptons
    # "leps_no":{"name":"leps_no","title":"number of leptons","bin":6,"xmin":-0.5,"xmax":5.5},
}
