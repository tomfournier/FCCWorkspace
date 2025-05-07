import ROOT
import importlib

# Load userConfig 
userConfig = importlib.import_module("userConfig") 

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# global parameters
intLumi         = userConfig.intLumi * 1e6 #in pb-1
if final_state == 'mumu':
        ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
else:
        ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
delphesVersion  = '3.4.2'
energy          = ecm
collider        = 'FCC-ee'
inputDir        = userConfig.loc.ANALYSIS_FINAL
yaxis           = ['lin','log']
stacksig        = ['nostack']
formats         = ['png']
outdir          = userConfig.loc.PLOTS_Val

variables = [
        # lepton
        "mz",
        # "mz_zoom1", "mz_zoom2",
        # "mz_zoom3", "mz_zoom4",
        "recoil_m", # "leptonic_recoil_m_zoom1", 
        # "leptonic_recoil_m_zoom2", "leptonic_recoil_m_zoom3", 
        # "leptonic_recoil_m_zoom4", "leptonic_recoil_m_zoom5",
        # "leptonic_recoil_m_zoom6", "leptonic_recoil_m_zoom7",
        "BDT_Score",
        # control vaiables
        "acolinearity", "acoplanarity", "cosTheta_miss",
        # plot fundamental varibales:
        "leading_p", "leading_theta",
        "leading_phi", "subleading_p",
        "subleading_theta", "subleading_phi",
        # Zed
        "zll_p", "zll_theta", "zll_phi",
        # Higgsstralungness
        # "H",
        # "leps_no"
]

#cDictonnary with the analysis name as a key, and the list of selections to be plotted for this analysis. The name of the selections should be the same than in the final selection
selections = {}
selections['ZH'] = [ 
        # "sel0", 
        "sel_Baseline", 
        "sel_Baseline_MVA01",
        "sel_Baseline_MVA02", 
        "sel_Baseline_MVA03",
        "sel_Baseline_MVA04",
        "sel_Baseline_MVA05", 
        "sel_Baseline_MVA06",
        "sel_Baseline_MVA07",
        "sel_Baseline_MVA08", 
        "sel_Baseline_MVA09",
        # "sel0_MRecoil_Mll_73_120_pll_05",                    
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA01",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA02",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA03",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA04",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA05",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA06",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA07",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA08",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA09",
        # "sel0_MRecoil_Mll_73_120_pll_05_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA01_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA02_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA03_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA04_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA05_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA06_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA07_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA08_costhetamiss",
        # "sel0_MRecoil_Mll_73_120_pll_05_MVA09_costhetamiss",
        "sel_Baseline_no_costhetamiss",
        "sel_Baseline_no_costhetamiss_MVA01",
        "sel_Baseline_no_costhetamiss_MVA02",
        "sel_Baseline_no_costhetamiss_MVA03",
        "sel_Baseline_no_costhetamiss_MVA04",
        "sel_Baseline_no_costhetamiss_MVA05",
        "sel_Baseline_no_costhetamiss_MVA06",
        "sel_Baseline_no_costhetamiss_MVA07",
        "sel_Baseline_no_costhetamiss_MVA08",
        "sel_Baseline_no_costhetamiss_MVA09",
        # "sel_mll_86_96",
        "sel_Baseline_without_mrec",
        "sel_Baseline_without_mrec_1",
        "sel_Baseline_without_mrec_2",
        "sel_Baseline_without_mrec_3",
        "sel_Baseline_without_mrec_4",
        "sel_Baseline_without_mrec_5",
        "sel_Baseline_with_mrec",
]

extralabel = {}
extralabel['sel0']                                              = "Selection: No Selection"
extralabel["sel_MVA01"]                                         = "Baseline_MVA01"
extralabel["sel_MVA02"]                                         = "Baseline_MVA02"
extralabel["sel_MVA03"]                                         = "Baseline_MVA03"
extralabel["sel_MVA04"]                                         = "Baseline_MVA04"
extralabel["sel_MVA05"]                                         = "Baseline_MVA05"
extralabel['sel_Baseline']                                      = "Baseline"
extralabel["sel_Baseline_MVA01"]                                = "Baseline_MVA01"
extralabel["sel_Baseline_MVA02"]                                = "Baseline_MVA02"
extralabel["sel_Baseline_MVA03"]                                = "Baseline_MVA03"
extralabel["sel_Baseline_MVA04"]                                = "Baseline_MVA04"
extralabel["sel_Baseline_MVA05"]                                = "Baseline_MVA05"
extralabel["sel_Baseline_MVA06"]                                = "Baseline_MVA06"
extralabel["sel_Baseline_MVA07"]                                = "Baseline_MVA07"
extralabel["sel_Baseline_MVA08"]                                = "Baseline_MVA08"
extralabel["sel_Baseline_MVA09"]                                = "Baseline_MVA09"
extralabel['sel_APC1']                                          = 'APC1'
extralabel['sel_APC1_MVA02']                                    = "APC1_MVA02"
extralabel['sel_APC1_MVA06']                                    = "APC1_MVA06"
extralabel['sel_APC1_MVA02_mll_80_100']                         = "APC1_MVA02_mll_80_100"
extralabel['sel_APC1_MVA02_mll_75_100']                         = "APC1_MVA02_mll_75_100"
extralabel['sel_APC1_MVA02_mll_73_120']                         = "APC1_MVA02_mll_73_120"
extralabel['sel_APC1_MVA02_mll_80_100_nopT']                    = "APC1_MVA02_mll_80_100_nopT"
extralabel['sel_APC1_MVA02_mll_80_100_pT20']                    = "APC1_MVA02_mll_80_100_pT20"
extralabel['sel_APC1_MVA02_mll_80_100_pT10']                    = "APC1_MVA02_mll_80_100_pT10"
extralabel["sel0_MRecoil"]                                      = "sel0_MRecoil"
extralabel["sel0_MRecoil_MVA02"]                                = "sel0_MRecoil_MVA02"
extralabel["sel0_MRecoil_Mll"]                                  = "sel0_MRecoil_Mll"
extralabel["sel0_MRecoil_Mll_MVA02"]                            = "sel0_MRecoil_Mll_MVA02"
extralabel["sel0_MRecoil_pll"]                                  = "sel0_MRecoil_pll"
extralabel["sel0_MRecoil_pll_MVA02"]                            = "sel0_MRecoil_pll_MVA02"
extralabel["sel0_Mll"]                                          = "sel0_Mll"
extralabel["sel0_Mll_MVA02"]                                    = "sel0_Mll_MVA02"
extralabel["sel0_pll"]                                          = "sel0_pll"
extralabel["sel0_pll_MVA02"]                                    = "sel0_pll_MVA02"
extralabel["sel0_MRecoil_Mll_80_100"]                           = "sel0_MRecoil_Mll_80_100"
extralabel["sel0_MRecoil_Mll_75_100"]                           = "sel0_MRecoil_Mll_75_100"
extralabel["sel0_MRecoil_Mll_73_120"]                           = "sel0_MRecoil_Mll_73_120"
extralabel["sel0_MRecoil_pll_20"]                               = "sel0_MRecoil_pll_20"
extralabel["sel0_MRecoil_pll_15"]                               = "sel0_MRecoil_pll_15"
extralabel["sel0_MRecoil_pll_10"]                               = "sel0_MRecoil_pll_10"
extralabel["sel0_MRecoil_pll_05"]                               = "sel0_MRecoil_pll_05"
extralabel["sel0_MRecoil_Mll_73_120_pll_05"]                    = "sel0_MRecoil_Mll_73_120_pll_05"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA01"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA01"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA02"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA02"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA03"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA03" 
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA04"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA04"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA05"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA05"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA06"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA06"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA07"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA07"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA08"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA08"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA09"]              = "sel0_MRecoil_Mll_73_120_pll_05_MVA09"
extralabel["sel_MVA02_costhetamiss"]                            = "sel_MVA02_costhetamiss" 
extralabel["sel_MVA04_costhetamiss"]                            = "sel_MVA04_costhetamiss"
extralabel["sel_MVA06_costhetamiss"]                            = "sel_MVA06_costhetamiss"
extralabel["sel_MVA08_costhetamiss"]                            = "sel_MVA08_costhetamiss"
extralabel["sel_MVA09_costhetamiss"]                            = "sel_MVA09_costhetamiss"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_costhetamiss"]       = "sel0_MRecoil_Mll_73_120_pll_05_costhetamiss" 
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA01_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA01_costhetamiss" 
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA02_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA02_costhetamiss" 
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA03_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA03_costhetamiss" 
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA04_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA04_costhetamiss" 
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA05_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA05_costhetamiss" 
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA06_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA06_costhetamiss"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA07_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA07_costhetamiss"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA08_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA08_costhetamiss"
extralabel["sel0_MRecoil_Mll_73_120_pll_05_MVA09_costhetamiss"] = "sel0_MRecoil_Mll_73_120_pll_05_MVA09_costhetamiss"
extralabel["sel_Baseline_no_costhetamiss"]                      = "sel_Baseline_no_costhetamiss"   
extralabel["sel_Baseline_no_costhetamiss_MVA01"]                = "sel_Baseline_no_costhetamiss_MVA01"
extralabel["sel_Baseline_no_costhetamiss_MVA02"]                = "sel_Baseline_no_costhetamiss_MVA02"
extralabel["sel_Baseline_no_costhetamiss_MVA03"]                = "sel_Baseline_no_costhetamiss_MVA03"
extralabel["sel_Baseline_no_costhetamiss_MVA04"]                = "sel_Baseline_no_costhetamiss_MVA04"
extralabel["sel_Baseline_no_costhetamiss_MVA05"]                = "sel_Baseline_no_costhetamiss_MVA05"
extralabel["sel_Baseline_no_costhetamiss_MVA06"]                = "sel_Baseline_no_costhetamiss_MVA06"
extralabel["sel_Baseline_no_costhetamiss_MVA07"]                = "sel_Baseline_no_costhetamiss_MVA07"
extralabel["sel_Baseline_no_costhetamiss_MVA08"]                = "sel_Baseline_no_costhetamiss_MVA08"
extralabel["sel_Baseline_no_costhetamiss_MVA09"]                = "sel_Baseline_no_costhetamiss_MVA09"
extralabel["sel_mll_86_96"]                                     = " 86 < m_{l^-l^+} < 96"
extralabel["sel_Baseline_without_mrec"]                         =  "sel_Baseline_without_mrec"
extralabel["sel_Baseline_without_mrec_1"]                       =  "sel_Baseline_without_mrec_1"
extralabel["sel_Baseline_without_mrec_2"]                       =  "sel_Baseline_without_mrec_2"
extralabel["sel_Baseline_without_mrec_3"]                       =  "sel_Baseline_without_mrec_3"
extralabel["sel_Baseline_without_mrec_4"]                       =  "sel_Baseline_without_mrec_4"
extralabel["sel_Baseline_without_mrec_5"]                       =  "sel_Baseline_without_mrec_5"
extralabel["sel_Baseline_with_mrec"]                            =  "sel_Baseline_with_mrec"

colors = {}
colors['mumuH']      = ROOT.kRed
colors['tautauH']    = ROOT.kMagenta
colors['nunuH']      = ROOT.kOrange
colors['eeH']        = ROOT.kRed
colors['qqH']        = ROOT.kSpring
colors['WWmumu']     = ROOT.kBlue+1
colors['ZZ']         = ROOT.kGreen+2
colors['Zqq']        = ROOT.kYellow+2
colors['Zll']        = ROOT.kCyan
colors['eeZ']        = ROOT.kSpring+10
colors['gagatautau'] = ROOT.kViolet+7
colors['gagamumu']   = ROOT.kBlue-8
colors['ZH']         = ROOT.kRed
colors['WW']         = ROOT.kBlue+1
colors['VV']         = ROOT.kGreen+3
colors['rare']       = ROOT.kSpring

if final_state=="mumu":
        ee_ll = f'wzp6_ee_{final_state}_ecm{ecm}'
elif final_state=="ee":
        ee_ll = f'wzp6_ee_{final_state}_Mee_30_150_ecm{ecm}'

plots = {}
plots['ZH'] = {'signal':{f'{final_state}H':[f"wzp6_ee_{final_state}H_ecm{ecm}"]},
               'backgrounds':{'eeZ':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}", 
                                     f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}"],
                              'WW':[f'p8_ee_WW_ecm{ecm}'],
                              'Zll':[ee_ll], 'ZZ':[f'p8_ee_ZZ_ecm{ecm}'],
                              'rare':[f"wzp6_ee_tautau_ecm{ecm}", f"wzp6_gaga_{final_state}_60_ecm{ecm}",
                                      f"wzp6_gaga_tautau_60_ecm{ecm}", f"wzp6_ee_nuenueZ_ecm{ecm}"]
        }
}

legend = {}
legend['mumuH']      = 'Z(#mu^{-}#mu^{+})H'
legend['tautauH']    = 'Z(#tau^{-}#tau^{+})H'
legend['qqH']        = 'Z(q#bar{q})H'
legend['eeH']        = 'Z(e^{-}e^{+})H'
legend['nunuH']      = 'Z(#nu#bar{#nu})H'
legend['Zqq']        = 'Z#rightarrow q#bar{q}'
legend['Zll']        = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['eeZ']        = 'e^{+}(e^{-})Z'
legend['Wmumu']      = 'W^{+}(#bar{#nu}#mu^{+})W^{-}(#nu#mu^{-})'
legend['gagamumu']   = '#gamma#gamma#mu^{-}#mu^{+}'
legend['gagatautau'] = '#gamma#gamma#tau^{-}#tau^{+}'
legend['ZH']         = 'ZH'
legend['WW']         = 'W^{+}W^{-}'
legend['ZZ']         = 'ZZ'
legend['VV']         = 'VV boson'
legend['rare']       = 'Rare'
