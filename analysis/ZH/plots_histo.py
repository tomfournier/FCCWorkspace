import ROOT

# global parameters
intLumi        = 7.2e+06 #in pb-1
ana_tex        = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
inputDir       = '/eos/user/t/tofourni/public/FCC/FCCWorkplace/analysis/ZH_XSec/MidTerm/mumu/output/final/'
yaxis          = ['lin','log']
stacksig       = ['stack']
formats        = ['pdf']
outdir         = '/eos/user/t/tofourni/public/FCC/FCCWorkplace/analysis/ZH_XSec/MidTerm/mumu/output/plots/'

variables = [   #muons
                "leading_zll_lepton_p",
                "leading_zll_lepton_theta",
                "subleading_zll_lepton_p",
                "subleading_zll_lepton_theta",
                #Zed
                "zll_m",
                "zll_p",
                "zll_theta",
                #more control variables
                "zll_leptons_acolinearity",
                "zll_leptons_acoplanarity",
                #Recoil
                "zll_recoil_m",
                #missing Information
                "cosTheta_miss",
                #Higgsstrahlungness
                "H"
               ]
###Dictonnary with the analysis name as a key, and the list of selections to be plotted for this analysis. The name of the selections should be the same than in the final selection
selections = {}
selections['ZH'] = [ "sel_Baseline_no_costhetamiss",] 
                    # "sel_Baseline_costhetamiss"]

extralabel = {}
extralabel["sel_Baseline_no_costhetamiss"] = "Baseline without cos#theta_{miss} cut"
extralabel["sel_Baseline_costhetamiss"] = "Baseline with cos#theta_{miss} cut"   

colors = {}
colors['mumuH'] = ROOT.kRed
colors['eeH'] = ROOT.kRed
colors['Zmumu'] = ROOT.kCyan
colors['Zee'] = ROOT.kCyan
colors['eeZ'] = ROOT.kSpring+10
colors['WWmumu'] = ROOT.kBlue+1
colors['WWee'] = ROOT.kBlue+1
colors['gagamumu'] = ROOT.kBlue-8
colors['gagaee'] = ROOT.kBlue-8
colors['WW'] = ROOT.kBlue+1
colors['ZZ'] = ROOT.kGreen+2

plots = {}
plots['ZH'] = {'signal':{'mumuH':['wzp6_ee_mumuH_ecm240']},
               'backgrounds':{
                                'WWmumu':['p8_ee_WW_mumu_ecm240'],
                                'ZZ':['p8_ee_ZZ_ecm240'],
                                'Zmumu':['wzp6_ee_mumu_ecm240'],
                                'eeZ':["wzp6_egamma_eZ_Zmumu_ecm240",
                                     "wzp6_gammae_eZ_Zmumu_ecm240"],
                                'gagamumu':["wzp6_gaga_mumu_60_ecm240"]
                                }
              }

legend = {}
legend['mumuH'] = 'Z(#mu^{-}#mu^{+})H'
legend['eeH'] = 'Z(e^{-}e^{+})H'
legend['Zmumu'] = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['Zee'] = 'Z/#gamma#rightarrow e^{+}e^{-}'
legend['eeZ'] = 'e^{+}(e^{-})#gamma'
legend['WWmumu'] = 'W^{+}(#nu_{#mu}#mu^{+})W^{-}(#bar{#nu}_{#mu}#mu^{-})'
legend['WWee'] = 'W^{+}(#nu_{e}e^{+})W^{-}(#bar{#nu}_{e}e^{-})'
legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee'] = '#gamma#gamma#rightarrow e^{+}e^{-}'
legend['WW'] = 'W^{+}W^{-}'
legend['ZZ'] = 'ZZ'
