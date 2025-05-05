import ROOT
import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# global parameters
intLumi        = userConfig.intLumi * 1e6 #in pb-1
if final_state == 'mumu':
     ana_tex        = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
else:
     ana_tex        = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
delphesVersion = '3.4.2'
energy         = ecm
collider       = 'FCC-ee'
inputDir       = userConfig.loc.FINAL
yaxis          = ['lin','log']
stacksig       = ['stack', 'nostack']
formats        = ['png']
outdir         = userConfig.loc.PLOTS

variables = [   
     # muons
     "leading_p", "leading_theta",
     "subleading_p", "subleading_theta",
     # Zed
     "zll_m", "zll_p", "zll_theta",
     # more control variables
     "acolinearity", "acoplanarity",
     # Recoil
     "zll_recoil_m",
     # missing Information
     "cosTheta_miss",
     # Higgsstrahlungness
     "H"
]

# Dictonnary with the analysis name as a key, and the list of selections to be plotted for this analysis. The name of the selections should be the same than in the final selection
selections = {}
selections['ZH'] = ["sel_Baseline_no_costhetamiss"] # "sel_Baseline_costhetamiss"]

extralabel = {}
extralabel["sel_Baseline_no_costhetamiss"] = "Baseline without cos#theta_{miss} cut"
# extralabel["sel_Baseline_costhetamiss"] = "Baseline with cos#theta_{miss} cut"   

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
colors['rare'] = ROOT.kSpring

if userConfig.training:
     dict = {'signal':{f'{final_state}H':[f'wzp6_ee_{final_state}H_ecm{ecm}']},
             'backgrounds':{
                  f'WW{final_state}':[f'p8_ee_WW_{final_state}_ecm{ecm}'],
                  'ZZ':[f'p8_ee_ZZ_ecm240'],
                  f'Z{final_state}':[f'wzp6_ee_{final_state}_ecm{ecm}'],
                  'eeZ':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
                         f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}"],
                         f'gaga{final_state}':[f"wzp6_gaga_{final_state}_60_ecm{ecm}"]}
     }
else:
     dict = {'signal':{f'{final_state}H':[f'wzp6_ee_{final_state}H_ecm{ecm}']}, 
             'backgrounds':{
                  'eeZ':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}", f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}"],
                  'WW':[f'p8_ee_WW_ecm{ecm}'], 
                  f'Z{final_state}':[f'wzp6_ee_{final_state}_ecm{ecm}'], 
                  'ZZ':[f'p8_ee_ZZ_ecm{ecm}'],
                  'rare':[f"wzp6_ee_tautau_ecm{ecm}", f"wzp6_gaga_{final_state}_60_ecm{ecm}",
                          f"wzp6_gaga_tautau_60_ecm{ecm}", f"wzp6_ee_nuenueZ_ecm{ecm}"]}
     }

plots = {}
plots['ZH'] = dict

legend = {}
legend['mumuH'] = 'Z(#mu^{-}#mu^{+})H'
legend['eeH'] = 'Z(e^{-}e^{+})H'
legend['Zmumu'] = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['Zee'] = 'Z/#gamma#rightarrow e^{+}e^{-}'
legend['eeZ'] = 'e^{+}(e^{-})#gamma'
legend['WWmumu'] = 'W^{+}W^{-}[#nu_{#mu}#mu]'
legend['WWee'] = 'W^{+}W^{-}(#nu_{e}e)'
legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee'] = '#gamma#gamma#rightarrow e^{+}e^{-}'
legend['WW'] = 'W^{+}W^{-}'
legend['ZZ'] = 'ZZ'
legend['rare'] = 'Rare'