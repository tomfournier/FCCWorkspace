import ROOT
import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# global parameters
intLumi        = userConfig.intLumi * 1e6 # in pb-1
intLumiLabel   = "L = 10.8 ab^{-1}"
if final_state=='mumu':
     ana_tex   = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
elif final_state=='ee':
     ana_tex   = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png']

outdir         = userConfig.loc.PLOTS_CUTFLOW
inputDir       = userConfig.loc.CUTFLOW 

plotStatUnc    = True


procs = {}
procs['signal'] = {f'{final_state}H':[f'wzp6_ee_{final_state}H_ecm{ecm}']}

if final_state=="mumu":
     procs['backgrounds'] = {f'WW{final_state}':[f'p8_ee_WW_{final_state}_ecm{ecm}'],
                             'ZZ':[f'p8_ee_ZZ_ecm240'],
                             f'Z{final_state}':[f'wzp6_ee_{final_state}_ecm{ecm}'],
                             'eeZ':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
                                    f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}"],
                                    f'gaga{final_state}':[f"wzp6_gaga_{final_state}_60_ecm{ecm}"]}
elif final_state=="ee":
     procs['backgrounds'] = {f'WW{final_state}':[f'p8_ee_WW_{final_state}_ecm{ecm}'],
                             'ZZ':[f'p8_ee_ZZ_ecm240'],
                             f'Z{final_state}':[f'wzp6_ee_{final_state}_Mee_30_150_ecm{ecm}'],
                             'eeZ':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
                                    f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}"],
                                    f'gaga{final_state}':[f"wzp6_gaga_{final_state}_60_ecm{ecm}"]}
else:
     raise ValueError(f"final_state {final_state} not supported")

extralabel = {}
extralabel["sel_Baseline_no_costhetamiss"] = "Baseline without cos#theta_{miss} cut"
extralabel["sel_Baseline_costhetamiss"]    = "Baseline with cos#theta_{miss} cut"   

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

legend = {}
legend['mumuH'] = 'Z(#mu^{-}#mu^{+})H'
legend['eeH'] = 'Z(e^{-}e^{+})H'
legend['Zmumu'] = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['Zee'] = 'Z/#gamma#rightarrow e^{+}e^{-}'
legend['eeZ'] = 'e^{+}(e^{-})#gamma'
legend['WWmumu'] = 'W^{+}W^{-}[#nu_{#mu}#mu]'
legend['WWee'] = 'W^{+}W^{-}[#nu_{e}e]'
legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee'] = '#gamma#gamma#rightarrow e^{+}e^{-}'
legend['WW'] = 'W^{+}W^{-}'
legend['ZZ'] = 'ZZ'
legend['rare'] = 'Rare'

hists = {}

if final_state=='mumu':
     xtitle = ["All events", "#geq 1 #mu^{#pm} + ISO", "#geq 2 #mu^{#pm} + OS", "86 < m_{#mu^{+}#mu^{#minus}} < 96", 
               "20 < p_{#mu^{+}#mu^{#minus}} < 70", "120 < m_{rec} < 140", "|cos#theta_{miss}| < 0.98"]
elif final_state=='ee':
     xtitle = ["All events", "#geq 1 e^{#pm} + ISO", "#geq 2 e^{#pm} + OS", "86 < m_{e^{+}e^{#minus}} < 96", 
               "20 < p_{e^{+}e^{#minus}} < 70", "120 < m_{rec} < 140", "|cos#theta_{miss}| < 0.98"]

hists["cutFlow"] = {
    "output":   "cutFlow",
    "logy":     True,
    "stack":    False,
    "xmin":     0,
    "xmax":     6,
    "ymin":     1e4,
    "ymax":     1e10,
    "xtitle":   xtitle,
    "ytitle":   "Events",
    "scaleSig": 1
}