import ROOT
import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# global parameters
intLumi        = 1.
intLumiLabel   = "L = 10.8 ab^{-1}"
if final_state=='mumu':
     ana_tex   = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
elif final_state=='ee':
     ana_tex   = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png']

if userConfig.final:
     outdir    = userConfig.loc.PLOTS_CUTFLOW_FINAL
     inputDir  = userConfig.loc.CUTFLOW_FINAL
else:
     outdir    = userConfig.loc.PLOTS_CUTFLOW_MVA
     inputDir  = userConfig.loc.CUTFLOW_MVA

plotStatUnc    = False


procs = {}
procs['signal'] = {f'{final_state}H':[f'wzp6_ee_{final_state}H_ecm{ecm}']}

if final_state=="mumu":
        ee_ll = f'wzp6_ee_{final_state}_ecm{ecm}'
elif final_state=="ee":
        ee_ll = f'wzp6_ee_{final_state}_Mee_30_150_ecm{ecm}'

if userConfig.final:
     procs['backgrounds'] = {f'WW{final_state}':[f'p8_ee_WW_ecm{ecm}'],
                    'ZZ':[f'p8_ee_ZZ_ecm240'], f'Zg':[ee_ll, f"wzp6_ee_tautau_ecm{ecm}"],
                    'Rare':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}", f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
                            f"wzp6_gaga_{final_state}_60_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}",
                            f"wzp6_ee_nuenueZ_ecm{ecm}"]}
else:
     procs['backgrounds'] = {f'WW{final_state}':[f'p8_ee_WW_{final_state}_ecm{ecm}'],
                         'ZZ':[f'p8_ee_ZZ_ecm240'],
                         f'Zg':[ee_ll],
                         'eeZ':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
                                   f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}"],
                                   f'gaga{final_state}':[f"wzp6_gaga_{final_state}_60_ecm{ecm}"]}

colors = {}
colors['mumuH'] = ROOT.kRed
colors['eeH'] = ROOT.kRed
colors['Zmumu'] = ROOT.kCyan
colors['Zee'] = ROOT.kCyan
colors['Zg'] = ROOT.kMagenta+2
colors['eeZ'] = ROOT.kSpring+10
colors['WWmumu'] = ROOT.kOrange-2
colors['WWee'] = ROOT.kOrange-2
colors['gagamumu'] = ROOT.kBlue-8
colors['gagaee'] = ROOT.kBlue-8
colors['WW'] = ROOT.kOrange-2
colors['ZZ'] = ROOT.kBlue-7
colors['Rare'] = ROOT.kBlue-5

legend = {}
legend['mumuH'] = 'Z(#mu^{-}#mu^{+})H'
legend['eeH'] = 'Z(e^{-}e^{+})H'
legend['Zmumu'] = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['Zee'] = 'Z/#gamma#rightarrow e^{+}e^{-}'
if final_state=='ee':
     legend['Zg'] = 'Z/#gamma#rightarrow e^{+}e^{-}, #tau^{+}#tau^{-}'
elif final_state=='mumu':
      legend['Zg'] = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}, #tau^{+}#tau^{-}'
legend['eeZ'] = 'e^{+}(e^{-})#gamma'
legend['WWmumu'] = 'W^{+}W^{-}[#nu_{#mu}#mu]'
legend['WWee'] = 'W^{+}W^{-}[#nu_{e}e]'
legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee'] = '#gamma#gamma#rightarrow e^{+}e^{-}'
legend['WW'] = 'W^{+}W^{-}'
legend['ZZ'] = 'ZZ'
legend['Rare'] = 'Rare'

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
    "xmax":     7,
    "ymin":     1e4,
    "ymax":     1e11,
    "xtitle":   xtitle,
    "ytitle":   "Events",
    "scaleSig": 1
}