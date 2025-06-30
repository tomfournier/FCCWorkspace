import ROOT
import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")
from userConfig import loc, get_loc, ecm, sel, lumi, plot_file

final_state = input('Select a channel [ee, mumu]: ')

# global parameters
intLumi        = lumi * 1e6
intLumiLabel   = "L = {} ab^{}".format(lumi, '{-1}')
if final_state == 'mumu':
     ana_tex        = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
elif final_state =='ee':
     ana_tex        = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
delphesVersion = '3.4.2'
energy         = ecm
collider       = 'FCC-ee'
inputDir       = get_loc(loc.HIST_MVAFINAL,  final_state, ecm, sel)
yaxis          = ['lin','log']
stacksig       = ['nostack']
formats        = [plot_file]
outdir         = get_loc(loc.PLOTS_MVA, final_state, ecm, sel)

variables = [   
     # Leptons
     "leading_p", "leading_theta", "subleading_p", "subleading_theta",
     # Zed
     "zll_m", "zll_p", "zll_theta", "zll_phi",
     # more control variables
     "acolinearity", "acoplanarity",
     # Recoil
     "zll_recoil_m",
     # missing Information
     "visibleEnergy", "cosTheta_miss",
     # Higgsstrahlungness
     "H"
]

# Dictonnary with the analysis name as a key, and the list of selections to be plotted for this analysis. The name of the selections should be the same than in the final selection
_120 = '_120' if userConfig.recoil120 else ''
selections = {}
selections['ZH'] = [sel]

_miss, _120 = ' and cos#theta_{miss} cut' if userConfig.miss else '', ' and 120 < m_{recoil} < 140 GeV' if userConfig.recoil120 else ''
_bdt, _vis = ' and cos#theta_{miss} as input' if userConfig.bdt else '', ' and E_{vis} cut' if userConfig.vis else ''
_leading, _visbdt =   ' and p_{leading}, p_{sub} cuts' if userConfig.leading else '', '_visBDT' if userConfig.visbdt else ''

selection = 'Baseline'+_leading+_vis+_visbdt+_120+_miss+_bdt

extralabel = {}
extralabel[sel] = selection


colors = {}
colors['mumuH']    = ROOT.kRed
colors['eeH']      = ROOT.kRed
colors['Zmumu']    = ROOT.kCyan
colors['Zee']      = ROOT.kCyan
colors['eeZ']      = ROOT.kSpring+10
colors['WWmumu']   = ROOT.kBlue+1
colors['WWee']     = ROOT.kBlue+1
colors['gagamumu'] = ROOT.kBlue-8
colors['gagaee']   = ROOT.kBlue-8
colors['WW']       = ROOT.kBlue+1
colors['ZZ']       = ROOT.kGreen+2

ee_ll = f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if final_state=='ee' else f'wzp6_ee_mumu_ecm{ecm}'

plots = {}
plots['ZH'] = {
     'signal':  {f'{final_state}H':[f'wzp6_ee_{final_state}H_ecm{ecm}']},
     'backgrounds':{
          f'WW{final_state}':[f'p8_ee_WW_{final_state}_ecm{ecm}'],
          'ZZ':[f'p8_ee_ZZ_ecm240'], f'Z{final_state}':[ee_ll],
          'eeZ':[f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}", f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}"],
          f'gaga{final_state}':[f"wzp6_gaga_{final_state}_60_ecm{ecm}"]}
}

legend = {}
legend['mumuH']    = 'Z(#mu^{-}#mu^{+})H'
legend['eeH']      = 'Z(e^{-}e^{+})H'
legend['Zmumu']    = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['Zee']      = 'Z/#gamma#rightarrow e^{+}e^{-}'
legend['eeZ']      = 'e^{+}(e^{-})#gamma'
legend['WWmumu']   = 'W^{+}W^{-}[#nu_{#mu}#mu]'
legend['WWee']     = 'W^{+}W^{-}[#nu_{e}e]'
legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee']   = '#gamma#gamma#rightarrow e^{+}e^{-}'
legend['WW']       = 'W^{+}W^{-}'
legend['ZZ']       = 'ZZ'


