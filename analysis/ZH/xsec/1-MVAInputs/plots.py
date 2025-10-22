import ROOT
import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")
from userConfig import loc, get_loc, ecm, lumi, plot_file

cat = input('Select a channel [ee, mumu]: ')

# global parameters
intLumi        = lumi * 1e6
intLumiLabel   = "L = {} ab^{}".format(lumi, '{-1}')
if   cat == 'mumu': ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
elif cat =='ee':    ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
delphesVersion = '3.4.2'
energy         = ecm
collider       = 'FCC-ee'
inputDir       = get_loc(loc.HIST_MVA, cat, ecm, '')
yaxis          = ['lin','log']
stacksig       = ['nostack']
formats        = plot_file
outdir         = get_loc(loc.PLOTS_MVA, cat, ecm, '')

variables = [
    # Leptons
    "leading_p", "leading_theta", "subleading_p", "subleading_theta",
    # "leading_phi", "subleading_phi",
    # Zed
    "zll_m", "zll_p", "zll_theta", "zll_phi",
    # more control variables
    "acolinearity", "acoplanarity", "zll_deltaR",
    # Recoil
    "zll_recoil_m",
    # missing Information
    "visibleEnergy", "cosTheta_miss", "missingMass",
    # Higgsstrahlungness
    "H", "BDTscore"
]

# Dictonnary with the analysis name as a key, and the list of selections to be plotted for this analysis. 
# The name of the selections should be the same than in the final selection
selections = {}
selections['ZH'] = [
    "sel0"
]

extralabel = {}
extralabel['sel0'] = 'No cut'

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

ee_ll = f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee' else f'wzp6_ee_mumu_ecm{ecm}'

plots = {}
plots['ZH'] = {
     'signal':  {f'{cat}H':[f'wzp6_ee_{cat}H_ecm{ecm}']},
     'backgrounds':{
          f'WW{cat}':[f'p8_ee_WW_{cat}_ecm{ecm}'],
          'ZZ':[f'p8_ee_ZZ_ecm240'], f'Z{cat}':[ee_ll],
          'eeZ':[f"wzp6_egamma_eZ_Z{cat}_ecm{ecm}", f"wzp6_gammae_eZ_Z{cat}_ecm{ecm}"],
          f'gaga{cat}':[f"wzp6_gaga_{cat}_60_ecm{ecm}"]}
}

legend = {}
legend['mumuH']    = 'Z(#mu^{+}#mu^{-})H'
legend['eeH']      = 'Z(e^{+}e^{-})H'
legend['Zmumu']    = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['Zee']      = 'Z/#gamma#rightarrow e^{+}e^{-}'
legend['eeZ']      = 'e^{+}(e^{-})#gamma'
legend['WWmumu']   = 'W^{+}W^{-}[#nu_{#mu}#mu]'
legend['WWee']     = 'W^{+}W^{-}[#nu_{e}e]'
legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee']   = '#gamma#gamma#rightarrow e^{+}e^{-}'
legend['WW']       = 'W^{+}W^{-}'
legend['ZZ']       = 'ZZ'
