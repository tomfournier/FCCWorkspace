##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, ROOT

# Plot configuration and paths
from package.userConfig import (
    loc, get_params,
    plot_file
)

cat, ecm, lumi = get_params(os.environ.copy(), '1-run.json', is_final=True)
if cat not in ['ee', 'mumu']:
    raise ValueError(f'Invalid channel: {cat}. Must be "ee" or "mumu"')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# global parameters
intLumi        = lumi * 1e6
intLumiLabel   = 'L = {} ab^{}'.format(lumi, '{-1}')
if   cat == 'mumu': ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
elif cat =='ee':    ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
delphesVersion = '3.4.2'
energy         = ecm
collider       = 'FCC-ee'
inputDir       = loc.get('HIST_MVA', cat, ecm)
yaxis          = ['lin','log']
stacksig       = ['nostack']
formats        = plot_file
outdir         = loc.get('PLOTS_MVA', cat, ecm)

# Variables to plot from pre-selection outputs
variables = [
    # Leptons kinematics
    'leading_p',    'leading_pT',    'leading_theta',
    'subleading_p', 'subleading_pT', 'subleading_theta',
    # 'leading_phi', 'subleading_phi',

    # Z boson properties
    'zll_m', 'zll_p', 'zll_pT', 'zll_theta',
    # 'zll_phi',

    # Angular correlation
    'acolinearity', 'acoplanarity', 'deltaR',

    # Recoil mass (Higgs candidate)
    'zll_recoil_m',

    # Visible and invisible information
    'visibleEnergy', 'cosTheta_miss', 'missingMass',

    # Higgsstrahlungness
    'H'
]



#############################################
### DEFINE CONFIG DICTIONARY FOR THE PLOT ###
#############################################

# Dictonnary with the analysis name as a key,
# and the list of selections to be plotted for this analysis.
# The name of the selections should be the same than in the final selection
selections = {}
selections['ZH'] = [
    'sel0',
    'Baseline'
]

# Extra labels to display under plot titles per selection
extralabel = {}
extralabel['sel0'] = 'No cut'
extralabel['Baseline'] = 'Baseline'

# Plot configuration: signal and backgrounds per analysis
plots = {}
plots['ZH'] = {
    'signal':  {
        f'{cat}H': [f'wzp6_ee_{cat}H_ecm{ecm}']},

    'backgrounds':{
        f'WW{cat}':   [f'p8_ee_WW_{cat}_ecm{ecm}'],
        'ZZ':         [f'p8_ee_ZZ_ecm{ecm}'],
        f'Z{cat}':    [f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee'
                       else f'wzp6_ee_mumu_ecm{ecm}'],
        'eeZ':        [f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
                       f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}'],
        f'gaga{cat}': [f'wzp6_gaga_{cat}_60_ecm{ecm}']}
}

# Color palette for each process in the legend
colors = {}
colors[f'{cat}H']    = ROOT.kRed
colors['WW']         = ROOT.kBlue+1
colors['ZZ']         = ROOT.kGreen+2
colors[f'Z{cat}']    = ROOT.kCyan
colors['eeZ']        = ROOT.kSpring+10
colors[f'WW{cat}']   = ROOT.kBlue+1
colors[f'gaga{cat}'] = ROOT.kBlue-8

# Legend labels used in ROOT plots
legend = {}
legend['mumuH']    = 'Z(#mu^{+}#mu^{-})H'
legend['eeH']      = 'Z(e^{+}e^{-})H'

legend['WWmumu']   = 'W^{+}W^{-}[#nu_{#mu}#mu]'
legend['WWee']     = 'W^{+}W^{-}[#nu_{e}e]'
legend['WW']       = 'W^{+}W^{-}'

legend['ZZ']       = 'ZZ'

legend['Zmumu']    = 'Z/#gamma#rightarrow #mu^{+}#mu^{-}'
legend['Zee']      = 'Z/#gamma#rightarrow e^{+}e^{-}'

legend['eeZ']      = 'e^{+}(e^{-})#gamma'

legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee']   = '#gamma#gamma#rightarrow e^{+}e^{-}'
