##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, ROOT

# Load plot configuration, directory paths, and output settings
from package.config import quarks
from package.userConfig import (
    loc, get_params,
    plot_file
)
from sel.final.leptonic import histos_ll
from sel.final.hadronic import histos_qq

# Load analysis parameters: decay category, CoM energy, luminosity
cat, ecm, lumi, test = get_params(os.environ.copy(), '1-run.json', is_final=True, qq_allowed=True)



############################
### GLOBAL PLOT SETTINGS ###
############################

# Luminosity and experiment information
intLumi        = lumi * 1e6                           # Integrated luminosity in pb^-1
intLumiLabel   = 'L = {} ab^{}'.format(lumi, '{-1}')  # LaTeX label for luminosity
if   cat == 'mumu': ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
elif cat =='ee':    ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
elif cat =='qq':    ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow q#bar{q} + X'
else: raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')

# Detector and analysis metadata
delphesVersion = '3.4.2'   # Detector simulation version
energy         = ecm       # Center-of-mass energy [GeV]
collider       = 'FCC-ee'  # Collider identifier

# Input/output directories and plot configuration
inputDir       = loc.get('HIST_MVA',  cat, ecm)  # Input histograms from final-selection.py
outdir         = loc.get('PLOTS_MVA', cat, ecm)  # Output plots directory
yaxis          = ['lin','log']  # Y-axis scale options (linear and logarithmic)
stacksig       = ['nostack']    # Signal display mode (nostack = overlaid)
formats        = plot_file      # Output file formats (e.g., png, pdf)

# Scale factors for signal and background (for visual comparison)
scaleSig       = 50. if cat=='qq' else 1.   # Signal scale     (1.0 = no scaling)
scaleBkg       = 1.                         # Background scale (1.0 = no scaling)

# Plot appearance settings
strictRange    = True      # Use strict axis ranges from histogram definitions
setGrid        = True      # Display grid lines on plots
customLabel    = 'Training sample'  # Custom label shown on plots



###########################
### KINEMATIC VARIABLES ###
###########################

# Comprehensive list of kinematic variables to plot (sorted alphabetically)
# These variables are computed in pre-selection.py and filled into histograms by final-selection.py
variables = sorted(histos_ll.keys()) if cat in ['ee', 'mumu'] else \
    (sorted(histos_qq.keys()) if cat=='qq' else [])



#####################################
### PLOT CONFIGURATION DICTIONARY ###
#####################################

# Dictionary associating analysis names with selection strategies
# Keys: analysis identifier | Values: list of selection cut names to plot
# Selection names must match those defined in final-selection.py
selections: dict[str, list[str]] = {}
selections['ZH'] = ['sel0', 'Baseline']
if test: selections['ZH'].append('test')

# Additional descriptive labels for each selection cut
# Displayed below plot titles for clarity
extralabel = {}
extralabel['sel0']     = 'No cut'         # Diagnostic: no selection applied
extralabel['Baseline'] = 'Baseline'       # Standard selection
extralabel['test']     = 'test'           # Test selection

# Process and sample definitions for the analysis
# Dictionary structure: analysis_name -> {'signal': {...}, 'backgrounds': {...}}
# Each process can contain multiple samples from different sources
plots = {}
plots['ZH'] = {
    'signal': {
        f'{cat}H': [f'wzp6_ee_{cat}H_ecm{ecm}'] if cat in ['ee', 'mumu'] else
        [f'wzp6_ee_{x}H_ecm{ecm}' for x in quarks]
    },

    'backgrounds': {
        f'WW{cat}':   [f'p8_ee_WW_ecm{ecm}' if cat=='qq' else f'p8_ee_WW_{cat}_ecm{ecm}'],
        'ZZ':         [f'p8_ee_ZZ_ecm{ecm}'],
        f'Z{cat}':    [f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee'
                       else f'wzp6_ee_{cat}_ecm{ecm}'],
        'eeZ':        [f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
                       f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}'],
    }
}
if cat in ['ee', 'mumu']:
    # plots['ZH']['signal']['ZH'] = [f'wzp6_ee_{cat}H_H{y}_ecm{ecm}' for y in h_decays]
    plots['ZH']['backgrounds'][f'gaga{cat}'] = [f'wzp6_gaga_{cat}_60_ecm{ecm}']
# elif cat == 'qq':
#     plots['ZH']['signal']['ZH'] = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in quarks for y in h_decays]
# else:
#     raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')



# ROOT color assignments for each process (used in legend and stacked histograms)
# Ensures consistent color scheme across all plots
colors = {}
colors['ZH']         = ROOT.kBlack      # Signal from exclusive sample (test)
colors[f'{cat}H']    = ROOT.kRed        # Signal: bright red
colors['WW']         = ROOT.kBlue+1     # WW background: blue
colors['ZZ']         = ROOT.kGreen+2    # ZZ background: green
colors[f'Z{cat}']    = ROOT.kCyan       # Z+jets background: cyan
colors['eeZ']        = ROOT.kSpring+10  # Radiative Z: spring color
colors[f'WW{cat}']   = ROOT.kBlue+1     # WW with leptons: blue
colors[f'gaga{cat}'] = ROOT.kBlue-8     # Diphoton: dark blue

# LaTeX legend labels for ROOT plots
# Maps process names to formatted particle physics notation
legend = {}
legend['ZH']       = 'ZH'
legend['mumuH']    = 'Z(#mu^{+}#mu^{-})H'
legend['eeH']      = 'Z(e^{+}e^{-})H'
legend['qqH']      = 'Z(q#bar{q})H'

legend['WWmumu']   = 'W^{+}W^{-}[#nu_{#mu}#mu]'
legend['WWee']     = 'W^{+}W^{-}[#nu_{e}e]'
legend['WWqq']     = 'W^{+}W^{-}[had]'
legend['WW']       = 'W^{+}W^{-}'

legend['ZZ']       = 'ZZ'

legend['Zmumu']    = 'Z/#gamma^{*}#rightarrow #mu^{+}#mu^{-}'
legend['Zee']      = 'Z/#gamma^{*}#rightarrow e^{+}e^{-}'
legend['Zqq']      = 'Z/#gamma^{*}#rightarrow q#bar{q}'

legend['eeZ']      = 'e^{+}(e^{-})#gamma'

legend['gagamumu'] = '#gamma#gamma#rightarrow#mu^{+}#mu^{-}'
legend['gagaee']   = '#gamma#gamma#rightarrow e^{+}e^{-}'
legend['gagaqq']   = '#gamma#gamma#rightarrow q#bar{q}'
