'''Pseudo-data generation and statistical analysis tools for bias tests.

Provides:
- Signal process construction: `_signal_lists()`.
- Cross-section scaling calculations: `_scaling()`.
- Pseudo-data histogram generation: `make_pseudodata()`, `make_pseudosignal()`.
- Statistical analysis datacard creation: `make_datacard()`.
- Support for signal injection and Higgs decay channel variations.
- Integration with Combine tool for maximum likelihood fits.

Functions:
- `_signal_lists()`: Generate nested lists of signal process names for each Higgs decay.
- `_scaling()`: Compute scaling factors for signal channel variations while preserving total cross-section.
- `make_pseudodata()`: Create pseudo-data histogram combining backgrounds and perturbed signal.
- `make_pseudosignal()`: Generate signal-only histogram with specified decay channel variation.
- `make_datacard()`: Build Combine-compatible statistical analysis datacard with shape templates.

Conventions:
- Signal process naming follows FCC pattern: `wzp6_ee_{z_decay}H_H{h_decay}_ecm{ecm}`.
- For invisible Higgs decays, 'ZZ' channel replaced with 'ZZ_noInv' to avoid double-counting.
- Cross-section scaling preserves total signal yield: scale_target = 1 + (variation - 1) * xsec_tot / xsec_target.
- Variation parameter >1.0 increases total signal (e.g., 1.05 for +5%), <1.0 decreases.
- Target channel scaling computed such that total signal cross-section matches variation factor.
- Pseudo-data combines backgrounds (unscaled) plus signal (scaled per variation parameter).
- Pseudo-signal contains only signal histograms (no backgrounds).
- Datacard organized by categories (channels) with processes (signal + backgrounds).
- Process indices: signal=0, background1=1 (or -1 if floating), background2=2 (or -2), etc.
- Background uncertainties specified as log-normal nuisance parameters (lnN format in datacard).
- Categories and processes iterated in synchrony; rates placeholders filled from ROOT histograms.

Usage:
- Test fit bias by generating pseudo-experiments with known signal variations.
- Create pseudo-data with +/- variations in specific Higgs decay channels.
- Generate Combine datacards for performing maximum likelihood fits on pseudo-data.
- Validate signal injection recovery and measurement bias across decay channels.

Lazy Imports:
- hist and uproot are lazy-loaded only when needed
- ROOT's TFile is lazy-loaded only when file operations are performed
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import hist

from ..tools.utils import mkdir
from ..tools.process import getMetaInfo, getHist



########################
### HELPER FUNCTIONS ###
########################

def _signal_lists(
    cat: str,
    z_decays: list[str],
    h_decays: list[str],
    target: str,
    ecm: int = 240,
    tot: bool = True
    ) -> list[list[str]]:
    '''Generate lists of signal process names for different Higgs decay channels.
    
    Args:
        cat (str): Category name for the Z decay channel.
        z_decays (list[str]): List of Z boson decay channels.
        h_decays (list[str]): List of Higgs boson decay channels.
        target (str): Target channel for the analysis (e.g., 'inv' for invisible).
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        tot (bool, optional): If True, use all Z decays; if False, use only the specified category. Defaults to True.
    
    Returns:
        list[list[str]]: Nested list where each sublist contains process names for a specific Higgs decay.
    '''
    # Use all Z decays or just the specified category
    cats = z_decays if tot else [cat]
    # Template for process naming convention
    template = f'wzp6_ee_{{0}}H_H{{1}}_ecm{ecm}'
    # Replace 'ZZ' with 'ZZ_noInv' for invisible channel target
    h_list = [y.replace('ZZ', 'ZZ_noInv') if target=='inv' else y for y in h_decays]

    return [[template.format(x, y) for x in cats] for y in h_list]

def _scaling(
    sigs: list[list[str]], 
    h_decays: list[str], 
    target: str, 
    variation: float, 
    verbose: bool = True
    ) -> tuple[float, 
               float, 
               float]:
    '''Calculate scaling factors for signal variations in bias tests.
    
    Args:
        sigs (list[list[str]]): Nested list of signal process names for each Higgs decay.
        h_decays (list[str]): List of Higgs decay channel names.
        target (str): Target Higgs decay channel to perturb.
        variation (float): Scaling factor for total cross-section (e.g., 1.05 for +5%).
        verbose (bool, optional): If True, print scaling information. Defaults to True.
    
    Returns:
        tuple: (scale_target, xsec_tot, xsec_tot_new) where scale_target is the scaling factor applied to the target channel, xsec_tot is the original total cross-section, and xsec_tot_new is the new total cross-section after variation.
    '''
    # Initialize cross-section accumulators
    xsec_tot = xsec_target = 0.0

    # Calculate total and target cross-sections
    for h_decay, sig in zip(h_decays, sigs):
        xsec = sum(getMetaInfo(s, rmww=True) for s in sig)
        xsec_tot += xsec
        if h_decay==target:
            xsec_target = xsec

    # Calculate scaling parameters
    xsec_delta = xsec_tot * (variation - 1.0)      # Total cross-section change
    scale_target = 1.0 + xsec_delta / xsec_target  # Scale for target channel
    xsec_tot_new = xsec_tot * variation            # New total cross-section

    if verbose:
        print(f'----->[Info] Making pseudo data for {target} channel')
        print(f'----->[Info] Perturbation: {(variation-1)*100:.2f} %, Scale: {scale_target:.3f}')
        print(f'----->[Info] Target xsec: {xsec_target:.3e} pb-1')

    return scale_target, xsec_tot, xsec_tot_new



######################
### MAIN FUNCTIONS ###
######################

#________________________________________
def make_pseudodata(
    hName: str, 
    inDir: str, 
    procs: list[str], 
    processes: dict[str, list[str]], 
    cat: str, 
    z_decays: list[str], 
    h_decays: list[str], 
    target: str,
    ecm: int = 240, 
    variation: float = 1.05, 
    suffix: str = '', 
    proc_scales: dict[str, float] = None, 
    tot: bool = True):
    '''Create pseudo-data histogram with perturbed signal for bias testing.
    
    Generates pseudo-data by combining backgrounds and signal with a specified
    variation in the target channel to test fit bias.
    
    Args:
        hName (str): Histogram name to retrieve.
        inDir (str): Input directory containing histograms.
        procs (list[str]): List of process names (signal first, then backgrounds).
        processes (dict[str, list[str]]): Dictionary mapping process names to sample names.
        cat (str): Category name for Z decay channel.
        z_decays (list[str]): List of Z boson decay channels.
        h_decays (list[str]): List of Higgs boson decay channels.
        target (str): Target Higgs decay channel to perturb.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        variation (float, optional): Scaling factor for total signal (e.g., 1.05 for +5%). Defaults to 1.05.
        suffix (str, optional): Suffix for histogram retrieval. Defaults to ''.
        proc_scales (dict[str, float], optional): Optional scaling factors per process. Defaults to None.
        tot (bool, optional): If True, use all Z decays; if False, use only the specified category. Defaults to True.
    
    Returns:
        ROOT.TH1: ROOT histogram containing the pseudo-data.
    '''
    # Initialize process scales dictionary
    if proc_scales is None:
        proc_scales = {}

    # Generate signal process lists and calculate scaling factors
    sigs = _signal_lists(cat, z_decays, h_decays, target, ecm=ecm, tot=tot)

    scale_target, xsec_tot, xsec_tot_new = _scaling(
        sigs, h_decays, target, variation, verbose=True
    )

    # Initialize histograms
    hist_pseudo = hist_old = hist_new = h_bkg = None

    # Add all background processes
    for proc in procs[1:]:
        scale = proc_scales.get(proc, 1.0)
        h = getHist(
            hName, processes[proc], inDir, 
            suffix=suffix, proc_scale=scale
        )
        
        if hist_pseudo is None: 
            hist_pseudo = h.Clone('h_pseudo')
            h_bkg = h.Clone('h_bkg')
        else: 
            hist_pseudo.Add(h)
            h_bkg.Add(h)

    # Get ZH signal scaling factor
    zh_scale = proc_scales.get('ZH', 1.0)
        
    # Add signal processes with target channel perturbation
    for h_decay, sig in zip(h_decays, sigs):
        h = getHist(
            hName, sig, inDir, 
            suffix=suffix, 
            proc_scale=zh_scale
        )
        
        # Store original signal histogram
        if hist_old is None: 
            hist_old = h.Clone('h_old')
        else:
            hist_old.Add(h)

        # Apply scaling to target channel
        if h_decay==target:
            h.Scale(scale_target)

        # Store perturbed signal histogram
        if hist_new is None: 
            hist_new = h.Clone('h_new')
        else: 
            hist_new.Add(h)
        
        # Add signal to pseudo-data
        hist_pseudo.Add(h)

    # Calculate and report signal increase
    delta_pct = (hist_new.Integral() / hist_old.Integral() - 1.0) * 100
    scale_ratio = xsec_tot_new / xsec_tot

    print(f'----->[Info] Signal increased by {delta_pct:.2f} %')
    print(f'----->[CROSS-CHECK] Scale ratio {scale_ratio:.2f} vs target {variation}\n')
    return hist_pseudo

#____________________________
def make_datacard(
    outDir: str, 
    procs: list[str], 
    target: str, 
    bkg_unc: float, 
    categories: list[str], 
    freezeBkgs: bool = False,
    floatBkgs: bool = False, 
    plot_dc: bool = False
    ) -> None:
    '''Generate a datacard for statistical analysis with Combine.
    
    Creates a text datacard compatible with the Combine tool for performing
    fits and extracting signal strength.
    
    Args:
        outDir (str): Output directory for the datacard file.
        procs (list[str]): List of process names (signal first, then backgrounds).
        target (str): Target channel name for the datacard.
        bkg_unc (float): Background uncertainty (log-normal nuisance parameter).
        categories (list[str]): List of category names (e.g., different channels).
        freezeBkgs (bool, optional): If True, freeze background normalizations. Defaults to False.
        floatBkgs (bool, optional): If True, float backgrounds as negative process indices. Defaults to False.
        plot_dc (bool, optional): If True, print the datacard content to console. Defaults to False.
    
    Returns:
        None
    '''
    # Number of processes and categories
    nprocs = len(procs)
    ncats = len(categories)

    # Column width for formatting
    col_w = 12

    # Format strings for categories and processes
    cats_str       = ''.join([f'{cat:<{col_w}}'  for cat  in categories])
    procs_str      = ''.join([f'{proc:<{col_w}}' for proc in procs]) * ncats
    cats_procs_str = ''.join([f'{cat:<{col_w}}'  for cat  in categories for _ in range(nprocs)])

    # Process indices: signal=0, backgrounds=1,2,3,... (or negative if floating)
    p = -1 if floatBkgs else 1
    procs_idx = [0] + [p*i for i in range(1, nprocs)]
    cats_procs_idx_str = ''.join([f'{idx:<{col_w}}' for idx in procs_idx]) * ncats

    # Rates placeholder (will be filled from ROOT file)
    rates_cats  = f'{-1:<{col_w}}' * ncats
    rates_procs = f'{-1:<{col_w}}' * (ncats * nprocs)

    # Separator line for datacard sections
    sep = '#' * (22 + len(cats_procs_str))
    # Build datacard content line by line
    dc_lines = [
        'imax *',  # Number of bins (automatic)
        'jmax *',  # Number of processes minus 1 (automatic)
        'kmax *',  # Number of systematics (automatic)
        sep,
        f'shapes *        * datacard_{target}.root $CHANNEL_$PROCESS',      # Shape templates
        f'shapes data_obs * datacard_{target}.root $CHANNEL_data_{target}', # Observed data
        sep,
        f'bin                        {cats_str}',   # Bin names
        f'observation                {rates_cats}', # Observed event counts
        sep,
        f'bin                        {cats_procs_str}',     # Bins for each process
        f'process                    {procs_str}',          # Process names
        f'process                    {cats_procs_idx_str}', # Process indices
        f'rate                       {rates_procs}',        # Expected rates
        sep,
    ]

    # Add systematic uncertainties
    if not freezeBkgs and not floatBkgs:
        # Add individual background normalization uncertainties
        for proc in procs[1:]:
            vals = ''.join(
                f"{bkg_unc if p1==proc else '-':<{col_w}}"
                for _ in categories for p1 in procs
            )
            dc_lines.append(f"{'norm_'+proc:<15} {'lnN':<10} {vals}")
            
    else:
        # Only add signal normalization uncertainty (negligible)
        vals = ''.join(
            f"{1.000000005 if p==procs[0] else '-':<{col_w}}"
            for _ in categories for p in procs
        )
        dc_lines.append(f"{'norm_'+procs[0]:<15} {'lnN':<10} {vals}")

    # Combine all lines into final datacard text
    dc = '\n'.join(dc_lines) + '\n'

    # Write datacard to file
    fOut = f'{outDir}/datacard_{target}.txt'
    print(f'----->[Info] Saving datacard to {fOut}')
    with open(fOut, 'w') as f: f.write(dc)

    # Optionally print datacard content
    if plot_dc: print(f'\n{dc}\n')

#___________________________________
def pseudo_datacard(
    inDir: str, 
    outDir: str,
    cat: str,
    ecm: int,
    target: str,
    pert: float,
    z_decays: list[str],
    h_decays: list[str],
    processes: dict[str, list[str]],
    tot: bool = False,
    scales: str = '',
    freeze: bool = False,
    float_bkg: bool = False,
    plot_dc: bool = False
    ) -> None:
    '''
    Generate pseudodata histogram and datacard for a specific Higgs decay target.
    
    This function can be called directly from bias_test.py to benefit from cached
    histograms in the same process, or used standalone via subprocess.
    
    Args:
        cat (str): Final state category (ee or mumu).
        ecm (int): Center of mass energy.
        sel (str): Selection strategy.
        target (str): Target Higgs decay channel.
        pert (float): Perturbation/variation factor.
        tot (bool): Whether to use all Z decays.
        proc_scales (dict): Process-specific scale factors.
        freeze (bool, optional): Freeze background normalizations. Defaults to False.
        float_bkg (bool, optional): Float backgrounds. Defaults to False.
        plot_dc (bool, optional): Print datacard to console. Defaults to False.
    '''

    import ROOT

    # Define histogram names and categories
    hNames = ('zll_recoil_m',)
    categories = (f'z_{cat}',)
    
    # List of processes and their samples
    procs = ['ZH' if tot else f'Z{cat}H', 'WW', 'ZZ', 'Zgamma', 'Rare']

    # Set process-wise scale factors based on polarization or luminosity
    proc_scales = {
        'ILC':  {'ZH': 1.048, 'WW': 0.971, 'ZZ': 0.939, 'Zgamma': 0.919}, ## change fit to ASIMOV -t -1 !!!
        'polL': {'ZH': 1.554, 'WW': 2.166, 'ZZ': 1.330, 'Zgamma': 1.263},
        'polR': {'ZH': 1.047, 'WW': 0.219, 'ZZ': 1.011, 'Zgamma': 1.018},
    }
    
    # Collect histograms
    hists = []
    for i, categorie in enumerate(categories):
        for proc in procs:
            h = getHist(hNames[i], processes[proc], inDir)
            h.SetName(f'{categorie}_{proc}')
            hists.append(h)

        # Generate pseudodata histogram
        hist_pseudo = make_pseudodata(
            hNames[i], inDir, procs, processes, cat, z_decays, h_decays,
            target, ecm=ecm, variation=pert, tot=tot,
            proc_scales=proc_scales.get(scales)
        )
        hist_pseudo.SetName(f'{categorie}_data_{target}')
        hists.append(hist_pseudo)

    # Create output directory and save histograms
    mkdir(outDir)
    print('----->[Info] Saving pseudo histograms')
    fOut = f'{outDir}/datacard_{target}.root'

    with ROOT.TFile(fOut, 'RECREATE') as f:
        for hist in hists:
            hist.Write()

    print(f'----->[Info] Histograms saved in {fOut}')

    # Generate datacard for combine fit
    print('----->[Info] Making datacard')
    make_datacard(
        outDir, procs, target, 1.01, categories,
        freezeBkgs=freeze, floatBkgs=float_bkg, 
        plot_dc=plot_dc
    )

#___________________________
def hist_from_datacard(
    inDir: str,
    target: str,
    cat: str,
    procs: list[str]
    ) -> tuple['hist.Hist', 
               'hist.Hist']:

    import uproot

    datacard = uproot.open(f'{inDir}/datacard_{target}.root')

    hist_pseudo = datacard[f'z_{cat}_data_{target}'].to_hist()
    hist_sig    = datacard[f'z_{cat}_{procs[0]}'].to_hist()
    
    # Build background histogram by summing all background processes
    hist_bkg = None
    for proc in procs[1:]:
        h = datacard[f'z_{cat}_{proc}'].to_hist()
        if hist_bkg is None:
            hist_bkg = h.copy()
        else:
            # Use view to access and modify the underlying data
            hist_bkg.view(flow=False).value[:] += h.view(flow=False).value
            hist_bkg.view(flow=False).variance[:] += h.view(flow=False).variance

    # Subtract background from pseudo-data
    if hist_bkg is not None:
        hist_pseudo.view(flow=False).value[:] -= hist_bkg.view(flow=False).value
        hist_pseudo.view(flow=False).variance[:] += hist_bkg.view(flow=False).variance
    
    return hist_sig, hist_pseudo