
#################################
### IMPORT STANDARD LIBRARIES ###
#################################

# Standard library and scientific computing imports
import uproot

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
from pathlib import Path



########################
### ARGUMENT PARSING ###
########################

from package.logger import get_logger

LOGGER = get_logger(__name__)


##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Import utilities and plotting configurations
from .python.plotter import (
    set_plt_style,
    set_labels,
    savefigs
)


################################
### MATPLOTLIB STYLE SETTING ###
################################

set_plt_style()
plt.ioff()


########################
### HELPER FUNCTIONS ###
########################

def _make_fig(figsize: tuple = (12, 8),
              dpi: int = 100
              ) -> tuple[plt.Figure, plt.Axes]:
    """Create matplotlib figure and axes with standard formatting.

    Args:
        figsize: Figure size as (width, height) in inches
        dpi: Figure resolution in dots per inch

    Returns:
        Tuple of (matplotlib Figure, Axes objects)
    """
    return plt.subplots(figsize=figsize, dpi=dpi)


def _compute_cumsum_and_z(
        data: np.ndarray,
        is_signal: np.ndarray,
        bins: np.ndarray,
        reverse: bool = False
         ) -> tuple[np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray]:
    """Helper function to compute cumulative distributions and significance.

    Calculates cumulative sum of signal and background histograms, then computes
    significance as S/sqrt(S+B) for each bin. Optionally reverses cumulative
    sum direction (left-to-right vs right-to-left cuts).

    Args:
        data: Data array to histogram
        is_signal: Boolean mask for signal events
        bins: Bin edges
        reverse: If True, compute right-to-left cumsum instead of left-to-right

    Returns:
        Tuple of (S_cumsum, B_cumsum, Z_cumsum, bin_centers) arrays
    """
    hist_sig, _ = np.histogram(data[is_signal], bins=bins)
    hist_bkg, _ = np.histogram(data[~is_signal], bins=bins)

    if reverse:
        S_cumsum, B_cumsum = np.cumsum(hist_sig), np.cumsum(hist_bkg)
    else:
        S_cumsum = np.cumsum(hist_sig[::-1])[::-1]
        B_cumsum = np.cumsum(hist_bkg[::-1])[::-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        Z_cumsum = S_cumsum / np.sqrt(S_cumsum + B_cumsum)
        Z_cumsum = np.where(np.isfinite(Z_cumsum), Z_cumsum, 0)

    bin_centers = (bins[1:] + bins[:-1]) / 2
    return S_cumsum, B_cumsum, Z_cumsum, bin_centers


def _plot_histograms(
        ax: plt.Axes,
        datasets: dict[str, tuple[np.ndarray, str, str]],
        bins: int = 200,
        lim: tuple[float | int, float | int] | None = None,
        log_scale: bool = False,
        histtype: str = 'step',
        alpha: float = 1.0,
        density: bool = False,
        linewidth: int = 2
         ) -> tuple[float | None, float | None]:
    """Plot multiple overlaid histograms on shared axes.

    Handles different histogram styles (step, scatter) and manages NaN values
    in data arrays. Supports both linear and log scales.

    Args:
        ax: Matplotlib axes to plot on
        datasets: Dict mapping display labels to (data, color, style) tuples
                  where style is 'step', 'bar', or 'scatter'
        bins: Number of histogram bins
        log_scale: If True, use logarithmic y-axis
        histtype: Matplotlib histogram type ('step' or 'bar')
        alpha: Transparency level for histograms (0-1)
        density: If True, normalize histograms to unit area
        linewidth: Line width for step histograms

    Returns:
        Tuple of (xmin, xmax) data range, or (None, None) if all data is NaN
    """
    all_data = np.concatenate([data for data, _, _ in datasets.values() if len(data) > 0])
    if len(all_data) == 0:
        return 0, 1

    # Use nanmin/nanmax to handle NaN values, but check if all values are NaN
    xmin = np.nanmin(all_data)
    xmax = np.nanmax(all_data)

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        # All values are NaN, use default range
        return None, None

    if lim is not None:
        if xmin < lim[0]: xmin = lim[0]
        if xmax > lim[1]: xmax = lim[1]

    for label, (data, color, style) in datasets.items():
        if len(data) > 0:
            if style == 'scatter':
                n, b = np.histogram(data, bins=bins, range=(xmin, xmax))
                B = (b[:-1] + b[1:]) / 2
                ax.scatter(B, n, color=color, marker='.', label=label)
            else:
                ax.hist(data, bins=bins, range=(xmin, xmax), histtype=histtype,
                        label=label, color=color, linewidth=linewidth, alpha=alpha, density=density)

    if log_scale:
        ax.set_yscale('log')

    return xmin, xmax



##############################
### DATA LOADING FUNCTIONS ###
##############################

def load_data(proc_dir: Path, branches: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load and concatenate data from all chunk files in process directory.

    Optimized for fast loading with low memory consumption by:
    - Concatenating jagged arrays before flattening (cheaper than repeated flattens)
    - Using awkward concatenation for metadata-only operations
    - Sequential I/O (more efficient than threaded I/O for uproot)

    Args:
        proc_dir: Path to process directory containing chunk<N>.root files
        branches: Optional list of specific branches to load. If None, loads all branches.

    Returns:
        Dictionary mapping variable names to flattened numpy arrays
    """
    if not proc_dir.exists():
        LOGGER.warning(f'Process directory not found: {proc_dir}')
        return {}

    # Find all chunk files
    chunk_files = sorted(glob(str(proc_dir / 'chunk*.root')))

    if not chunk_files:
        chunk_files = [proc_dir]

    LOGGER.info(f'Found {len(chunk_files)} chunk files in {proc_dir.name}')

    try:
        # Initialize data dictionary for concatenation
        accumulated_data: dict[str, list] = {}

        # Read all chunk files sequentially
        for chunk_file in tqdm(chunk_files, desc='Loading chunks'):
            with uproot.open(chunk_file) as file:
                tree = file['events']

                # Determine which branches to load
                keys_to_load = branches if branches else tree.keys()

                # Read all branches at once
                arrays = tree.arrays(keys_to_load, library='ak')

                # Read branches
                for name in keys_to_load:
                    try:
                        if name not in accumulated_data:
                            accumulated_data[name] = []
                        accumulated_data[name].append(arrays[name])
                    except Exception as e:
                        LOGGER.warning(f'Could not read branch {name}:\n{e}')

        # Concatenate jagged arrays first (cheap, metadata only), then flatten once (expensive)
        final_data: dict[str, np.ndarray] = {}
        for branch_name, arrays in accumulated_data.items():
            concatenated = ak.concatenate(arrays)
            # Flatten jagged arrays to 1D numpy arrays
            final_data[branch_name] = np.asarray(
                ak.flatten(concatenated) if isinstance(concatenated, ak.Array) else concatenated
            )

        LOGGER.info(f'Successfully loaded {len(final_data)} branches')
        return final_data

    except Exception as e:
        LOGGER.error(f'Error loading data from {proc_dir}: {e}')
        return {}



##########################
### PLOTTING FUNCTIONS ###
##########################

def n_radiated(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        scale: bool = False,
        format: list[str] = ['png']
         ) -> None:
    """Plot histogram of number of radiated photons per lepton pair.

    Shows the distribution of photon multiplicities, useful for understanding
    FSR activity and reconstruction efficiency as a function of radiation level.

    Args:
        data: Dictionary of numpy arrays from ROOT file
        outDir: Output directory for plots
        label: LaTeX label for the final state
        scale: If True, use logarithmic y-axis
    """
    if 'n_radiated' not in data:
        LOGGER.warning('n_radiated branch not found, skipping n_radiated distribution')
        return

    n_radiated = data['n_radiated'].astype(int)

    fig, ax = _make_fig()
    try:
        # Get unique values and counts
        unique_vals, counts = np.unique(n_radiated, return_counts=True)

        # Create bar plot
        ax.bar(unique_vals, counts, width=0.8, color='steelblue', edgecolor='black')

        set_labels(ax, 'Number of Radiated Photons', 'Number of Leptons', right=label)
        ax.set_xticks(unique_vals)
        ax.set_xlim(unique_vals.min() - 0.5, unique_vals.max() + 0.5)

        if scale:
            ax.set_yscale('log')

        ax.grid(axis='x')

        suffix = '_log' if scale else '_lin'
        savefigs(fig, outDir, 'n_radiated', suffix, format)

    finally:
        plt.close(fig)


def photon_distributions(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 100,
        scale: bool = False,
        format: list[str] = ['png']
         ) -> None:
    """Plot photon distributions split by ISR/FSR origin.

    Compares reconstructed photon kinematics against true values, separated by
    whether photons originate from Initial State Radiation (ISR), Final State
    Radiation (FSR), or other sources.

    Note: Arrays are flattened from jagged structure where each event can have
    multiple photons. After flattening, index i corresponds to the i-th photon
    across all events, so indices between origin masks and photon variables align.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
        scale: If True, use logarithmic y-axis
    """
    if 'fromISR' not in data or 'fromFSR' not in data:
        LOGGER.warning('ISR/FSR origin branches not found, skipping photon distributions')
        return

    from_isr = data['fromISR'].astype(bool)
    from_fsr = data['fromFSR'].astype(bool)
    from_other = ~from_isr & ~from_fsr

    variables = [
        ('ph_p',     'PH_p',     r'$p_\gamma$ [GeV]'),
        ('ph_pT',    'PH_pT',    r'$p_{T,\gamma}$ [GeV]'),
        ('ph_theta', 'PH_theta', r'$\theta_\gamma$'),
    ]

    for var_name, true_var, xlabel in variables:
        if var_name not in data or true_var not in data:
            LOGGER.debug(f'Variable {var_name} not found, skipping')
            continue

        true_data = data[true_var]

        # Validate array lengths match
        if len(true_data) != len(from_isr):
            LOGGER.warning(f'Array length mismatch for {var_name:<15}: '
                           f'{len(true_data)} vs {len(from_isr)}, skipping')
            continue

        # Create datasets dict for helper function
        datasets = {
            'ISR':              (true_data[from_isr],   'blue',  'step'),
            'FSR':              (true_data[from_fsr],   'green', 'step'),
            'Other radiations': (true_data[from_other], 'red',   'step'),
            'Reco':             (data[var_name],        'black', 'scatter'),
        }

        fig, ax = _make_fig()
        try:
            xmin, xmax = _plot_histograms(ax, datasets, bins, log_scale=scale)
            ax.set_xlim(xmin, xmax)
            set_labels(ax, xlabel, 'Number of Photons', right=label)
            ax.legend()

            suffix = '_log' if scale else '_lin'
            out = outDir / 'photon'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, var_name, suffix, format)

        finally:
            plt.close(fig)


def leptons_origin(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 100,
        scale: bool = False,
        iso_name: str = '',
        iso_cut: float | int = 1e10,
        format: list[str] = ['png']
         ) -> None:
    """Plot lepton distributions colored by particle origin.

    Compares reconstructed lepton kinematics against true values, separated by
    decay origin (direct from signal, tau/Z/W/H decay, or hadron decay). Applies
    isolation cut to select clean leptons.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
        scale: If True, use logarithmic y-axis
        iso_cut: Isolation cut threshold for relative isolation
    """
    required_keys = ['leps_iso', 'LEPS_iso', 'LEPS_p', 'LEPS_pT', 'LEPS_theta', 'lepton_origin']
    if not all(k in data for k in required_keys):
        LOGGER.warning('Required branches not found for lepton origin plots')
        return

    name_iso, name_ISO = f'leps_iso{iso_name}', f'LEPS_iso{iso_name}'
    cut = data[name_iso] < iso_cut
    CUT = data[name_ISO] < iso_cut
    origin = np.abs(data['lepton_origin'])[CUT]

    # Pre-compute origin masks
    origin_masks = {
        'From signal':        origin == 0,
        'From $\\tau$ decay': origin == 15,
        'From $Z$ decay':     origin == 23,
        'From $W$ decay':     origin == 24,
        'From $H$ decay':     origin == 25,
        'From hadron decay':  origin > 25,
    }

    origin_colors = {
        'From signal':        'red',
        'From $\\tau$ decay': 'cyan',
        'From $Z$ decay':     'blue',
        'From $W$ decay':     'orange',
        'From $H$ decay':     'green',
        'From hadron decay':  'magenta',
    }

    variables = [
        ('leps_p',     'LEPS_p',     r'$p_{\ell^{\pm}}$ [GeV]'),
        ('leps_pT',    'LEPS_pT',    r'$p_{T,\ell^{\pm}}$ [GeV]'),
        ('leps_theta', 'LEPS_theta', r'$\theta_{\ell^{\pm}}$'),
        ('leps_iso',   'LEPS_iso',   r'$I_{rel}$'),
        ('leps_iso_ch', 'LEPS_iso_ch', r'$I_{rel}$ (only charged)'),
        ('leps_iso_ne', 'LEPS_iso_ne', r'$I_{rel}$ (only neutral)'),
        ('leps_iso_ph', 'LEPS_iso_ph', r'$I_{rel}$ (no photon)'),
        ('leps_iso_PH', 'LEPS_iso_PH', r'$I_{rel}$ (no photon, only charged)'),
    ]

    for var_name, true_var, xlabel in variables:
        if var_name not in data or true_var not in data:
            LOGGER.warning(f'Variable {var_name} not found, skipping')
            continue

        var_data = data[var_name][cut]
        true_data = data[true_var][CUT]

        if len(true_data) != len(origin):
            LOGGER.warning(f'Array length mismatch for {var_name:<15}: skipping')
            continue

        # Create datasets dict
        datasets = {
            label: (true_data[mask], color, 'step')
            for label, mask in origin_masks.items()
            if (color := origin_colors[label])
        }
        datasets['Reco'] = (var_data, 'black', 'scatter')

        fig, ax = _make_fig()
        try:
            xmin, xmax = _plot_histograms(ax, datasets, bins, log_scale=scale)
            if 'iso' in var_name:
                xmax = 10 if xmax > 10 else xmax
            ax.set_xlim(xmin, xmax)
            set_labels(ax, xlabel, 'Number of Leptons', right=label)
            ax.legend()

            suffix = '_log' if scale else '_lin'
            out = outDir / 'origin' / name_iso
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, var_name, suffix, format)

        finally:
            plt.close(fig)


def leptons_fsr_comparison(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 100,
        scale: bool = False,
        format: list[str] = ['png']
         ) -> None:
    """Plot lepton distributions colored by particle origin.
    """
    required_keys = ['leps_iso', 'LEPS_iso', 'LEPS_p', 'LEPS_pT', 'LEPS_theta', 'lepton_origin']
    if not all(k in data for k in required_keys):
        LOGGER.warning('Required branches not found for lepton origin plots')
        return

    variables = [
        ('leps_p',     'fsr_p',     'LEPS_p',     r'$(p_{\ell^{\pm},Reco}-p_{\ell^{\pm},MC})/p_{\ell^{\pm},MC}$ [GeV]',          None),
        ('leps_pT',    'fsr_pT',    'LEPS_pT',    r'$(p_{T,\ell^{\pm},Reco}-p_{T,\ell^{\pm},MC})/p_{T,\ell^{\pm},MC}$ [GeV]',    None),
        ('leps_theta', 'fsr_theta', 'LEPS_theta', r'$(\theta_{\ell^{\pm},Reco}-\theta_{\ell^{\pm},MC})/\theta_{\ell^{\pm},MC}$', None),
    ]

    for var_name, fsr_var, true_var, xlabel, lim in variables:
        if var_name not in data or true_var not in data or fsr_var not in data:
            LOGGER.warning(f'Variable {var_name}, {fsr_var} or {true_var} not found, skipping')
            continue

        var_data  = data[var_name]
        fsr_data  = data[fsr_var]
        true_data = data[true_var]

        if (len(var_data) != len(true_data)) or (len(fsr_data) != len(true_data)):
            LOGGER.warning(f'Array length mismatch for {var_name} or {fsr_var}: skipping')
            continue

        diff_var = (var_data - true_data) / true_data
        diff_fsr = (fsr_data - true_data) / true_data

        datasets = {
            'no FSR': (diff_var, 'blue', 'step'),
            'FSR':    (diff_fsr, 'red',  'step')
        }

        fig, ax = _make_fig()
        try:
            xmin, xmax = _plot_histograms(ax, datasets, bins, lim, scale)

            ax.set_xlim(xmin, xmax)
            set_labels(ax, xlabel, 'Number of Leptons', right=label)
            ax.legend()

            suffix = '_log' if scale else '_lin'
            out = outDir / 'fsr_comparison'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, var_name, suffix, format)

        finally:
            plt.close(fig)


def correlation_distributions(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 200,
        scale: bool = False,
        iso_name: str = '',
        iso_cut: float | int = 1e10,
        format: list[str] = ['png']
         ) -> None:
    """Plot lepton-photon correlation distributions split by origin criterion.

    Analyzes angular and kinematic correlations between leptons and photons,
    separated by whether photon and lepton share a common parent particle.
    Useful for understanding photon-lepton associations in FSR recovery.

    Note: Arrays are flattened from jagged structure where each event can have
    multiple photons. After flattening, index i corresponds to the i-th photon
    across all events, so indices between same_parent and correlation variables align.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
        scale: If True, use logarithmic y-axis
        iso_cut: Isolation cut for lepton pairs
    """
    if 'same_parent' not in data:
        LOGGER.warning("'same_parent' branch not found, skipping correlation distributions")
        return

    cut = data[f'leps_iso{iso_name}_pair'] < iso_cut
    CUT = data[f'LEPS_iso{iso_name}_pair'] < iso_cut

    same_parent = data['same_parent'].astype(bool)[CUT]
    diff_parent = ~same_parent

    variables = [
        ('cosTheta',     'CosTheta',     r'$\cos\theta_{\ell^{\pm}\gamma}$'),
        ('acolinearity', 'Acolinearity', r'Acolinearity'),
        ('acoplanarity', 'Acoplanarity', r'Acoplanarity'),
        ('acopolarity',  'Acopolarity',  r'Acopolarity'),
        ('deltaR',       'DeltaR',       r'$\Delta R$'),
    ]

    for reco_var, true_var, xlabel in variables:
        if reco_var not in data or true_var not in data:
            LOGGER.warning(f'Variables {reco_var}, {true_var} not found, skipping')
            continue

        reco_data = data[reco_var][cut]
        true_data = data[true_var][CUT]

        true_same = true_data[same_parent]
        true_diff = true_data[diff_parent]

        fig, ax = _make_fig()
        try:
            all_data = np.concatenate([reco_data, true_data])
            xmin, xmax = np.nanmin(all_data), np.nanmax(all_data[all_data < 20])

            # Create datasets dict
            datasets = {
                'MC (Same parent)': (true_same, 'green', 'hist'),
                'MC (Diff parent)': (true_diff, 'red',   'hist'),
                'Reco':             (reco_data, 'black', 'scatter'),
            }

            xmin, xmax = _plot_histograms(ax, datasets, bins, log_scale=scale,
                                          histtype='bar', alpha=0.8, density=False)

            set_labels(ax, xlabel, 'Number of Pairs', right=label)
            ax.set_xlim(xmin, xmax)
            ax.legend(loc='upper center')

            suffix = '_log' if scale else '_lin'
            out = outDir / 'correlation' / f'iso{iso_name}'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, reco_var, suffix, format)

        finally:
            plt.close(fig)


def correlation_scan(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        iso_name: str,
        iso_cut: float | int = 1e10,
        incr: float | int = 0.01,
        format: list[str] = ['png']
         ) -> None:
    """Plot correlation-based cut optimization with significance calculations.

    Scans correlation variable thresholds to find cuts that maximize signal
    significance (S/sqrt(S+B)), showing efficiency curves for different
    cut strategies (left-sided vs right-sided).

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        iso_cut: Isolation cut for lepton pairs
        incr: Increment step for cut values
    """
    if 'same_parent' not in data:
        LOGGER.warning("'same_parent' branch not found, skipping correlation scan")
        return

    cut = data[iso_name] < iso_cut
    same_parent = data['same_parent'].astype(bool)[cut]
    diff_parent = ~same_parent

    variables = [
        ('CosTheta',     False, r'$\cos\theta_{\ell^{\pm}\gamma}$'),
        ('Acolinearity', False, r'Acolinearity'),
        ('Acoplanarity', False, r'Acoplanarity'),
        ('Acopolarity',  True,  r'Acopolarity'),
        ('DeltaR',       True,  r'$\Delta R$'),
    ]

    for var, reverse, xlabel in variables:
        if var not in data:
            LOGGER.warning(f'Variable {var} not found, skipping')
            continue

        true_data = data[var][cut]
        cuts = np.arange(np.floor(np.nanmin(true_data)),
                         np.ceil(np.nanmax(true_data[true_data < 20])) + incr, incr)

        total_S, total_B = np.sum(same_parent), np.sum(diff_parent)
        LOGGER.debug(f'Initial: Total S = {total_S:,}, B = {total_B:,}, Z = {total_S / np.sqrt(total_S + total_B) * 100:.2f}')
        LOGGER.debug(f'Computing efficiency vs {var} with {len(cuts)-1} bins...')

        S_cumsum, B_cumsum, Z_cumsum, cuts_center = _compute_cumsum_and_z(
            true_data, same_parent, cuts, reverse
        )

        fig, ax = _make_fig()
        try:
            # Compute plot range
            valid_idx = Z_cumsum > 0
            if np.any(valid_idx):
                valid_x = cuts_center[valid_idx]
                xmin, xmax = valid_x[0], valid_x[-1]
            else:
                xmin, xmax = cuts_center[0], cuts_center[-1]
            margin = (xmax - xmin) * 0.02
            ax.set_xlim(xmin - margin, xmax + margin)

            ax_twin = ax.twinx()

            line1 = ax.scatter(cuts_center, Z_cumsum, color='blue', label='Significance Z')
            set_labels(ax, xlabel, r'Significance $S / \sqrt{S + B}$', right=label)
            ax.tick_params(axis='y', labelcolor='blue')
            ax.yaxis.label.set_color('blue')

            line2 = ax_twin.scatter(cuts_center, S_cumsum, color='green', label='True pairing')
            line3 = ax_twin.scatter(cuts_center, B_cumsum, color='red', label='Fake pairing')
            set_labels(ax_twin, ylabel='Count', left=' ')
            ax_twin.tick_params(axis='y')
            ax_twin.set_yscale('log')
            ax_twin.grid(False, axis='y')

            # Find and mark optimal cut
            max_idx = np.argmax(Z_cumsum)
            opt_iso = cuts_center[max_idx]
            opt_z = Z_cumsum[max_idx]
            opt_s, opt_b = S_cumsum[max_idx], B_cumsum[max_idx]

            rel = r'$\leq$' if reverse else r'$\geq$'
            eff = opt_s / (opt_s + opt_b) * 100
            lab = f'Max Z = {opt_z:,.0f}\neff = {eff:.2f}' + r'\%' + '\n' + f'{xlabel} {rel} {opt_iso:.2f}'
            line4 = ax.scatter([opt_iso], [opt_z], color='red', s=150, marker='*',
                               label=lab, zorder=5)

            # Combine legends
            lines = [line1, line2, line3, line4]
            labels = [l.get_label() for l in lines]
            ymin, ymax = ax_twin.get_ylim()
            ax_twin.set_ylim(ymin, ymax * 100)
            ax.legend(lines, labels, loc='upper center')

            # Save figure
            out = outDir / 'efficiency' / 'correlation' / iso_name
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, f'significance_{var}', format=format)

            rel = '<=' if reverse else '>='
            LOGGER.debug(f'Optimal cut: {var} {rel} {opt_iso:.3f} with Z = {opt_z:,.0f} (S = {opt_s:,}, B = {opt_b:,})')

        finally:
            plt.close(fig)


def significance(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        incr: float | int = 0.1,
        format: list[str] = ['png']
         ) -> float | int:
    """Analyze and plot significance vs isolation cut using cumulative cut analysis.

    Computes significance S/sqrt(S+B) as a function of isolation threshold,
    where S = leptons not from hadronization and B = leptons from hadrons.
    Identifies optimal isolation cut that maximizes significance.

    Args:
        data: Dictionary of numpy arrays from ROOT file
        outDir: Output directory for plots
        label: LaTeX label for the final state
        incr: Increment for isolation cut values

    Returns:
        Optimal isolation cut value
    """

    # Check for required variables
    required_vars = ['lepton_origin', 'LEPS_iso']
    if not all(var in data for var in required_vars):
        LOGGER.warning('Missing required variables for significance computation')
        return

    # Extract data
    origin = np.abs(data['lepton_origin'])
    from_had = origin > 25

    xlabels = [r'$I_{rel}$', r'$I_{rel}$ (only charged)', r'$I_{rel}$ (only neutral)', r'$I_{rel}$ (no photon)']
    for iso, xlabel in zip(['', '_ch', '_ne', '_ph', '_PH'], xlabels):
        Iso_name = f'LEPS_iso{iso}'
        leps_iso = data[Iso_name]

        # Define bins for isolation
        iso_cuts = np.arange(np.floor(np.nanmin(leps_iso)), np.ceil(np.nanmax(leps_iso)) + incr, incr)

        total_S, total_B = np.sum(~from_had), np.sum(from_had)
        LOGGER.debug(f'Initial: Total S = {total_S:,}, B = {total_B:,}, Z = {total_S / np.sqrt(total_S + total_B) * 100:.2f}')
        LOGGER.debug(f'Computing efficiency vs isolation with {len(iso_cuts)-1} bins...')

        S_cumsum, B_cumsum, Z_cumsum, iso_centers = _compute_cumsum_and_z(
            leps_iso, ~from_had, iso_cuts, reverse=True
        )

        fig, ax = _make_fig()
        ax.set_xlim(None, 10)
        ax_twin = ax.twinx()

        line1 = ax.scatter(iso_centers, Z_cumsum, color='blue', label='Significance Z')
        set_labels(ax, r'$I_{rel}$', r'Significance $S / \sqrt{S + B}$', right=label)
        ax.tick_params(axis='y', labelcolor='blue')
        ax.yaxis.label.set_color('blue')

        line2 = ax_twin.scatter(iso_centers, S_cumsum, color='green', label='Not from jet')
        line3 = ax_twin.scatter(iso_centers, B_cumsum, color='red', label='From jet')
        set_labels(ax_twin, ylabel='Count', left=' ')
        ax_twin.tick_params(axis='y')
        ax_twin.set_yscale('log')
        ax_twin.grid(False, axis='y')

        # Find and mark optimal cut
        max_idx = np.argmax(Z_cumsum)
        opt_iso = iso_centers[max_idx]
        opt_z = Z_cumsum[max_idx]
        opt_s, opt_b = S_cumsum[max_idx], B_cumsum[max_idx]

        eff = opt_s / (opt_s + opt_b) * 100
        line4 = ax.scatter([opt_iso], [opt_z], color='red', s=150, marker='*',
                           label=f'Max Z = {opt_z:,.0f}\neff = {eff:.2f}' + r'\%' + '\n' + f'{xlabel} $\\leq$ {opt_iso:.2f}',
                           zorder=5)

        # Combine legends
        lines = [line1, line2, line3, line4]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower right')

        # Save figure
        out = outDir / 'efficiency' / 'isolation'
        out.mkdir(exist_ok=True, parents=True)
        savefigs(fig, out, f'significance_iso{iso}', format=format)
        plt.close(fig)

        LOGGER.info(f'Optimal cut: iso <= {opt_iso:.3f} with Z = {opt_z:,.0f} (S = {opt_s:,}, B = {opt_b:,})')
    return opt_iso
