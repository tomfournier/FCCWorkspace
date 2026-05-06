#!/usr/bin/env python3
"""
Plotting script for FSR recovery analysis.

This script reads FSR recovery results and generates comprehensive plots to visualize:
- Photon distributions split by ISR/FSR origin
- Lepton distributions comparing reconstructed, true, and FSR-recovered quantities
- Correlation distributions split by same_parent criterion
- Distribution of number of radiated photons

Usage:
    python3 plots.py --cat mumu --ecm 240 [--procs process1-process2]
"""

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Standard library and scientific computing imports
import sys
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak

# Start execution timer
t = time()


########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger

parser = create_parser(
    cat_single=True,
    allow_qq=False,
    is_plot=True,
    optimize=True,
    only_procs=True,
    description='FSR Recovery Plots Script'
)
arg = parse_args(parser, validate_cat=True)
set_log(arg)

LOGGER = get_logger(__name__)


##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Import configuration paths and plot settings
from package.userConfig import loc, plot_file
# Import utilities and plotting configurations
from package.config import timer, process_label
from package.plots.python.plotter import (
    set_plt_style,
    set_labels,
    savefigs
)


################################
### MATPLOTLIB STYLE SETTING ###
################################

set_plt_style()
plt.ioff()


#######################
### HELPER FUNCTION ###
#######################

def _make_fig(figsize: tuple = (12, 8),
              dpi: int = 100
              ) -> tuple[plt.Figure, plt.Axes]:
    """Create a matplotlib figure and axes."""
    return plt.subplots(figsize=figsize, dpi=dpi)


def load_data(proc_dir: Path) -> dict[str, np.ndarray]:
    """Load and concatenate data from all chunk files in process directory.

    Handles jagged arrays by flattening them into 1D arrays for plotting.

    Args:
        proc_dir: Path to process directory containing chunk<N>.root files

    Returns:
        Dictionary mapping variable names to flattened numpy arrays
    """
    if not proc_dir.exists():
        LOGGER.warning(f'Process directory not found: {proc_dir}')
        return {}

    # Find all chunk files
    chunk_files = sorted(glob(str(proc_dir / 'chunk*.root')))

    if not chunk_files:
        LOGGER.warning(f'No chunk files found in {proc_dir}')
        return {}

    LOGGER.info(f'Found {len(chunk_files)} chunk files in {proc_dir.name}')

    try:
        # Initialize data dictionary for concatenation
        accumulated_data: dict[str, list] = {}

        # Read all chunk files
        for chunk_file in chunk_files:
            with uproot.open(chunk_file) as file:
                tree = file['events']

                # Read all branches
                for key in tree.keys():
                    try:
                        # Read as awkward array
                        arr = tree[key].array(library='ak')

                        if key not in accumulated_data:
                            accumulated_data[key] = []

                        accumulated_data[key].append(arr)

                    except Exception as e:
                        LOGGER.warning(f'Could not read branch {key}: {e}')

        # Concatenate all accumulated data
        final_data: dict[str, np.ndarray] = {}
        for branch_name, arrays in accumulated_data.items():
            concatenated = ak.concatenate(arrays)

            # Flatten jagged arrays (convert to 1D numpy arrays for plotting)
            if isinstance(concatenated, ak.Array):
                # This is a jagged array, flatten it
                flattened = ak.flatten(concatenated)
                final_data[branch_name] = np.asarray(flattened)
            else:
                # Regular array
                final_data[branch_name] = np.asarray(concatenated)

        LOGGER.info(f'Successfully loaded {len(final_data)} branches')
        return final_data

    except Exception as e:
        LOGGER.error(f'Error loading data from {proc_dir}: {e}')
        return {}


##########################
### PLOTTING FUNCTIONS ###
##########################

def photon_distributions(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 100,
        scale: bool = False
         ) -> None:
    """Plot photon distributions split by ISR/FSR origin.

    Note: Arrays are flattened from jagged structure where each event can have
    multiple photons. After flattening, index i corresponds to the i-th photon
    across all events, so indices between fromISR/fromFSR and photon variables align.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
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
        if var_name not in data:
            LOGGER.debug(f'Variable {var_name} not found, skipping')
            continue

        var_data  = data[var_name]
        true_data = data[true_var]

        # Validate array lengths match
        if len(true_data) != len(from_isr):
            LOGGER.warning(f'Array length mismatch for {var_name:<15}: '
                           f'{len(var_data)} vs {len(from_isr)}, skipping')
            continue

        isr_data   = true_data[from_isr]
        fsr_data   = true_data[from_fsr]
        other_data = true_data[from_other]

        fig, ax = _make_fig()
        try:

            # Determine common range
            all_data = np.concatenate([var_data, isr_data, fsr_data])

            if len(all_data) == 0:
                LOGGER.warning(f'No valid data for {var_name}, skipping')
                plt.close(fig)
                continue

            xmin, xmax = np.min(all_data), np.max(all_data)

            if len(isr_data) > 0:
                ax.hist(isr_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label='ISR', color='blue', linewidth=2)
            if len(fsr_data) > 0:
                ax.hist(fsr_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label='FSR', color='green', linewidth=2)
            if len(other_data) > 0:
                ax.hist(other_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label='Other radiations', color='red', linewidth=2)
            if len(var_data) > 0:
                n, b = np.histogram(var_data, bins=bins, range=(xmin, xmax))
                B = (b[:-1] + b[1:]) / 2
                ax.scatter(B, n, color='black', marker='.', label='Reco')

            ax.set_xlim(xmin, xmax)

            set_labels(ax, xlabel, 'Number of Photons', right=label)
            if scale:
                ax.set_yscale('log')
            ax.legend()

            suffix = '_log' if scale else '_lin'
            out = outDir / 'photon'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, var_name, suffix=suffix, format=plot_file)

        finally:
            plt.close(fig)


def leptons_origin(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 100,
        scale: bool = False,
        iso_cut: float | int = 1e10
         ) -> None:
    """Plot photon distributions split by ISR/FSR origin.

    Note: Arrays are flattened from jagged structure where each event can have
    multiple photons. After flattening, index i corresponds to the i-th photon
    across all events, so indices between fromISR/fromFSR and photon variables align.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
    """
    if 'fromISR' not in data or 'fromFSR' not in data:
        LOGGER.warning('ISR/FSR origin branches not found, skipping photon distributions')
        return

    cut = data['leps_iso'] < iso_cut
    CUT = data['LEPS_iso'] < iso_cut
    origin = np.abs(data['lepton_origin'])[CUT]

    from_ini = origin == 0
    from_tau = origin == 15
    from_Z   = origin == 23
    from_W   = origin == 24
    from_H   = origin == 25
    from_had = origin > 25

    variables = [
        ('leps_p',     'LEPS_p',     r'$p_{\ell^{\pm}}$ [GeV]'),
        ('leps_pT',    'LEPS_pT',    r'$p_{T,\ell^{\pm}}$ [GeV]'),
        ('leps_theta', 'LEPS_theta', r'$\theta_{\ell^{\pm}}$'),
        ('leps_iso',   'LEPS_iso',   r'$I_{rel}$'),
    ]

    for var_name, true_var, xlabel in variables:
        if var_name not in data:
            LOGGER.debug(f'Variable {var_name} not found, skipping')
            continue

        var_data  = data[var_name][cut]
        true_data = data[true_var][CUT]

        # Validate array lengths match
        if len(true_data) != len(origin):
            LOGGER.warning(f'Array length mismatch for {var_name:<15}: '
                           f'{len(var_data)} vs {len(origin)}, skipping')
            continue

        ini_data = true_data[from_ini]
        tau_data = true_data[from_tau]
        Z_data   = true_data[from_Z]
        W_data   = true_data[from_W]
        H_data   = true_data[from_H]
        had_data = true_data[from_had]

        fig, ax = _make_fig()
        try:

            # Determine common range
            all_data = np.concatenate([var_data, ini_data, tau_data, Z_data, W_data, H_data, had_data])

            if len(all_data) == 0:
                LOGGER.warning(f'No valid data for {var_name}, skipping')
                plt.close(fig)
                continue

            xmin, xmax = np.min(all_data), np.max(all_data)
            if 'iso' in var_name:
                xmax = 10

            if len(ini_data) > 0:
                ax.hist(ini_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label=r'From signal', color='red', linewidth=2)
            if len(tau_data) > 0:
                ax.hist(tau_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label=r'From $\tau$ decay', color='cyan', linewidth=2)
            if len(Z_data) > 0:
                ax.hist(Z_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label=r'From $Z$ decay', color='blue', linewidth=2)
            if len(W_data) > 0:
                ax.hist(W_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label=r'From $W$ decay', color='orange', linewidth=2)
            if len(H_data) > 0:
                ax.hist(H_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label=r'From $H$ decay', color='green', linewidth=2)
            if len(had_data) > 0:
                ax.hist(had_data, bins=bins, range=(xmin, xmax), histtype='step',
                        label='From hadron decay', color='magenta', linewidth=2)
            if len(var_data) > 0:
                n, b = np.histogram(var_data, bins=bins, range=(xmin, xmax))
                B = (b[:-1] + b[1:]) / 2
                ax.scatter(B, n, color='black', marker='.', label='Reco')

            ax.set_xlim(xmin, xmax)

            set_labels(ax, xlabel, 'Number of Leptons', right=label)
            if scale:
                ax.set_yscale('log')
            ax.legend()

            suffix = '_log' if scale else '_lin'
            out = outDir / 'origin'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, var_name, suffix=suffix, format=plot_file)

        finally:
            plt.close(fig)


def lepton_distributions(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 200,
        scale: bool = False
         ) -> None:
    """Plot lepton distributions comparing reconstructed, true, and FSR-recovered quantities.

    Note: Arrays are flattened from jagged structure where each event can have
    multiple leptons. After flattening, index i corresponds to the i-th lepton
    across all events, so indices align between all lepton variables.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
    """
    variables = [
        ('leps_p',     'leps_FSR_p',     'LEPS_FSR_p',     r'$p_{\ell}$'),
        ('leps_pT',    'leps_FSR_pT',    'LEPS_FSR_pT',    r'$p_{T,\ell}$'),
        ('leps_theta', 'leps_FSR_theta', 'LEPS_FSR_theta', r'$\theta_{\ell}$'),
    ]

    for reco_var, reco_fsr_var, true_fsr_var, xlabel in variables:
        # Check if all variables exist
        required_vars = [reco_var, reco_fsr_var, true_fsr_var]
        if not all(var in data for var in required_vars):
            LOGGER.debug(f'Missing variables for {xlabel}, skipping')
            continue

        fig, ax = _make_fig()
        try:
            # Flatten arrays if needed
            reco_data = data[reco_var]
            reco_fsr_data = data[reco_fsr_var]
            true_fsr_data = data[true_fsr_var]

            # Determine common range
            all_data = np.concatenate([reco_data, reco_fsr_data, true_fsr_data])

            if len(all_data) == 0:
                LOGGER.warning(f'No valid data for {reco_var}, skipping')
                plt.close(fig)
                continue

            xmin, xmax = np.min(all_data), np.max(all_data)

            # Reconstructed vs True
            ax.hist(reco_data, bins=bins, range=(xmin, xmax), histtype='step',
                    label='Reconstructed', color='blue', density=False, linewidth=2)

            # FSR-recovered vs True FSR
            ax.hist(reco_fsr_data, bins=bins, range=(xmin, xmax), histtype='step',
                    label='FSR-Recovered (ILC)', color='red', density=False, linewidth=2, linestyle='dashed')
            ax.hist(true_fsr_data, bins=bins, range=(xmin, xmax),
                    label='True (pre-FSR)', color='green', density=False)

            set_labels(ax, xlabel, 'Number of Leptons', right=label)
            if scale:
                ax.set_yscale('log')

            ax.set_xlim(xmin, xmax)
            ax.legend()

            suffix = '_log' if scale else '_lin'
            out = outDir / 'lepton'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, reco_var, suffix=suffix, format=plot_file)

        finally:
            plt.close(fig)




def correlation_distributions(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        bins: int = 200,
        scale: bool = False,
        iso_cut: float | int = 1e10
         ) -> None:
    """Plot correlation distributions split by same_parent criterion.

    Note: Arrays are flattened from jagged structure where each event can have
    multiple photons. After flattening, index i corresponds to the i-th photon
    across all events, so indices between same_parent and correlation variables align.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
    """
    if 'same_parent' not in data:
        LOGGER.warning("'same_parent' branch not found, skipping correlation distributions")
        return

    cut = data['leps_iso_pair'] < iso_cut
    CUT = data['LEPS_iso_pair'] < iso_cut

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
            # Determine common range
            all_data = np.concatenate([reco_data, true_data])
            xmin, xmax = np.nanmin(all_data), np.nanmax(all_data)

            # Same parent
            if len(reco_data) > 0:
                n, b = np.histogram(reco_data, bins=bins, range=(xmin, xmax))
                B = (b[:-1] + b[1:]) / 2
                ax.scatter(B, n, color='black', marker='.', label='Reco')

            if len(true_same) > 0:
                ax.hist(true_same, bins=bins, range=(xmin, xmax), alpha=0.8,
                        label='MC (Same parent)', color='green', density=False)

            # Different parent
            if len(true_diff) > 0:
                ax.hist(true_diff, bins=bins, range=(xmin, xmax), alpha=0.8,
                        label='MC (Diff parent)', color='red', density=False)

            set_labels(ax, xlabel, 'Number of Pairs', right=label)

            ax.set_xlim(xmin, xmax)
            if scale:
                ax.set_yscale('log')
            ax.legend(loc='upper center')

            suffix = '_log' if scale else '_lin'
            out = outDir / 'correlation'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, reco_var, suffix=suffix, format=plot_file)

        finally:
            plt.close(fig)


def correlation_scan(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        iso_cut: float | int = 1e10,
        incr: float | int = 0.01
         ) -> None:
    """Plot correlation distributions split by same_parent criterion.

    Note: Arrays are flattened from jagged structure where each event can have
    multiple photons. After flattening, index i corresponds to the i-th photon
    across all events, so indices between same_parent and correlation variables align.

    Args:
        data: Dictionary of numpy arrays from ROOT file (already flattened)
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histograms
    """
    if 'same_parent' not in data:
        LOGGER.warning("'same_parent' branch not found, skipping correlation distributions")
        return

    cut = data['LEPS_iso_pair'] < iso_cut

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
            LOGGER.warning(f'Variables {var} not found, skipping')
            continue

        true_data = data[var][cut]
        cuts = np.arange(np.floor(np.nanmin(true_data)), np.ceil(np.nanmax(true_data)) + incr, incr)

        total_S, total_B = np.sum(same_parent), np.sum(diff_parent)
        LOGGER.info(f'Initial: Total S = {total_S:,}, B = {total_B:,}, Z = {total_S / np.sqrt(total_S + total_B) * 100:.2f}')
        LOGGER.info(f'Computing efficiency vs {var} with {len(cuts)-1} bins...')

        # Create 1D histograms
        hist_sig, _ = np.histogram(true_data[same_parent], bins=cuts)
        hist_bkg, _ = np.histogram(true_data[diff_parent], bins=cuts)

        # Compute cumulative sums for cuts
        if reverse:
            S_cumsum, B_cumsum = np.cumsum(hist_sig), np.cumsum(hist_bkg)
        else:
            S_cumsum, B_cumsum = np.cumsum(hist_sig[::-1])[::-1], np.cumsum(hist_bkg[::-1])[::-1]

        Z_cumsum = np.zeros_like(S_cumsum, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_cumsum = S_cumsum / np.sqrt(S_cumsum + B_cumsum)
            Z_cumsum[np.isnan(Z_cumsum)] = 0
            Z_cumsum[np.isinf(Z_cumsum)] = 0

        cuts_center = (cuts[1:] + cuts[:-1]) / 2


        fig, ax = _make_fig()
        try:
            # Determine x-range based on non-zero Z_cumsum values (exclude NaN/inf regions)
            valid_idx = Z_cumsum > 0
            if np.any(valid_idx):
                valid_x = cuts_center[valid_idx]
                xmin, xmax = valid_x[0], valid_x[-1]
            else:
                xmin, xmax = cuts_center[0], cuts_center[-1]
            # Add small margin for clarity
            margin = (xmax - xmin) * 0.02
            ax.set_xlim(xmin - margin, xmax + margin)
            ax_twin = ax.twinx()

            line1 = ax.scatter(cuts_center, Z_cumsum, color='blue', label='Significance Z')
            set_labels(ax, xlabel, r'Significance $S / \sqrt{S + B}$', right=label)
            ax.tick_params(axis='y', labelcolor='blue')
            ax.yaxis.label.set_color('blue')

            line2 = ax_twin.scatter(cuts_center, S_cumsum, color='green', label='True pairing')
            line3 = ax_twin.scatter(cuts_center, B_cumsum, color='red',   label='Fake pairing')
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
            line4 = ax.scatter([opt_iso], [opt_z], color='red', s=150, marker='*',
                               label=fr'Max Z = {opt_z:,.0f} at {xlabel} {rel} {opt_iso:.2f}', zorder=5)

            # Combine legends
            lines = [line1, line2, line3, line4]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='lower right')

            # Save figure
            out = outDir / 'efficiency'
            out.mkdir(exist_ok=True, parents=True)
            savefigs(fig, out, f'significance_{var}', format=plot_file)
            plt.close(fig)

            rel = '<=' if reverse else '>='
            LOGGER.info(f'Optimal cut: {var} {rel} {opt_iso:.3f} with Z = {opt_z:,.0f} (S = {opt_s:,}, B = {opt_b:,})')

        finally:
            plt.close(fig)


def n_radiated(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        scale: bool = False
         ) -> None:
    """Plot distribution of number of radiated photons.

    Args:
        data: Dictionary of numpy arrays from ROOT file
        outDir: Output directory for plots
        label: LaTeX label for the final state
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
        savefigs(fig, outDir, 'n_radiated', suffix=suffix, format=plot_file)

    finally:
        plt.close(fig)


def significance(
        data: dict[str, np.ndarray],
        outDir: Path,
        label: str,
        incr: float | int = 0.1,
         ) -> float | int:
    """Analyze significance vs isolation using cumulative cut analysis.

    Creates significance curves showing how significance changes when applying
    cumulative cuts: keeping only pairs with leps_iso <= iso_cut.

    Args:
        data: Dictionary of numpy arrays from ROOT file
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for iso axis
        scale: Whether to use logarithmic scale
    """

    # Check for required variables
    required_vars = ['lepton_origin', 'LEPS_iso']
    if not all(var in data for var in required_vars):
        LOGGER.warning('Missing required variables for efficiency computation')
        return

    # Extract data
    origin = np.abs(data['lepton_origin'])
    from_had = origin > 25
    leps_iso = data['LEPS_iso']

    # Define bins for isolation
    iso_cuts = np.arange(np.floor(np.nanmin(leps_iso)), np.ceil(np.nanmax(leps_iso)) + incr, incr)

    # Calculate cumulative S, B, and Z for each iso cut
    total_S, total_B = np.sum(~from_had), np.sum(from_had)

    LOGGER.info(f'Initial: Total S = {total_S:,}, B = {total_B:,}, Z = {total_S / np.sqrt(total_S + total_B) * 100:.2f}')
    LOGGER.info(f'Computing efficiency vs isolation with {len(iso_cuts)-1} bins...')

    # Create 1D histograms
    hist_sig, _ = np.histogram(leps_iso[~from_had], bins=iso_cuts)
    hist_bkg, _ = np.histogram(leps_iso[from_had],  bins=iso_cuts)

    # Compute cumulative sums for cuts: keep pairs with iso <= iso_cut
    S_cumsum, B_cumsum = np.cumsum(hist_sig), np.cumsum(hist_bkg)

    # Calculate efficiency: eff = S / (S + B)
    Z_cumsum = np.zeros_like(S_cumsum, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_cumsum = S_cumsum / np.sqrt(S_cumsum + B_cumsum)
        Z_cumsum[np.isnan(Z_cumsum)] = 0
        Z_cumsum[np.isinf(Z_cumsum)] = 0

    # Compute bin centers for plotting
    iso_centers = (iso_cuts[:-1] + iso_cuts[1:]) / 2

    # Create main plot with significance and S/B
    fig, ax = _make_fig()
    ax.set_xlim(None, 10)
    ax_twin = ax.twinx()

    line1 = ax.scatter(iso_centers, Z_cumsum, color='blue', label='Significance Z')
    set_labels(ax, r'$I_{rel}$', r'Significance $S / \sqrt{S + B}$', right=label)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.yaxis.label.set_color('blue')

    line2 = ax_twin.scatter(iso_centers, S_cumsum, color='green', label='Not from jet')
    line3 = ax_twin.scatter(iso_centers, B_cumsum, color='red',   label='From jet')
    set_labels(ax_twin, ylabel='Count', left=' ')
    ax_twin.tick_params(axis='y')
    ax_twin.set_yscale('log')
    ax_twin.grid(False, axis='y')

    # Find and mark optimal cut
    max_idx = np.argmax(Z_cumsum)
    opt_iso = iso_centers[max_idx]
    opt_z = Z_cumsum[max_idx]
    opt_s, opt_b = S_cumsum[max_idx], B_cumsum[max_idx]

    line4 = ax.scatter([opt_iso], [opt_z], color='red', s=150, marker='*',
                       label=fr'Max Z = {opt_z:,.0f} at $I_{{rel}}$ = {opt_iso:.2f}', zorder=5)

    # Combine legends
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right')

    # Save figure
    out = outDir / 'efficiency'
    out.mkdir(exist_ok=True, parents=True)
    savefigs(fig, out, 'significance_iso', format=plot_file)
    plt.close(fig)

    LOGGER.info(f'Optimal cut: iso <= {opt_iso:.3f} with Z = {opt_z:,.0f} (S = {opt_s:,}, B = {opt_b:,})')
    return opt_iso




##########################
### EXECUTION FUNCTION ###
##########################

def main():
    """Main plotting function"""
    cat, ecm = arg.cat, arg.ecm

    # Get input and output directories
    inDir  = loc.get('FSR_TREE',  cat, ecm, type=Path)
    outDir = loc.get('PLOTS_FSR', cat, ecm, type=Path)

    if not inDir.exists():
        LOGGER.error(f'Input directory not found: {inDir}')
        sys.exit(1)

    # Create output directory
    outDir.mkdir(exist_ok=True, parents=True)

    # Set LaTeX label
    if cat == 'mumu':
        label = r'$Z(\rightarrow\mu^+\mu^-)H'
    elif cat == 'ee':
        label = r'$Z(\rightarrow e^+e^-)H'
    else:
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')

    # Get process directories
    if arg.procs == '':
        proc_dirs = sorted([d for d in inDir.iterdir() if d.is_dir()])
    else:
        proc_dirs = [inDir / p for p in arg.procs.split('-')]

    if not proc_dirs:
        LOGGER.error(f'No process directories found in {inDir}')
        sys.exit(1)

    LOGGER.info(f'Found {len(proc_dirs)} process(es) to process')

    # Process each directory
    for proc_dir in proc_dirs:
        if not proc_dir.exists():
            LOGGER.warning(f'Process directory not found: {proc_dir}, skipping')
            continue

        proc = proc_dir.name
        LOGGER.info(f'Processing {proc}')

        proc_label = proc.replace(f'wzp6_ee_{cat}H', '').replace(f'_ecm{ecm}', '').replace('_H', '')
        if proc_label:
            Label = label + r'(\rightarrow ' + process_label[proc_label] + ')$'
        else:
            Label = label + r')$'

        # Load data from all chunk files
        data = load_data(proc_dir)

        if not data:
            LOGGER.warning(f'Could not load data from {proc_dir}, skipping')
            continue

        # Create subdirectory for this process
        proc_outDir = outDir / proc_dir.name
        proc_outDir.mkdir(exist_ok=True, parents=True)

        # Generate significance plot
        cut = significance(data, proc_outDir, Label)
        correlation_scan(data, proc_outDir, Label, cut)

        # Generate plots
        LOGGER.info(f'Generating plots for {proc_dir.name}')
        for scale in [False, True]:
            photon_distributions(data, proc_outDir, Label, scale=scale)
            lepton_distributions(data, proc_outDir, Label, scale=scale)
            leptons_origin(data, proc_outDir, Label, scale=scale, iso_cut=cut)
            correlation_distributions(data, proc_outDir, Label, scale=scale, iso_cut=cut)
            n_radiated(data, proc_outDir, Label, scale=scale)

        LOGGER.info(f'Plots for {proc_dir.name} saved to {proc_outDir}\n')


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    try:
        main()
    finally:
        timer(t)
