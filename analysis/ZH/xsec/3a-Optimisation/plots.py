#!/usr/bin/env python3
"""
Plotting script for chi2_recoil_frac optimization results.

This script reads the JSON results from optimize_ll.py and generates
comprehensive plots to visualize the pairing efficiency as a function
of the chi2_recoil_frac parameter.

Usage:
    python3 plots.py --cat mumu --ecm 240 [--procs process1-process2]
"""

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

print('----->[Info] Loading modules')

# Standard library and scientific computing imports
from time import time
from pathlib import Path
import json
import sys
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import uproot

# Start execution timer
t = time()

print('----->[Info] Loading custom modules')

# Import configuration paths and plot settings
from package.userConfig import loc, plot_file
# Import utilities and plotting configurations
from package.config import timer
from package.plots.python.plotter import (
    set_plt_style,
    set_labels,
    savefigs
)


########################
### ARGUMENT PARSING ###
########################

# Command-line argument parsing
parser = ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu)',
                    choices=['ee', 'mumu'], type=str, default='mumu')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)',
                    choices=[240, 365], type=int, default=240)
parser.add_argument('--procs', help='Process(es) to plot', type=str, default='')
arg = parser.parse_args()


set_plt_style()
plt.ioff()



#######################
### HELPER FUNCTION ###
#######################

def _make_fig(figsize: tuple = (12, 8),
              dpi: int = 100
              ) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=figsize, dpi=dpi)


##########################
### PLOTTING FUNCTIONS ###
##########################

def load_results(results_file: Path) -> dict[str, dict]:
    """Load optimization results from JSON file.

    Args:
        results_file: Path to the results.json file

    Returns:
        Dictionary with results keyed by chi2_frac
    """
    if not results_file.exists():
        print(f"Warning: Results file not found: {results_file}")
        return {}

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def extract_arrays(results: dict[str, dict]) -> tuple[np.ndarray, ...]:
    """Extract numpy arrays from results dictionary.

    Args:
        results: Results dictionary from JSON

    Returns:
        Tuple of (chi2_fracs, efficiencies, n_correct, n_partial, n_incorrect, n_total)
    """
    if not results:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    chi2_fracs = np.array([float(k) for k in results.keys()])
    efficiencies = np.array([results[k]['efficiency'] for k in results.keys()])
    n_correct = np.array([results[k]['n_correct'] for k in results.keys()])
    n_partial = np.array([results[k]['n_partial'] for k in results.keys()])
    n_incorrect = np.array([results[k]['n_incorrect'] for k in results.keys()])
    n_total = np.array([results[k]['n_total'] for k in results.keys()])

    # Sort by chi2_frac
    sort_idx = np.argsort(chi2_fracs)

    return (chi2_fracs[sort_idx], efficiencies[sort_idx],
            n_correct[sort_idx], n_partial[sort_idx],
            n_incorrect[sort_idx], n_total[sort_idx])


def efficiency(
        chi2_fracs: np.ndarray,
        efficiencies: np.ndarray,
        outDir: Path,
        label: str,
        full_range: bool = False) -> None:
    """Plot efficiency vs chi2_recoil_frac.

    Args:
        chi2_fracs: Array of chi2_recoil_frac values
        efficiencies: Array of efficiency values
        proc_name: Process name for title
        cat: Final state category (ee or mumu)
        outDir: Output directory for plots
    """
    if len(chi2_fracs) == 0:
        return

    # Find optimal chi2_frac (maximum efficiency)
    optimal_idx = np.argmax(efficiencies)
    optimal_chi2 = chi2_fracs[optimal_idx]
    optimal_eff = efficiencies[optimal_idx]

    # Create figure
    fig, ax = _make_fig()
    try:
        # Plot efficiency curve
        ax.scatter(chi2_fracs, efficiencies, marker='.')

        # Highlight optimal point
        ax.scatter(optimal_chi2, optimal_eff, color='r', marker='*', s=150)

        # Add vertical line at optimal point
        ax.axvline(x=optimal_chi2, color='gray', alpha=0.8,
                   label=f'Optimal: $\\chi^2_{{frac}}$ = {optimal_chi2:.2f}')

        ax.set_xlim(0, 1)
        if full_range: ax.set_ylim(0, 1)

        set_labels(ax, r'$\chi^2_{recoil\_frac}$',
                   'Pairing Efficiency', right=label)
        ax.legend()

        savefigs(fig, outDir, 'efficiency', format=plot_file)

    finally:
        plt.close(fig)


def pairing_composition(
        chi2_fracs: np.ndarray,
        n_correct: np.ndarray,
        n_partial: np.ndarray,
        n_incorrect: np.ndarray,
        n_total: np.ndarray,
        outDir: Path,
        label) -> None:
    """Plot composition of pairing results (correct, partial, incorrect).

    Args:
        chi2_fracs: Array of chi2_recoil_frac values
        n_correct: Number of correct pairings
        n_partial: Number of partial pairings
        n_incorrect: Number of incorrect pairings
        n_total: Total number of events
        proc_name: Process name for title
        cat: Final state category (ee or mumu)
        outDir: Output directory for plots
    """
    if len(chi2_fracs) == 0:
        return

    # Create figure
    fig, ax = _make_fig()

    try:
        # Plot stacked area
        ax.fill_between(chi2_fracs, 0, n_correct / n_total * 100,
                        label='Both Correct', color='green', alpha=0.7)
        ax.fill_between(chi2_fracs, n_correct / n_total * 100,
                        (n_correct + n_partial) / n_total * 100,
                        label='One Correct', color='orange', alpha=0.7)
        ax.fill_between(chi2_fracs, (n_correct + n_partial) / n_total * 100,
                        (n_correct + n_partial + n_incorrect) / n_total * 100,
                        label='Both Incorrect', color='red', alpha=0.7)

        # Labels and formatting
        set_labels(ax, r'$\chi^2_{recoil\_frac}$', 'Percentage of Events', right=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower center')

        # Save figure
        savefigs(fig, outDir, 'pairing_composition', format=plot_file)

    finally:
        plt.close(fig)


def event_counts(
        chi2_fracs: np.ndarray,
        n_correct: np.ndarray,
        n_partial: np.ndarray,
        n_incorrect: np.ndarray,
        outDir: Path,
        label: str) -> None:
    """Plot absolute event counts for different pairing outcomes.

    Args:
        chi2_fracs: Array of chi2_recoil_frac values
        n_correct: Number of correct pairings
        n_partial: Number of partial pairings
        n_incorrect: Number of incorrect pairings
        proc_name: Process name for title
        cat: Final state category (ee or mumu)
        outDir: Output directory for plots
    """
    if len(chi2_fracs) == 0:
        return

    # Create figure
    fig, ax = _make_fig()

    try:
        # Plot lines for each category
        ax.scatter(chi2_fracs, n_correct, color='green', marker='.', label='Both Correct')
        ax.scatter(chi2_fracs, n_partial, color='orange', marker='.', label='One Correct')
        ax.scatter(chi2_fracs, n_incorrect, color='red', marker='.',label='Both Incorrect')

        # Labels and formatting
        set_labels(ax, r'$\chi^2_{recoil\_frac}$', 'Number of Events', right=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.legend()

        # Save figure
        savefigs(fig, outDir, 'event_counts', format=plot_file)

    finally:
        plt.close(fig)


def load_data(root_file: Path) -> dict[str, np.ndarray]:
    """Load distributions from ROOT file produced by optimize_ll.py.

    Args:
        root_file: Path to ROOT file containing distributions

    Returns:
        Dictionary mapping variable names to numpy arrays
    """
    if not root_file.exists():
        print(f"Warning: ROOT file not found: {root_file}")
        return {}

    try:
        with uproot.open(root_file) as file:
            if 'distributions' not in file:
                print(f"Warning: 'distributions' not found in {root_file}")
                return {}

            tree = file['distributions']
            distributions = {}

            # Read all branches
            for key in tree.keys():
                distributions[key] = tree[key].array(library='np')

            return distributions
    except Exception as e:
        print(f"Error loading ROOT file {root_file}: {e}")
        return {}


def plot_comp(
        reco: np.ndarray,
        true: np.ndarray,
        var_name: str,
        xlabel: str,
        outDir: Path,
        label: str,
        bins: int = 100) -> None:
    """Plot comparison between reconstructed and true distributions.

    Args:
        reconstructed: Array of reconstructed values
        true: Array of true values
        var_name: Name of the variable
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histogram
    """
    # Create figure with subplots
    fig, ax = _make_fig()

    try:
        # Determine common range for both histograms
        xmin = min(np.floor(reco).min(), np.floor(true).min())
        xmax = max(np.ceil(reco).max(),  np.ceil(true).max())

        # Left plot: Overlay histograms
        ax.hist(reco, bins=bins, range=(xmin, xmax),
                alpha=0.6, label='Reconstructed', color='blue', density=True)
        ax.hist(true, bins=bins, range=(xmin, xmax),
                alpha=0.6, label='True', color='red', density=True)

        set_labels(ax, xlabel, 'Density', right=label)
        ax.set_xlim(xmin, xmax)
        ax.legend()

        # Save figure
        savefigs(fig, outDir, var_name, format=plot_file)

    finally:
        plt.close(fig)


def compare_dists(
        old_file: Path,
        optimal_file: Path,
        outDir: Path,
        label: str) -> None:
    """Compare distributions between old (chi2_frac=0.4) and optimal chi2_frac.

    Args:
        old_file: Path to results_old.root
        optimal_file: Path to results_optimal.root
        outDir: Output directory for plots
        label: LaTeX label for the final state
    """
    print("----->[Info] Loading distribution files")
    old_dists = load_data(old_file)
    opt_dists = load_data(optimal_file)

    if not old_dists or not opt_dists:
        print("Warning: Could not load distributions")
        return

    print("----->[Info] Generating distribution comparison plots")

    # Define variable pairs (reconstructed, true)
    comparisons = [
        # Leptons
        ('l1_e',     'lep1_e',     'l1 Energy'),
        ('l1_p',     'lep1_p',     'l1 Momentum'),
        ('l1_pt',    'lep1_pt',    'l1 Transverse Momentum'),
        ('l1_theta', 'lep1_theta', 'l1 Polar Angle'),

        ('l2_e',     'lep2_e',     'l2 Energy'),
        ('l2_p',     'lep2_p',     'l2 Momentum'),
        ('l2_pt',    'lep2_pt',    'l2 Transverse Momentum'),
        ('l2_theta', 'lep2_theta', 'l2 Polar Angle'),

        # Z system
        ('z_e',     'Z_e',     'Z Energy'),
        ('z_p',     'Z_p',     'Z Momentum'),
        ('z_pt',    'Z_pt',    'Z Transverse Momentum'),
        ('z_theta', 'Z_theta', 'Z Polar Angle'),

        # Pair variables
        ('mass',   'Mass',   'Z Mass'),
        ('recoil', 'Recoil', 'Recoil Mass'),
    ]

    # Create subdirectories for old and optimal
    old_outDir = outDir / 'old'
    opt_outDir = outDir / 'optimal'
    old_outDir.mkdir(parents=True, exist_ok=True)
    opt_outDir.mkdir(parents=True, exist_ok=True)

    # Generate comparison plots for each distribution
    for reco_var, true_var, display_name in comparisons:
        # Check if variables exist
        if reco_var not in old_dists or true_var not in old_dists:
            print(f"Warning: Variables {reco_var} or {true_var} not found in old distributions")
            continue

        if reco_var not in opt_dists or true_var not in opt_dists:
            print(f"Warning: Variables {reco_var} or {true_var} not found in optimal distributions")
            continue

        # Plot old distributions
        plot_comp(
            old_dists[reco_var],
            old_dists[true_var],
            reco_var,
            display_name,
            old_outDir,
            f'{label} - $\\chi^2_{{frac}}=0.40$'
        )

        # Plot optimal distributions
        plot_comp(
            opt_dists[reco_var],
            opt_dists[true_var],
            reco_var,
            display_name,
            opt_outDir,
            f'{label} - Optimal $\\chi^2_{{frac}}$'
        )



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    """Main plotting function"""
    cat, ecm = arg.cat, arg.ecm

    # Get input and output directories
    inDir  = loc.get('OPTIMISATION_RES',   cat, ecm, type=Path)
    outDir = loc.get('PLOTS_OPTIMISATION', cat, ecm, type=Path)

    if not inDir.exists():
        print(f"Error: Input directory not found: {inDir}")
        sys.exit(1)

    # Get processes to plot
    if arg.procs == '':
        procs = [p.name for p in inDir.iterdir() if p.is_dir()]
    else:
        procs = arg.procs.split('-')

    if not procs:
        print(f"Error: No processes found in {inDir}")
        sys.exit(1)

    # Set LaTeX label
    if cat == 'mumu':
        label = r'$Z(\mu^+\mu^-)H$'
    elif cat == 'ee':
        label = r'$Z(e^+e^-)H$'
    else:
        label = 'ZH'

    # Plot results for each process
    for proc in sorted(procs):
        proc_dir = inDir / proc
        results_file = proc_dir / 'results.json'
        out_dir = outDir / proc

        if not results_file.exists():
            print(f"Warning: Results file not found for {proc}: {results_file}")
            continue

        print(f"\n----->[Info] Processing {proc}")

        # Load results
        results = load_results(results_file)

        if not results:
            print(f"Warning: No results loaded for {proc}")
            continue

        # Extract arrays
        chi2_fracs, efficiencies, n_correct, n_partial, n_incorrect, n_total = extract_arrays(results)

        # Create output directory
        out_dir.mkdir(exist_ok=True, parents=True)

        # Generate optimization plots
        print(f"----->[Info] Generating optimization plots for {proc}")
        efficiency(chi2_fracs, efficiencies, out_dir, label)
        pairing_composition(
            chi2_fracs, n_correct, n_partial, n_incorrect, n_total,
            out_dir, label
        )
        event_counts(chi2_fracs, n_correct, n_partial, n_incorrect, out_dir, label)

        # Generate distribution comparison plots
        print(f"----->[Info] Generating distribution comparison plots for {proc}")
        dist_outDir  = out_dir / 'distributions'
        old_file     = proc_dir / 'results_old.root'
        optimal_file = proc_dir / 'results_optimal.root'

        if old_file.exists() and optimal_file.exists():
            compare_dists(old_file, optimal_file, dist_outDir, label)
        else:
            print(f"Warning: Distribution files not found for {proc}")
            if not old_file.exists():
                print(f"  Missing: {old_file}")
            if not optimal_file.exists():
                print(f"  Missing: {optimal_file}")




if __name__ == '__main__':
    try:
        main()
    finally:
        timer(t)
