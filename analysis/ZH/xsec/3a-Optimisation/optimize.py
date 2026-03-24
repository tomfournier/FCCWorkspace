#!/usr/bin/env python3
"""
Example: Optimizing chi2_recoil_frac for true Z pairing in e+e- -> ZH analysis

This script demonstrates how to use the new MC truth functions to optimize
the chi2_recoil_frac parameter in resonanceBuilder_mass_recoil.

Usage:
    python3 optimize_chi2_recoil_frac.py <input_file> <output_dir>
"""

import sys
import os
sys.path.insert(0, '/afs/cern.ch/user/t/tofourni/eos/FCC/FCCWorkspace/install/python')

import ROOT
from ROOT import gROOT
import json
from pathlib import Path

# Enable C++ code loading
gROOT.ProcessLine('#include "../../../../functions/functions.h"')


class ZPairingOptimizer:
    """Optimize chi2_recoil_frac by comparing with MC truth Z pairing"""

    def __init__(self, input_file, output_dir, ecm=365):
        """
        Initialize optimizer

        Args:
            input_file: Path to FCCAnalyses ROOT file with MC info
            output_dir: Directory for output plots and results
            ecm: Center-of-mass energy (GeV)
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ecm = ecm

        # Load data
        self.df = ROOT.RDataFrame("events", input_file)
        self.total_events = self.df.Count().GetValue()
        print(f"Total events: {self.total_events}")

        self.results = {}

    def setup_mc_selections(self):
        """Apply basic selections on MC particles"""
        df = self.df

        # Select leptons (e+e-, mu+mu-, tau+tau-) from reconstruction
        # Status 1 = stable final state
        df = df.Define("mc_leptons",
                       "MCParticle::sel_pdgID(abs(MCParticles_PDG)==11 or abs(MCParticles_PDG)==13, true)(MCParticles)")

        df = df.Define("true_z_info",
                       "getTrueZ_from_H_decay(MCParticles, MCParticles_parents, MCParticles_daughters)")

        df = df.Filter("true_z_info[0].mass > 0",
                       "Valid true Z from non-Higgs decay")

        return df

    def test_chi2_value(self, chi2_frac, resonance_mass=91.2, recoil_mass=125.):
        """
        Test a specific chi2_recoil_frac value

        Args:
            chi2_frac: Value to test (0.0 to 1.0)
            resonance_mass: Z boson mass (GeV)
            recoil_mass: Higgs mass (GeV) [not used for filter here]

        Returns:
            Dictionary with efficiency and stats
        """
        # chi2_str = f"{chi2_frac:.2f}".replace(".", "p")

        # This is a placeholder - in actual analysis you would:
        # 1. Get reconstructed leptons
        # 2. Build Z with resonanceBuilder_mass_recoil(chi2_frac)
        # 3. Compare with true_z_info
        # 4. Count matches

        # Example structure:
        result = {
            'chi2_frac': chi2_frac,
            'efficiency': 0.0,  # Will be calculated
            'n_correct': 0,
            'n_total': self.total_events,
            'mass_reco': [],  # For plotting
            'mass_true': [],
        }

        return result

    def optimize(self, chi2_values=None):
        """
        Run optimization loop over chi2_recoil_frac values

        Args:
            chi2_values: List of values to test (default: 0.0 to 1.0 in steps of 0.1)
        """
        if chi2_values is None:
            chi2_values = [i * 0.1 for i in range(11)]  # 0.0 to 1.0

        print(f"\nOptimizing chi2_recoil_frac over values: {chi2_values}")

        for chi2_frac in chi2_values:
            print(f"  Testing chi2_recoil_frac = {chi2_frac:.2f}...", end='', flush=True)
            result = self.test_chi2_value(chi2_frac)
            self.results[chi2_frac] = result
            print(" Done")

        return self.results

    def analyze_results(self):
        """Analyze and display optimization results"""
        if not self.results:
            print("No results to analyze!")
            return

        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"{'chi2_recoil_frac':<20} {'Efficiency':<15} {'N_Correct':<15}")
        print("-"*60)

        best_chi2 = None
        best_eff = 0.0

        for chi2_frac in sorted(self.results.keys()):
            result = self.results[chi2_frac]
            eff = result['efficiency']
            print(f"{chi2_frac:<20.2f} {eff:<15.3f} {result['n_correct']:<15d}")

            if eff > best_eff:
                best_eff = eff
                best_chi2 = chi2_frac

        print("-"*60)
        print(f"{'OPTIMAL':<20} {best_eff:<15.3f}")
        print("="*60)

        return best_chi2, best_eff

    def plot_results(self):
        """Create visualization of optimization results"""
        import matplotlib.pyplot as plt
        import numpy as np

        chi2_values = sorted(self.results.keys())
        efficiencies = [self.results[c]['efficiency'] for c in chi2_values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Efficiency vs chi2_recoil_frac
        ax1.plot(chi2_values, efficiencies, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax1.set_xlabel('chi2_recoil_frac', fontsize=12)
        ax1.set_ylabel('Pairing Efficiency', fontsize=12)
        ax1.set_title('Z Pairing Efficiency vs chi2_recoil_frac', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # Highlight optimal value
        optimal_idx = np.argmax(efficiencies)
        ax1.plot(chi2_values[optimal_idx], efficiencies[optimal_idx], 'r*',
                 markersize=20, label=f'Optimal: {chi2_values[optimal_idx]:.2f}')
        ax1.legend(fontsize=11)

        # Plot 2: Number of correct pairings
        n_correct = [self.results[c]['n_correct'] for c in chi2_values]
        ax2.bar(chi2_values, n_correct, width=0.08, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('chi2_recoil_frac', fontsize=12)
        ax2.set_ylabel('Number of Correct Pairings', fontsize=12)
        ax2.set_title('Correct Pairings vs chi2_recoil_frac', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / 'chi2_optimization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {output_path}")
        plt.close()

    def save_results(self):
        """Save results to JSON"""
        output_path = self.output_dir / 'optimization_results.json'

        # Convert to JSON-serializable format
        json_results = {}
        for chi2_frac, result in self.results.items():
            json_results[f"{chi2_frac:.2f}"] = {
                'chi2_frac': result['chi2_frac'],
                'efficiency': result['efficiency'],
                'n_correct': result['n_correct'],
                'n_total': result['n_total'],
            }

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved: {output_path}")

        return output_path


def main():
    """Main analysis function"""

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Create optimizer
    optimizer = ZPairingOptimizer(input_file, output_dir)

    # Run optimization
    print("\n" + "="*60)
    print("Z PAIRING OPTIMIZATION FOR e+e- -> ZH")
    print("="*60)

    optimizer.setup_mc_selections()
    optimizer.optimize(chi2_values=[i * 0.05 for i in range(21)])  # 0.0 to 1.0 in 0.05 steps

    # Analyze results
    best_chi2, best_eff = optimizer.analyze_results()

    # Save and plot
    optimizer.plot_results()
    optimizer.save_results()

    print("\n" + "="*60)
    print(f"Recommended chi2_recoil_frac: {best_chi2:.2f}")
    print(f"Expected efficiency: {best_eff:.1%}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
