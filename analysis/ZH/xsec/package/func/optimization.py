
import sys, json, uproot

import numpy as np
import awkward as ak

from glob import glob
from tqdm import tqdm
from pathlib import Path

from ..logger import get_logger

LOGGER = get_logger(__name__)



class Optimizer:
    """Optimize chi2 pairing weighting parameter by comparing with MC truth.

    Analyzes reconstructed particle pair assignments against true MC pairings
    to find optimal chi2 weighting values. Supports two chi2 methods:
    - mll: chi2 = dr + frac * (dm - dr) where dr = recoil distance, dm = mass distance
    - pll: chi2 = (1-frac) * (dr + 0.6*(dm-dr)) + frac * dp (adds momentum distance)

    The optimizer scans parameter space and reports efficiency metrics for
    different event categories (zero pairs, one pair, multiple pairs).
    """""

    ######################
    ### INITIALISATION ###
    ######################

    def __init__(
            self,
            proc: str,
            inDir: Path,
            outDir: Path,
            ecm: int = 240,
            nevents: int = -1,
            chi2_method: str = 'mll'):
        """Initialize optimizer with pure Python/uproot backend.

        Sets up file I/O, configures chi2 method, and loads event data.
        Applies mass cut to filter out pairs within 3 GeV of Higgs mass (125 GeV).

        Parameters:
        -----------
        proc : str
            Process name (e.g., 'wzp6_ee_eeH_ecm240')
        inDir : Path
            Input directory containing ROOT files
        outDir : Path
            Output directory for results and distributions
        ecm : int
            Center-of-mass energy in GeV (240 or 365)
        nevents : int
            Maximum number of events to process (-1 = all events)
        chi2_method : str
            'mll' for chi2_recoil (mass + recoil distance)
            'pll' for chi2_pll (mass + recoil + momentum distance)
        """

        inFiles = glob(str(inDir / proc / '*'))
        if not inFiles:
            LOGGER.warning(f'No file found in {inDir / proc}\nSearching for {inDir / (proc+".root")} file')
            inFile: Path = inDir / (proc+'.root')
            if  inFile.exists():
                inFiles = [str(inFile)]
            else:
                LOGGER.error(f'No file found in {inDir / (proc+".root")}')
                sys.exit(1)

        self.inFiles = inFiles
        self.outDir = Path(outDir) / proc / chi2_method
        self.outDir.mkdir(parents=True, exist_ok=True)
        self.ecm = ecm
        self.chi2_method = chi2_method

        LOGGER.info(f'Loading data for process {proc} ({chi2_method = })')
        self.load_data(nevents)
        self.results = {}

    ####################
    ### DATA LOADING ###
    ####################

    def load_data(self, nevents=-1):
        """Load and filter event data from ROOT files.

        Efficiently accumulates data from multiple chunk files, applies mass cut
        (|mass - 125 GeV| > 3 GeV), and computes statistics on pair multiplicities.
        Uses awkward arrays for jagged data and maintains proper indexing across
        all events.

        Args:
            nevents: Maximum events to load (-1 for all)
        """
        # Accumulate data per branch
        data: dict[str, list] = {}

        self.total_events = 0
        event_count = 0

        for fpath in self.inFiles:
            with uproot.open(fpath) as file:
                tree = file['events']

                # Get total event count
                self.total_events += file['eventsSelected'].value

                # Read all branches
                branch_names = [name for name in tree.keys() if '/' not in name]
                arrays = tree.arrays(library='ak')

                # Determine slice length
                n_entries = len(arrays[branch_names[0]])
                if nevents > 0:
                    n_entries = min(n_entries, nevents - event_count)

                # Slice arrays
                for name in branch_names:
                    arrays[name] = arrays[name][:n_entries]

                # Determine which branches are jagged
                jagged_branches = [name for name in branch_names if not isinstance(ak.num(arrays[name], axis=-1), int)]

                # Apply mass cut: remove pairs within 3 GeV of Higgs mass (125 GeV)
                mass_cut = (np.abs(arrays['mass'] - 125) > 3)

                # Apply mass cut to jagged arrays
                filtered_arrays = {}
                for name in jagged_branches:
                    filtered_arrays[name] = arrays[name][mass_cut]

                # Accumulate filtered data
                for name in branch_names:
                    if name not in data:
                        data[name] = []
                    if name in jagged_branches:
                        data[name].append(filtered_arrays[name])
                    else:
                        data[name].append(arrays[name])

                event_count += n_entries
                if nevents > 0 and event_count >= nevents:
                    break

        # Concatenate all data and assign them as attributes
        for name in data.keys():
            setattr(self, name, ak.concatenate(data[name]))

        # Recalculate n_pair after filtering (number of pairs per event)
        self.n_pair = ak.num(self.mass)

        # Track actual number of events loaded (not metadata from file)
        self.n_events = len(self.mass)

        # Count events by number of pairs
        n_pair = np.asarray(self.n_pair)
        self.n_zero_pair  = int(np.sum(n_pair == 0))
        self.n_one_pair   = int(np.sum(n_pair == 1))
        self.n_multi_pair = int(np.sum(n_pair >  1))

        LOGGER.info(f'Loaded {self.n_events:,} events (total events in files: {self.total_events:,})')
        LOGGER.info(f'Event breakdown: {self.n_zero_pair:<10,} events with  0 pair\n'
                    f'                 {self.n_one_pair:<10,} events with  1 pair\n'
                    f'                 {self.n_multi_pair:<10,} events with >1 pair')

    ##############################
    ### OPTIMIZATION UTILITIES ###
    ##############################

    def best_pair_idx(
            self,
            frac: float,
            dm: ak.Array,
            drec: ak.Array,
            dp: ak.Array | None = None
             ) -> ak.Array:
        """
        Find best pair index for each event based on chi2 distance metric.

        Computes chi2 distance for each pair in each event and selects the pair
        with minimum chi2. For events with no pairs, returns None (handled by caller).

        Chi2 formulas:
        - mll method: chi2 = drec + frac * (dm - drec)
        - pll method: chi2 = (1 - frac) * (drec + 0.6 * (dm - drec)) + frac * dp

        Args:
            frac: Chi2 weighting parameter (0 to 1)
            dm: Squared distance to Z mass (mass - mZ)^2 for each pair
            drec: Squared distance to Higgs recoil mass (recoil - mH)^2 for each pair
            dp: Squared distance to Z momentum pll target, only for pll method

        Returns:
            Awkward array of best pair indices per event
        """
        if self.chi2_method == 'mll':
            chi2 = drec + frac * (dm - drec)
        elif self.chi2_method == 'pll':
            chi2 = (1 - frac) * (drec + 0.6 * (dm - drec)) + frac * dp
        else:
            raise ValueError(f"Unknown chi2_method: {self.chi2_method}")

        return ak.argmin(chi2, axis=1)


    def test_chi2(
            self,
            frac: float,
            dm: ak.Array,
            drec: ak.Array,
            dp: ak.Array | None = None
             ) -> dict[str, float | int | dict[str, float | int]]:
        """Test chi2 weighting parameter and compute pairing efficiency.

        Determines best pairs for each event using chi2 metric, compares with MC truth,
        and computes match statistics. Evaluates efficiency separately for events with
        zero, one, or multiple valid pairs to understand performance across categories.

        Args:
            frac: Chi2 weighting parameter to test
            dm: Precomputed squared Z mass distances for all pairs
            drec: Precomputed squared recoil distances for all pairs
            dp: Precomputed squared momentum distances (for pll method only)

        Returns:
            Dictionary containing:
            - 'frac': Input parameter value
            - 'overall': Statistics across all events
            - 'zero_pair', 'one_pair', 'multi_pair': Statistics by event category
            Each category contains: efficiency, n_correct, n_partial, n_incorrect, n_total
        """

        # Get best pair indices using precomputed distances
        best_idx = self.best_pair_idx(frac, dm, drec, dp)

        # Handle events with 0 pairs: mask them out
        n_pairs = ak.num(self.leading_mc)
        has_pairs = n_pairs > 0

        # Extract best pairs using vectorized awkward masking
        local_idx = ak.local_index(self.leading_mc, axis=1)
        mask_mc1 = local_idx == best_idx[:, None]
        mask_mc2 = local_idx == best_idx[:, None]
        reco_mc1 = np.asarray(ak.flatten(self.leading_mc[mask_mc1]))
        reco_mc2 = np.asarray(ak.flatten(self.subleading_mc[mask_mc2]))

        # Get true MC indices and pair counts
        true_mc1 = np.asarray(self.Leading_MC)
        true_mc2 = np.asarray(self.Subleading_MC)
        n_pair_array = np.asarray(self.n_pair)

        # Initialize matches array with 0 for all events
        matches = np.zeros(self.n_events, dtype=int)

        # Only compute matches for events that have pairs after filtering
        if len(reco_mc1) > 0:
            match1: np.ndarray = (reco_mc1 == true_mc1[has_pairs]) | (reco_mc1 == true_mc2[has_pairs])
            match2: np.ndarray = (reco_mc2 == true_mc1[has_pairs]) | (reco_mc2 == true_mc2[has_pairs])
            matches[has_pairs] = match1.astype(int) + match2.astype(int)

        # Create masks for different pair categories
        mask_zero  = n_pair_array == 0
        mask_one   = n_pair_array == 1
        mask_multi = n_pair_array >  1

        # Helper function to compute metrics for a category
        def compute_category_stats(
                matches: np.ndarray,
                mask: np.ndarray
                 ) -> dict[str, float | int]:

            cat_matches = matches[mask]
            n_pairs = n_pair_array[mask]
            n_correct   = int(np.sum(cat_matches == 2))
            n_no_pairs  = int(np.sum(n_pairs == 0))
            n_total     = int(np.sum(mask))
            return {
                'n_correct':   n_correct,
                'n_partial':   int(np.sum(cat_matches == 1)),
                'n_incorrect': int(np.sum((cat_matches == 0) & (n_pairs > 0))),
                'n_no_pairs':  n_no_pairs,
                'n_total':     n_total,
                'efficiency':  n_correct / max(n_total - n_no_pairs, 1),
            }

        # Compute stats for each category
        result = {
            'frac': frac,
            'overall': {
                'efficiency':  int(np.sum(matches == 2)) / max(self.n_events, 1),
                'n_correct':   int(np.sum(matches == 2)),
                'n_partial':   int(np.sum(matches == 1)),
                'n_incorrect': int(np.sum(matches == 0)),
                'n_total':     self.n_events,
            },
            'zero_pair':  compute_category_stats(matches, mask_zero),
            'one_pair':   compute_category_stats(matches, mask_one),
            'multi_pair': compute_category_stats(matches, mask_multi),
        }
        return result

    #############################
    ### OPTIMIZATION FUNCTION ###
    #############################

    def optimize(
            self,
            chi2_values: list[float] | np.ndarray,
            mass: int | float = 91.2,
            recoil: int | float = 125,
            precision: int = 2):
        """Scan chi2 weighting parameter space and compute efficiency for each value.

        Precomputes all distance metrics once, then efficiently scans parameter space
        by testing each chi2 value and storing results. This vectorized approach
        avoids redundant computations.

        Args:
            chi2_values: Array of chi2_frac values to scan (typically 0 to 1)
            mass: Z boson mass for distance calculation
            recoil: Higgs mass for recoil distance calculation
            precision: Decimal places for rounding results

        Returns:
            Dictionary of results keyed by rounded chi2_frac values
        """

        # Precompute squared distances once to avoid recalculation in each iteration
        dm   = (self.mass   - mass) ** 2
        drec = (self.recoil - recoil) ** 2

        # For pll method, also precompute momentum distance
        if self.chi2_method == 'pll':
            pll = 51 if self.ecm == 240 else (146 if self.ecm == 365 else -1)
            dp = (self.zll_p - pll) ** 2
        else:
            dp = None

        for chi2_frac in tqdm(chi2_values):
            result = self.test_chi2(chi2_frac, dm, drec, dp)
            chi2_frac_rounded = round(chi2_frac, precision)
            self.results[chi2_frac_rounded] = result
        return self.results

    #############################
    ### DATA SAVING FUNCTIONS ###
    #############################

    def extract_distributions(
            self,
            chi2_frac: float,
            mZ: float | int = 91.2,
            mH: float | int = 125) -> dict[str, np.ndarray]:
        """Extract kinematic variables for best pairings at given chi2 parameter.

        Selects best pair for each event using chi2 metric and extracts full
        kinematic information (momentum, angles) for both leptons and Z system,
        plus match quality indicators.

        Args:
            chi2_frac: Chi2 weighting parameter value
            mZ: Z boson mass (GeV) - used for distance metric
            mH: Higgs mass (GeV) - used for recoil distance metric

        Returns:
            Dictionary mapping variable names to numpy arrays:
            - Reconstructed leptons (p, pt, theta) for leading/subleading
            - True leptons (for events with surviving pairs)
            - Z system kinematics (p, pt, theta)
            - Pair variables (mass, recoil) both reco and true
            - 'matches': Integer array indicating pairing quality (0=none, 1=partial, 2=both correct)
        """
        # Precompute distances
        dm   = (self.mass   - mZ) ** 2
        drec = (self.recoil - mH) ** 2

        # For pll method, also compute momentum distance
        if self.chi2_method == 'pll':
            pll = 51 if self.ecm == 240 else (146 if self.ecm == 365 else -1)
            dp = (self.zll_p - pll) ** 2
        else:
            dp = None

        # Get best pair indices
        best_idx = self.best_pair_idx(chi2_frac, dm, drec, dp)

        # Fill None values (from events with 0 pairs) with 0 as placeholder
        best_idx = ak.fill_none(best_idx, 0)

        # Identify events with at least 1 pair after filtering
        n_pairs = ak.num(self.mass)
        has_pairs_mask = np.asarray(n_pairs > 0)

        # Compute matches for best pairings (same logic as in test_chi2)
        local_idx = ak.local_index(self.leading_mc, axis=1)
        mask_mc1 = local_idx == best_idx[:, None]
        mask_mc2 = local_idx == best_idx[:, None]
        reco_mc1 = np.asarray(ak.flatten(self.leading_mc[mask_mc1]))
        reco_mc2 = np.asarray(ak.flatten(self.subleading_mc[mask_mc2]))

        # Get true MC indices
        true_mc1 = np.asarray(self.Leading_MC)
        true_mc2 = np.asarray(self.Subleading_MC)

        # Initialize matches array
        matches = np.zeros(self.n_events, dtype=int)

        # Only compute matches for events that have pairs after filtering
        if len(reco_mc1) > 0:
            match1: np.ndarray = (reco_mc1 == true_mc1[has_pairs_mask]) | (reco_mc1 == true_mc2[has_pairs_mask])
            match2: np.ndarray = (reco_mc2 == true_mc1[has_pairs_mask]) | (reco_mc2 == true_mc2[has_pairs_mask])
            matches[has_pairs_mask] = match1.astype(int) + match2.astype(int)

        # Helper function to select values based on best pairing index for each event
        def select_by_idx(array, indices):
            """Select elements from jagged array using indices for each event (vectorized)"""
            if not isinstance(array, ak.Array):
                array = ak.Array(array)

            indices   = np.asarray(indices, dtype=int)
            local_idx = ak.local_index(array, axis=1)
            mask = local_idx == indices[:, None]
            selected = ak.flatten(array[mask])
            return np.asarray(selected)

        return {
            # Reconstructed leptons (selected based on best pairing)
            'leading_p':     select_by_idx(self.leading_p,     best_idx),
            'leading_pt':    select_by_idx(self.leading_pt,    best_idx),
            'leading_theta': select_by_idx(self.leading_theta, best_idx),

            'subleading_p':     select_by_idx(self.subleading_p,     best_idx),
            'subleading_pt':    select_by_idx(self.subleading_pt,    best_idx),
            'subleading_theta': select_by_idx(self.subleading_theta, best_idx),

            # True leptons (only for events with surviving pairs)
            'Leading_p':     np.asarray(self.Leading_p)[has_pairs_mask],
            'Leading_pt':    np.asarray(self.Leading_pt)[has_pairs_mask],
            'Leading_theta': np.asarray(self.Leading_theta)[has_pairs_mask],

            'Subleading_p':     np.asarray(self.Subleading_p)[has_pairs_mask],
            'Subleading_pt':    np.asarray(self.Subleading_pt)[has_pairs_mask],
            'Subleading_theta': np.asarray(self.Subleading_theta)[has_pairs_mask],

            # Reconstructed Z system (selected based on best pairing)
            'zll_p':     select_by_idx(self.zll_p,     best_idx),
            'zll_pt':    select_by_idx(self.zll_pt,    best_idx),
            'zll_theta': select_by_idx(self.zll_theta, best_idx),

            # True Z system (only for events with surviving pairs)
            'Zll_p':     np.asarray(self.Zll_p)[has_pairs_mask],
            'Zll_pt':    np.asarray(self.Zll_pt)[has_pairs_mask],
            'Zll_theta': np.asarray(self.Zll_theta)[has_pairs_mask],

            # Pair variables (selected based on best pairing)
            'mass':   select_by_idx(self.mass,   best_idx),
            'recoil': select_by_idx(self.recoil, best_idx),
            'Mass':   np.asarray(self.Mass)[has_pairs_mask],
            'Recoil': np.asarray(self.Recoil)[has_pairs_mask],

            # Match information (0=no match, 1=partial, 2=full match) - filtered to events with pairs
            'matches': matches[has_pairs_mask]
        }

    def save_distributions(self, chi2_frac: float, filename: str) -> None:
        """Save kinematic distributions to ROOT file for given chi2 parameter.

        Extracts all kinematic variables and match information for best pairings
        at the specified chi2_frac value, then saves as ROOT TTree for further analysis.

        Args:
            chi2_frac: Chi2 weighting parameter value
            filename: Output filename (e.g., 'results_baseline.root')
        """
        # Extract distributions
        distributions = self.extract_distributions(chi2_frac)

        # Create output file path
        outFile = self.outDir / filename
        outFile.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for ROOT file (convert all to numpy arrays)
        output_data: dict[str, np.ndarray] = {}
        for key, values in distributions.items():
            if isinstance(values, ak.Array):
                values = np.asarray(values)
            # Ensure proper dtype (float32 or float64)
            if not np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32)
            # Flatten to 1D if needed
            if values.ndim > 1:
                values = values.flatten()
            output_data[key] = values

        # Write to ROOT file as structured array / TTree
        with uproot.recreate(str(outFile)) as file:
            # Create a structured numpy array from the dictionary
            dtype = [(key, values.dtype) for key, values in output_data.items()]
            # Find the common length (should be the same for all arrays)
            n_events = len(next(iter(output_data.values())))

            # Create structured array
            structured_data = np.zeros(n_events, dtype=dtype)
            for key, values in output_data.items():
                structured_data[key] = values

            # Write as ROOT tree
            file["distributions"] = structured_data

        LOGGER.info(f'Variables distribution saved at {outFile}')


    def save_results(self):
        """Save optimization scan results to JSON file.

        Exports efficiency metrics for all tested chi2 values, organized by
        event category (overall, zero_pair, one_pair, multi_pair) for easy
        comparison and plotting.
        """
        outFile = self.outDir / 'results.json'

        json_results = {}
        for chi2_frac, result in self.results.items():
            res = {pair: {
                **{k: result[pair][k] for k in ['efficiency', 'n_correct', 'n_partial', 'n_incorrect', 'n_total']},
                **({'n_no_pairs': result[pair]['n_no_pairs']} if pair != 'overall' else {})
            } for pair in ['overall', 'zero_pair', 'one_pair', 'multi_pair']}
            json_results[f"{chi2_frac:.2f}"] = {'chi2_frac': result['frac']} | res

        outFile.write_text(json.dumps(json_results, indent=4))
        LOGGER.info(f'Results saved at {outFile}\n')
        return outFile
