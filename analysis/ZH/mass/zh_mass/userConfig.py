'''Core configuration for path templates and helpers.

Provides:
- Path templates via `loc` with placeholders: `cat`, `ecm`, `sel`.
- Type-flexible expansion with `LocPath.get()` and `loc.get(...)`.
- Bidirectional type conversion via `astype(str)` and `astype(Path)`.
- Global parameters: `plot_file`, `frac`, `nb`, `ww`, `cat`, `ecm`, `lumi`.
- Utilities: `get_loc()`, `event()`, `get_params()`.

Conventions:
- `lumi` is in ab^-1 (10.8 at 240 GeV; 3.12 at 365 GeV).
- Templates are expanded with `loc.get()` or `LocPath.get()`.
- Expanded paths can be str or pathlib.Path; both support `astype()`.

Usage:
- path = loc.EVENTS
- path1 = loc.EVENTS.get(cat='ee', ecm=240, sel='Baseline')
- path2 = loc.get('EVENTS', cat='ee', ecm=240, sel='Baseline', type=Path)
- path3 = path1.astype(str)
- path4 = path3.astype(Path)
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from path import loc, LocPath, PathObj
from logger import get_logger

LOGGER = get_logger(__name__)



#################################
### LABEL FOR SIGNAL FUNCTION ###
#################################

param_label = {
    'mean':     '#mu [GeV]',
    'mean_gt':  '#mu_{gt} [GeV]',
    'mean1':    '#mu offset [GeV]',
    'mean_gt1': '#mu_{gt} offset [GeV]',
    'yield':    'Events',
    'sigma':    '#sigma [GeV]',
    'sigma_gt': '#sigma_{gt} [GeV]',
    'alpha_1':  '#alpha_1',
    'alpha_2':  '#alpha_2',
}



#############################
##### LOCATION OF FILES #####
#############################

repo = str(PathObj(__file__).parent.parent.resolve())

# Templates as LocPath strings with placeholders
loc.ROOT        = LocPath(repo)                    # Repo root
loc.PACKAGE     = LocPath(f"{repo}/package")       # Python package code
loc.OUT         = LocPath(f"{repo}/output")        # Output root
loc.PLOTS       = LocPath(f"{repo}/output/plots")  # Plots root
loc.DATA        = LocPath(f"{repo}/output/data")   # Data artifacts
loc.TMP         = LocPath(f"{repo}/output/tmp")    # For temporary files

loc.EVENTS_DIR  = LocPath(f"{repo}/output/data/1-events")      # 1. Event samples
loc.HIST_DIR    = LocPath(f"{repo}/output/data/2-histograms")  # 2. Histograms root directory
loc.FIT_DIR     = LocPath(f"{repo}/output/data/3-combine")     # 3. Statistical fits

loc.EVENTS      = LocPath(f"{repo}/output/data/events/ecm/cat/full")  # Analysis samples
loc.EVENTS_TEST = LocPath(f"{repo}/output/data/events/ecm/cat/test")  # Test samples for analysis

loc.HIST_RAW    = LocPath(f"{repo}/output/data/histograms/1-raw-hists/ecm/cat")      # Raw histograms
loc.HIST_PROC   = LocPath(f"{repo}/output/data/histograms/2-processed/ecm/cat/sel")  # Processed histograms

loc.PLOTS_SEL   = LocPath(f"{repo}/output/plots/1-selection/ecm/cat")  # Analysis plots
loc.PLOTS_FIT   = LocPath(f"{repo}/output/plots/2-fit")                # Fit plots

loc.FIT         = LocPath(f"{repo}/output/data/combine/sel/ecm/cat")           # Combine root
loc.FIT_LOG     = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/log")       # Logs
loc.FIT_RESULT  = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/results")   # Results
loc.FIT_CARD    = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/datacard")  # Datacards
loc.FIT_WS      = LocPath(f"{repo}/output/data/combine/sel/ecm/cat/WS")        # Workspaces



#########################
### FIT CONFIGURATION ###
#########################

PARAMETER_RANGES = {
    'mean_cb1': (-2, 2), 'sigma_cb': (0, 1), 'mean_gt1': (-2, 2), 'sigma_gt': (0, 2),
    'alpha_1': (-10, 0), 'n_1': (-10, 10), 'cb_1': (0, 1),
    'alpha_2': (0, +10), 'n_2': (-10, 10), 'cb_2': (0, 1),
}

def with_ranges(values: dict[str, float | int], overrides: dict[str, tuple[float | int, float | int]] | None = None) -> dict[str, list]:
    ranges = dict(PARAMETER_RANGES)
    if overrides: ranges.update(overrides)
    return {name: [value, *ranges[name]] for name, value in values.items()}


param_config = {
    240: {
        'ee': {
            0: with_ranges({
                'mean_cb1': 0.13, 'sigma_cb': 0.46, 'mean_gt1': 0.55, 'sigma_gt': 0.83,
                'alpha_1': -0.17, 'n_1': 3.38, 'cb_1': 0.56,
                'alpha_2': +3.90, 'n_2': 0.10, 'cb_2': 0.37,
            }),
            1: with_ranges({
                'mean_cb1': 0.11, 'sigma_cb': 0.43, 'mean_gt1': 0.59, 'sigma_gt': 0.69,
                'alpha_1': -1.79, 'n_1': 2.90, 'cb_1': 0.57,
                'alpha_2': +3.74, 'n_2': 0.33, 'cb_2': 0.37,
            }),
            2: with_ranges({
                'mean_cb1': 0.09, 'sigma_cb': 0.39, 'mean_gt1': 0.50, 'sigma_gt': 0.74,
                'alpha_1': -0.14, 'n_1': 4.66, 'cb_1': 0.54,
                'alpha_2': +4.07, 'n_2': 0.003, 'cb_2': 0.38,
            }),
            3: with_ranges({
                'mean_cb1': 0.15, 'sigma_cb': 0.52, 'mean_gt1': 0.71, 'sigma_gt': 1.01,
                'alpha_1': -0.19, 'n_1': 2.48, 'cb_1': 0.57,
                'alpha_2': +4.13, 'n_2': 0.0001, 'cb_2': 0.37,
            }),
        },
        'mumu': {
            0: with_ranges({
                'mean_cb1': 0.09, 'sigma_cb': 0.43, 'mean_gt1': 0.09, 'sigma_gt': 0.82,
                'alpha_1': -0.20, 'n_1': 2.55, 'cb_1': 0.49,
                'alpha_2': +3.13, 'n_2': 1.39, 'cb_2': 0.41,
            }),
            1: with_ranges({
                'mean_cb1': 0.09, 'sigma_cb': 0.45, 'mean_gt1': 0.37, 'sigma_gt': 0.86,
                'alpha_1': -1.73, 'n_1': 4.02, 'cb_1': 0.45,
                'alpha_2': +3.37, 'n_2': 1.10, 'cb_2': 0.43,
            }),
            2: with_ranges({
                'mean_cb1': 0.09, 'sigma_cb': 0.42, 'mean_gt1': 0.66, 'sigma_gt': 0.67,
                'alpha_1': -0.22, 'n_1': 2.24, 'cb_1': 0.50,
                'alpha_2': +3.96, 'n_2': 0.26, 'cb_2': 0.26,
            }),
            3: with_ranges({
                'mean_cb1': 0.09, 'sigma_cb': 0.42, 'mean_gt1': 0.66, 'sigma_gt': 0.67,
                'alpha_1': -0.22, 'n_1': 2.24, 'cb_1': 0.60,
                'alpha_2': +3.96, 'n_2': 0.26, 'cb_2': 0.26,
            }),
        },
    },
    365: {
        'ee': {
            0: with_ranges({
                'mean_cb1': 0.30, 'sigma_cb': 1.50, 'mean_gt1': 0.47, 'sigma_gt': 1.60,
                'alpha_1': -0.60, 'n_1': 1.32, 'cb_1': 0.64,
                'alpha_2': +1.24, 'n_2': 10.0, 'cb_2': 0.24,
            }),
            1: with_ranges({
                'mean_cb1': 0.30, 'sigma_cb': 0.70, 'mean_gt1': 1.55, 'sigma_gt': 1.40,
                'alpha_1': -0.70, 'n_1': 1.30, 'cb_1': 0.70,
                'alpha_2': +1.50, 'n_2': 45.0, 'cb_2': 0.28,
            }),
            2: with_ranges({
                'mean_cb1': 0.40, 'sigma_cb': 1.60, 'mean_gt1': 0.70, 'sigma_gt': 2.80,
                'alpha_1': -0.40, 'n_1': 1.40, 'cb_1': 0.70,
                'alpha_2': +1.32, 'n_2': 30.0, 'cb_2': 0.28,
            }),
            3: with_ranges({
                'mean_cb1': 0.40, 'sigma_cb': 1.60, 'mean_gt1': 0.70, 'sigma_gt': 2.80,
                'alpha_1': -0.40, 'n_1': 1.40, 'cb_1': 0.52,
                'alpha_2': +1.32, 'n_2': 30.0, 'cb_2': 0.36,
            }),
        },
        'mumu': {
            0: with_ranges({
                'mean_cb1': 0.30, 'sigma_cb': 1.50, 'mean_gt1': 0.47, 'sigma_gt': 1.60,
                'alpha_1': -0.60, 'n_1': 1.32, 'cb_1': 0.50,
                'alpha_2': +1.24, 'n_2': 10.0, 'cb_2': 0.25,
            }),
            1: with_ranges({
                'mean_cb1': 0.40, 'sigma_cb': 1.40, 'mean_gt1': 0.70, 'sigma_gt': 2.00,
                'alpha_1': -0.70, 'n_1': 1.20, 'cb_1': 0.70,
                'alpha_2': +3.80, 'n_2': 15.0, 'cb_2': 0.28,
            }),
            3: with_ranges({
                'mean_cb1': 0.40, 'sigma_cb': 1.60, 'mean_gt1': 0.70, 'sigma_gt': 2.80,
                'alpha_1': -0.40, 'n_1': 1.40, 'cb_1': 0.70,
                'alpha_2': +1.32, 'n_2': 30.0, 'cb_2': 0.28,
            }),
        }
    }
}
