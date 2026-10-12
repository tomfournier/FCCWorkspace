'''Cutflow and selection efficiency analysis with visualization.

Provides:
- Cut expression parsing and variable extraction: `branches_from_cuts()`.
- Event count aggregation across cut steps: `CutFlow()`.
- Decay-mode efficiency analysis: `CutFlowDecays()`, `Efficiency()`.
- ASCII table generation: `write_table()`.
- Integration with ROOT graphics backend for publication plots.

Functions:
- `branches_from_cuts()`: Extract variable names referenced in cut filter expressions using regex.
- `write_table()`: Format and write aligned ASCII tables with configurable column widths.
- `CutFlow()`: Render stacked cutflow histogram with signal overlay, significance values, and yields table.
- `CutFlowDecays()`: Plot efficiency curves (normalized to first cut) for each Higgs decay mode.
- `Efficiency()`: Generate efficiency summary plots with pull diagrams and per-decay tables.

Conventions:
- Cut steps indexed sequentially (cut0, cut1, ...) matching provided labels.
- Yields computed with luminosity and cross-section scaling when available.
- Uncertainties treated as Poisson (√N) on raw counts, scaled linearly.
- Significance computed per cut as S/√(S+B) where S=signal, B=total background.
- Efficiency normalized to first cut (cut0) for decay mode comparisons (%).
- Channel-specific plots (ee/mumu) optionally combined into totals (tot).
- ROOT histograms stored in flow dict with keys: flow[process]['hist'] = [TH1, ...].

Usage:
- Analyze event selection efficiency by computing cumulative event counts per cut step.
- Generate publication-ready cutflow plots with stacked backgrounds and signal overlay.
- Validate selection consistency across Higgs decay modes via efficiency tables and plots.
- Export yield summaries and efficiency pulls as both plots and tabular data.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from tqdm import tqdm
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from tools.utils import get_df, mkdir
from tools.process import getMetaInfo
from constants import h_colors, h_labels
from logger import get_logger

LOGGER = get_logger(__name__)



######################
### MAIN FUNCTIONS ###
######################

# ___________________________________
def branches_from_cuts(
    cuts: dict[str, dict[str, str]],
    variables: list
     ) -> list:
    '''Extract variables used in cut expressions.

    Scans all cut expressions and identifies which variables from the provided
    list are actually referenced. Results are sorted for deterministic output.

    Args:
        cuts (dict[str, dict[str, str]]): Dictionary mapping selection names to cut definitions (cut_name -> expression).
        variables (list): List of variable names to search for.

    Returns:
        list: Sorted list of variables found in cut expressions.
    '''
    import re
    used = set()
    for sel_cuts in cuts.values():
        for expr in sel_cuts.values():
            if not expr:  # Skip empty expressions
                continue
            for var in variables:
                # Match whole words only using regex word boundaries
                if re.search(r'\b' + re.escape(var) + r'\b', expr):
                    used.add(var)
    return sorted(used)

def _required_branches(
    cuts: dict[str, dict[str, str]],
    selections: list[str],
    available: set[str],
    requested: list[str],
     ) -> list[str]:
    '''Choose explicit branches or only branches referenced by the selections.'''
    if requested:
        return list(dict.fromkeys(requested))
    selected_cuts = {sel: cuts[sel] for sel in selections if sel in cuts}
    return branches_from_cuts(selected_cuts, sorted(available))

# __________________________________________
def write_table(
    file_path: str,
    file_name: str,
    headers: list[str],
    rows: list[list[str]],
    first_col_width: int = 10,
    other_col_width: int = 25,
    header_sep: bool = True,
    footer_lines: list[str] | None = None,
    file_type: str = 'txt'
     ) -> None:
    '''Write a formatted aligned ASCII table to file.

    Creates nicely aligned columns with configurable widths and optional header separator
    and footer lines. Useful for exporting analysis summary tables.

    Format:
    - Column 0 (narrow): Cut step names, decay channels, etc.
    - Columns 1+: Data values, uncertainties, statistics.
    - Rows shorter than header count are padded with empty strings.
    - Header separator: dashed line below headers if header_sep=True.

    Args:
        file_path (str): Output directory path (created if missing).
        file_name (str): Base file name (without extension).
        headers (list[str]): Column header strings.
        rows (list[list[str]]): List of table rows; shorter rows automatically padded.
        first_col_width (int, optional): Width of first column in characters. Defaults to 10.
        other_col_width (int, optional): Width of other columns in characters. Defaults to 25.
        header_sep (bool, optional): If True, add dashed line below headers. Defaults to True.
        footer_lines (list[str] | None, optional): Optional lines appended after table. Defaults to None.
        file_type (str, optional): File extension (e.g., 'txt', 'dat'). Defaults to 'txt'.
    '''
    mkdir(file_path)
    ncols = len(headers)
    # Set column widths: narrower for first column, equal for others
    widths = [first_col_width] + [other_col_width] * (ncols - 1)

    # Create format string for aligned columns
    fmt = '{:<%d} ' % widths[0] + ' '.join(['{:<%d}' % w for w in widths[1:]])
    with open(f'{file_path}/{file_name}.{file_type}', 'w') as f:
        # Write headers
        f.write(fmt.format(*headers) + '\n')
        if header_sep:
            # Write separator line with dashes
            sep = ['-' * widths[0]] + ['-' * w for w in widths[1:]]
            f.write(fmt.format(*sep) + '\n')
        # Write data rows, padding if necessary
        for row in rows:
            row_fixed = [str(r) for r in row] + [''] * (ncols - len(row))
            f.write(fmt.format(*row_fixed) + '\n')
        # Append optional footer lines
        if footer_lines:
            f.write('\n')
            for line in footer_lines:
                f.write(str(line) + '\n')


# _________________________________________
def get_cutflow(
    inDir: str,
    outDir: str,
    cat: str,
    sels: list[str],
    plots: dict[str, Any],
    colors: dict[str, dict[str, str]],
    legend: dict[str, dict[str, str]],
    cuts: dict[str, dict[str, str]],
    cuts_label: dict[str, dict[str, str]],
    z_decays: list[str],
    H_decays: list[str],
    format: list[str] = ['png'],
    ecm: int = 240,
    lumi: float = 10.8,
    sig_scale: float = 1.0,
    branches: list[str] = [],
    scaled: bool = True,
    tot: bool = False,
    json_file: bool = False,
    loc_json: str = ''
     ) -> None:

    '''Compute cutflows from parquet/ROOT inputs and render plots.

    Args:
        inDir (str): Directory containing processed event files.
        outDir (str): Base output directory for plots and tables.
        cat (str): Channel tag (``ee`` or ``mumu``).
        sels (list[str]): Selection names to evaluate.
        plots (dict[str, Any]): Process groups, with ``signals`` and
            ``backgrounds`` mapping output labels to sample lists. The optional
            ``decays`` group maps ``cat`` and ``tot`` to decay-label/sample maps.
        colors (dict[str, dict[str, str]]): Color mapping per process for styling.
        legend (dict[str, dict[str, str]]): Legend labels per process.
        cuts (dict[str, dict[str, str]]): Cut definitions per selection.
        cuts_label (dict[str, dict[str, str]]): Human-readable labels per cut step.
        z_decays (list[str]): Z decay modes included.
        H_decays (list[str]): Higgs decay modes included.
        format (list[str], optional): Output image formats. Defaults to ['png'].
        ecm (int, optional): Center-of-mass energy. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        sig_scale (float, optional): Scale factor applied to signal yields. Defaults to 1.0.
        branches (list[str], optional): Optional list of columns to read from files. Defaults to [].
        scaled (bool, optional): If True, scale yields by cross section and luminosity. Defaults to True.
        tot (bool, optional): If True, also produce totals across lepton categories. Defaults to False.
        json_file (bool, optional): If True, persist intermediate JSON summaries. Defaults to False.
        loc_json (str, optional): Path where JSON snapshots are stored. Defaults to ''.
    '''

    import numpy as np
    from .python.helper import (
        find_sample_files, get_processed, get_count, get_cuts,
        _col_from_file, get_flows, dump_json
    )
    from plotting.plotting import CutFlow, CutFlowDecays

    process_plots = {
        'signals': {name: list(samples)
                    for name, samples in plots.get('signals', {}).items()},
        'backgrounds': {name: list(samples)
                        for name, samples in plots.get('backgrounds', {}).items()},
    }
    decay_plots = plots.get('decays', {})
    if not isinstance(decay_plots.get('cat', {}), dict):
        decay_plots = {'cat': decay_plots, 'tot': decay_plots}

    active_decay_groups = [decay_plots.get('cat', {})]
    if tot:
        active_decay_groups.append(decay_plots.get('tot', {}))
    sample_groups = {
        name: samples
        for group in [*process_plots.values(), *active_decay_groups]
        for name, samples in group.items()
    }

    # Initialize event and file list dictionaries.
    events, file_list = {}, {}
    events: dict[str, dict[str, dict[str, dict[str, Any]]]]

    # Collect metadata from all samples
    LOGGER.info('Getting processed events')
    for proc, samples in sample_groups.items():
        LOGGER.info(f'For proc {proc}')
        for sample in tqdm(samples):
            events[sample] = {}

            # Find all input files for this sample
            flist = find_sample_files(inDir, sample)
            file_list[sample] = flist
            # Extract cross section and total processed event count for luminosity scaling
            events[sample]['cross-section']   = getMetaInfo(sample, rmww=True)
            events[sample]['eventsProcessed'] = get_processed(flist)

    selected_sels = [sel for sel in sels if sel in cuts]
    LOGGER.info('Getting cuts from DataFrame')
    for proc, samples in sample_groups.items():
        LOGGER.info(f'For proc {proc}')
        for sample in tqdm(samples):
            flist = file_list.get(sample, [])
            processed = events[sample]['eventsProcessed']
            xsec = events[sample]['cross-section']
            scale_sample = (lumi * 1e6 * xsec / processed) if (processed and scaled) else 1.0

            for sel in selected_sels:
                events[sample][sel] = {'raw_count': {cut: 0. for cut in cuts[sel]},
                                       'cut': {}, 'err': {},
                                       'filter': dict(cuts[sel]),
                                       'label':  dict(cuts_label[sel])}

            event_files = [f for f in flist if _col_from_file(f)]
            histogram_sels = [sel for sel in selected_sels
                              if all(not expr for expr in cuts[sel].values())]
            for sel in histogram_sels:
                raw_counts = get_cuts(flist, list(cuts[sel]))
                events[sample][sel]['raw_count'].update(raw_counts)

            data_sels = [sel for sel in selected_sels if sel not in histogram_sels]
            if event_files and data_sels:
                available = set().union(*(_col_from_file(f) for f in event_files))
                read_branches = _required_branches(cuts, data_sels, available, branches)
                for filepath in event_files:
                    df = get_df(filepath, read_branches)
                    if df is None or df.empty:
                        continue
                    for sel in data_sels:
                        df_mask = np.ones(len(df), dtype=bool)
                        empty_cuts = [cut for cut, expr in cuts[sel].items() if not expr]
                        histogram_counts = get_cuts([filepath], empty_cuts)
                        for cut, expr in cuts[sel].items():
                            if expr:
                                count, df_mask = get_count(df, df_mask, [], cut, expr)
                            else:
                                count = histogram_counts[cut]
                            events[sample][sel]['raw_count'][cut] += float(count)

            for sel in selected_sels:
                raw_counts = events[sample][sel]['raw_count']
                if not event_files and processed:
                    raw_counts['cut0'] = float(processed)
                scale = scale_sample if scaled else 1.0
                events[sample][sel]['cut'] = {cut: count * scale for cut, count in raw_counts.items()}
                events[sample][sel]['err'] = {cut: np.sqrt(count) * scale for cut, count in raw_counts.items()}

    # Save event counts to JSON for bookkeeping
    LOGGER.info('Dumping events in a json file')
    out_json = f'{loc_json}/{ecm}/{cat}'
    dump_json(events, out_json, 'events')

    def render_cutflows(
        sel: str,
        decay_group: str,
        suffix: str = '',
        total: bool = False,
    ) -> None:
        plot_config = {
            **process_plots,
            'decays': decay_plots.get(decay_group, {}),
        }
        flow, flow_decay = get_flows(
            events, plot_config, cuts, sel,
            json_file=json_file, tot=total,
            loc_json=out_json + f'/{sel}'
        )
        CutFlow(flow, outDir, cat, sel, process_plots,
                colors, legend, cuts, cuts_label,
                ecm=ecm, lumi=lumi, suffix=suffix,
                format=format, sig_scale=sig_scale, tot=total)
        CutFlowDecays(
            flow_decay, outDir, cat, sel, cuts_label,
            ecm=ecm, lumi=lumi, format=format, suffix=suffix,
            yMin=0 if not total or cat == 'qq' else -30,
            yMax=150 if not total else 160, tot=total
        )

    # Generate plots and tables for each selection
    for sel in (selection for selection in sels if selection in cuts):
        render_cutflows(sel, 'cat')
        if tot:
            render_cutflows(sel, 'tot', '_tot', True)
