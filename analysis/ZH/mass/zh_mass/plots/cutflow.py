'''Cutflow and selection efficiency plotting.

Provides:
- Event count aggregation across selection cuts: `get_cutflow()`, `get_flows()`.
- Cutflow visualization with signal/background stacking: `CutFlow()`.
- Decay-mode efficiency analysis: `CutFlowDecays()`, `Efficiency()`.
- Output table generation: `write_table()`.
- Utility functions for cut expression parsing: `branches_from_cuts()`.
- Integration with ROOT graphics and file I/O for large-scale analyses.

Functions:
- `get_cutflow()`: Main entry point; reads events, applies cuts, and generates plots/tables.
- `CutFlow()`: Render stacked cutflow histogram with signal overlay and significance computation.
- `CutFlowDecays()`: Plot efficiency curves normalized to first cut for each Higgs decay mode.
- `Efficiency()`: Create efficiency summary plots with uncertainty bands and tabular summary.
- `write_table()`: Format and write ASCII tables with configurable column widths and headers.
- `branches_from_cuts()`: Extract variable names referenced in cut filter expressions.

Conventions:
- Cut steps indexed sequentially (cut0, cut1, ...) with user-provided labels.
- Yields computed with luminosity and cross-section scaling when available.
- Uncertainties treated as Poisson (sqrt(N)) on raw counts, scaled linearly.
- Significance computed per cut as S/sqrt(S+B) where S=signal, B=background total.
- Efficiency normalized to first cut (cut0) for decay mode comparisons.
- Channel-specific plots (ee/mumu) optionally combined into totals (tot).

Usage:
- Analyze event selection efficiency by computing cumulative event counts per cut step.
- Generate publication-ready cutflow plots with stacked backgrounds and signal overlay.
- Validate selection consistency across Higgs decay modes via efficiency tables.
- Export yield summaries and efficiency pulls as both plots and tabular data.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import os, copy

from tqdm import tqdm
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from ..tools.utils import get_df, mkdir
from ..tools.process import getMetaInfo
from ..logger import get_logger

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
    '''Write a formatted text table to file.

    Creates a nicely aligned ASCII table with configurable column widths.

    Args:
        file_path (str): Output directory path.
        file_name (str): Base file name (without extension).
        headers (list[str]): Column header strings.
        rows (list[list[str]]): List of table rows; short rows are automatically padded.
        first_col_width (int, optional): Width of first column in characters. Defaults to 10.
        other_col_width (int, optional): Width of other columns in characters. Defaults to 25.
        header_sep (bool, optional): If True, add dashed separator line below headers. Defaults to True.
        footer_lines (list[str] | None, optional): Optional list of lines appended after the table. Defaults to None.
        file_type (str, optional): File extension. Defaults to 'txt'.
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

# ______________________________________
def CutFlow(
    flow: dict[str,
               dict[str, Any |
                    dict[str, float]]],
    outDir: str,
    cat: str,
    sel: str,
    procs: list[str],
    colors: dict[str, dict[str, int]],
    legend: dict[str, dict[str, str]],
    cuts: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    ecm: int = 240,
    lumi: float = 10.8,
    outName: str = 'cutFlow',
    format: list[str] = ['png'],
    suffix: str = '',
    sig_scale: float = 1.0,
    yMin: float = 1e4,
    yMax: float = 1e10
     ) -> None:

    '''Render cutflow plot with signal, backgrounds, and significance.

    Produces a stacked histogram showing yields across cut steps, overlaid with
    signal histogram (optionally scaled). Computes and exports significance values
    and yields with uncertainties to a text table.

    Args:
        flow (dict[str, dict[str, ROOT.TH1 | dict[str, float]]]): Histograms and properties indexed by process name.
        outDir (str): Base directory for output plots and tables.
        cat (str): Channel (``ee`` or ``mumu``).
        sel (str): Selection identifier for retrieving cut definitions and labels.
        procs (list[str]): Process names in order [signal, background1, background2, ...].
        colors (dict[str, dict[str, ROOT.TColor]]): ROOT color mappings per process.
        legend (dict[str, dict[str, str]]): Human-readable labels for legend.
        cuts (dict[str, dict[str, str]]): Cut expression definitions per selection.
        labels (dict[str, dict[str, str]]): Axis labels corresponding to each cut step.
        ecm (int, optional): Beam energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        outName (str, optional): Output file stem. Defaults to 'cutFlow'.
        format (list[str], optional): Image formats (e.g., ['png', 'pdf']). Defaults to ['png'].
        suffix (str, optional): String appended to file names. Defaults to ''.
        sig_scale (float, optional): Multiplicative factor for signal visibility. Defaults to 1.0.
        yMin (float, optional): Log-scale Y-axis minimum. Defaults to 1e4.
        yMax (float, optional): Log-scale Y-axis maximum. Defaults to 1e10.
    '''

    import ROOT
    import numpy as np
    from .root import plotter
    from .root.helper import (
        mk_legend,
        style_hist,
        build_cfg,
    )
    from .root.plotter import setup_cutflow_hist, save_canvas

    # Create legend in upper right corner
    leg = mk_legend(
        len(procs), columns=1,
        x1=0.55, y1=0.99,
        x2=0.99, y2=0.9,
        border_size=0,
        fill_style=0,
        text_size=0.03,
        set_margin=0.2
    )

    # Setup signal histogram
    h_sig = flow[procs[0]]['hist'][0]
    hists_yields = [copy.deepcopy(h_sig)]
    style_hist(
        h_sig,
        color=colors[procs[0]],
        width=4, style=1,
        scale=sig_scale
    )

    # Add signal entry to legend with scale factor if applicable
    leg.AddEntry(h_sig, f'{legend[procs[0]]} (#times {int(sig_scale)})'
                        if sig_scale!=1 else legend[procs[0]], 'L')

    # Setup background stack and total background histogram
    st, h_bkg_tot = ROOT.THStack(), None
    st.SetName('stack')
    for bkg in procs[1:]:
        h_bkg = flow[bkg]['hist'][0]

        # Accumulate total background for overlay
        if h_bkg_tot is None:
            h_bkg_tot = h_bkg.Clone('h_bkg_tot')
        else:
            h_bkg_tot.Add(h_bkg)

        style_hist(
            h_bkg,
            color=ROOT.kBlack,
            width=1, style=1,
            fill_color=colors[bkg]
        )
        leg.AddEntry(h_bkg, legend[bkg], 'F')
        st.Add(h_bkg)
        hists_yields.append(h_bkg)

    # Style total background outline
    h_bkg_tot.SetLineColor(ROOT.kBlack)
    h_bkg_tot.SetLineWidth(2)

    # Extract yields and errors for significance calculation
    nbins = len(cuts[sel])
    contents = np.vstack([
        np.fromiter((float(hist.GetBinContent(i+1)) for i in range(nbins)), dtype=float)
        for hist in hists_yields
    ])

    # Compute significance for each cut step: S/sqrt(S+B)
    nsig = contents[0] / sig_scale
    nbkg = contents[1:].sum(axis=0) if contents.shape[0] > 1 else np.zeros_like(nsig)
    denom = np.sqrt(nsig + nbkg)
    # Handle zero division gracefully with -1 flag
    with np.errstate(divide='ignore', invalid='ignore'):
        sig_arr = np.where((nsig + nbkg)==0, -1.0, nsig / denom)
    significances = sig_arr.tolist()

    # Configure plot: log-y scale, set axis ranges
    cfg = build_cfg(
        h_sig,
        logX=False, logY=True,
        xmin=0,xmax=nbins,
        ymin=yMin,ymax=yMax,
        ecm=ecm, lumi=lumi,
        xtitle='None',
        cutflow=True
    )
    plotter.cfg = cfg

    # Setup canvas with cut step labels
    cat_label = 'e' if cat=='ee' else '#mu'
    canvas, dummy = setup_cutflow_hist(nbins, labels[sel], cat_label)

    # Draw histograms in order: dummy frame, backgrounds, total bkg, signal, legend
    dummy.Draw('HIST')
    st.Draw('SAME HIST')
    h_bkg_tot.Draw('SAME HIST')
    h_sig.Draw('SAME HIST')
    leg.Draw('SAME')

    # Save plot to disk
    s = sel.replace('_high', '').replace('_low', '')
    if '_high' in sel: d = 'high'
    elif '_low' in sel: d = 'low'
    else: d = 'nominal'
    out = f'{outDir}/yield/{s}/{d}/cutflow'
    save_canvas(canvas, out, outName, suffix, format)

    # Build table rows: cut index, significance, and yields with errors per process
    rows = []
    for i in range(nbins):
        row = [f'Cut {i}', f'{significances[i]:.3f}']
        for histProc in hists_yields:
            yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
            row.append('%.2e +/- %.2e' % (yield_, err))
        rows.append(row)
    # Export yields table to file
    write_table(
        out, outName+suffix,
        headers=['Cut', 'Significance'] + procs,
        rows=rows,
        first_col_width=10,
        other_col_width=25
    )


# _________________________________________
def get_cutflow(
    inDir: str,
    outDir: str,
    cat: str,
    sels: list[str],
    procs: list[str],
    processes: dict[str, list[str]],
    colors: dict[str, dict[str, str]],
    legend: dict[str, dict[str, str]],
    cuts: dict[str, dict[str, str]],
    cuts_label: dict[str, dict[str, str]],
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
        procs (list[str]): Processes for plotting; signal first.
        procs_decays (list[str]): Processes including decay-specific entries.
        processes (dict[str, list[str]]): Map of process -> list of sample identifiers.
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

    from .python.helper import (
        _col_from_file,
        find_sample_files,
        is_there_events,
        get_processed,
        get_count,
        get_flows,
        dump_json
    )

    # Initialize event and file list dictionaries
    events, file_list = {}, {}

    # Collect metadata from all samples
    LOGGER.info('Getting processed events')
    for proc in tqdm(procs):
        for sample in processes[proc]:
            events[sample] = {}

            # Find all input files for this sample
            flist = find_sample_files(inDir, sample)
            file_list[sample] = flist
            # Extract cross section and total processed event count for luminosity scaling
            events[sample]['cross-section']   = getMetaInfo(sample, rmww=False)
            events[sample]['eventsProcessed'] = get_processed(flist)

    # Preload column names from first file of each sample (avoids repeated I/O)
    col_map: dict[str, set] = {}
    for sample, flist in file_list.items():
        if flist and len(flist) > 0:
            col_map[sample] = _col_from_file(os.fspath(flist[0]))
        else:
            col_map[sample] = set()

    LOGGER.info('Getting cuts from DataFrame')
    for proc in procs:
        LOGGER.debug(f'From {proc}')
        for sample in processes[proc]:
            LOGGER.info(f'From sample {sample}')

            flist = file_list.get(sample, [])
            has_file = bool(flist)
            processed = events[sample]['eventsProcessed']
            xsec = events[sample]['cross-section']
            # Compute luminosity scaling factor: lumi [ab^-1] * 1e6 [fb/ab] * xsec [fb] / N_events
            scale_sample = (lumi * 1e6 * xsec / processed) if (processed and scaled) else 1.0

            # Process events from files if available
            if has_file and is_there_events(sample, inDir):
                for sel in sels:
                    if sel not in cuts: continue
                    LOGGER.info(f'For selection: {sel}')
                    # Initialize cut statistics storage
                    events[sample][sel] = {'raw_count': {}, 'cut': {}, 'err': {}, 'filter': {}}
                    raw_counts = {cut:0.0 for cut in cuts[sel].keys()}

                    # Process each event file
                    for f in flist:
                        df = get_df(f, branches) if branches else get_df(f)
                        if df is None or df.empty:
                            # Store filter expressions even if no data
                            for cut, filter in cuts[sel].items():
                                events[sample][sel]['filter'][cut] = filter
                            continue

                        # Apply cuts sequentially, accumulating event counts
                        df_mask = np.ones(len(df), dtype=bool)
                        for cut, filter in cuts[sel].items():
                            events[sample][sel]['filter'][cut] = filter
                            # Evaluate cut expression and accumulate passing events
                            count, df_mask = get_count(
                                df, df_mask, [f],
                                cut, filter,
                                columns=col_map.get(sample, None)
                            )
                            raw_counts[cut] += float(count)

                        # Scale yields and compute Poisson errors
                        for cut, total_raw in raw_counts.items():
                            scale = scale_sample if scaled else 1.0
                            events[sample][sel]['raw_count'][cut] = float(total_raw)
                            events[sample][sel]['cut'][cut] = float(total_raw) * scale
                            events[sample][sel]['err'][cut] = np.sqrt(total_raw) * scale

            else:
                # If no files available, use only generator-level first cut
                for sel in sels:
                    if sel not in cuts: continue
                    events[sample][sel] = {'raw_count': {}, 'cut': {}, 'err': {}, 'filter': {}}
                    for cut, filter in cuts[sel].items():
                        events[sample][sel]['filter'][cut] = filter
                        # Only populate first cut with scaled generator events
                        if cut=='cut0' and processed:
                            scale = lumi * 1e6 * xsec / processed
                            events[sample][sel]['raw_count'][cut] = processed
                            events[sample][sel]['cut'][cut] = scale * processed
                            events[sample][sel]['err'][cut] = scale * np.sqrt(processed)
                        else:
                            events[sample][sel]['raw_count'][cut] = 0
                            events[sample][sel]['cut'][cut] = 0
                            events[sample][sel]['err'][cut] = 0

    # Save event counts to JSON for bookkeeping
    LOGGER.info('Dumping events in a json file')
    out_json = f'{loc_json}/{ecm}/{cat}'
    dump_json(events, out_json, f'events_{cat}_{ecm}')

    # Generate plots and tables for each selection
    for sel in sels:
        if sel not in cuts: continue
        if not tot:
            # Single-channel plots
            LOGGER.info('Preparing dictionary for cutflow plots')
            # Construct histograms from cut yields
            flow, flow_decay = get_flows(
                procs, processes,
                cuts, events,
                cat, sel,
                ecm=ecm, json_file=json_file,
                loc_json=out_json+f'/{sel}'
            )
            # Render main cutflow with signal, backgrounds, and significance
            LOGGER.info('Making Cutflow plot')
            CutFlow(
                flow, outDir, cat, sel,
                procs, colors, legend, cuts, cuts_label,
                ecm=ecm, lumi=lumi,
                outName='cutFlow', format=format,
                sig_scale=sig_scale,
                yMin=1e4 if ecm==240 else (1e2 if ecm==365 else 1)
            )
        else:
            # Compute both channel-specific and combined plots
            procs_cat, procs_tot = copy.deepcopy(procs), copy.deepcopy(procs)
            procs_cat[0] = f'Z{cat}H'

            # Channel-specific cutflow
            LOGGER.info('Preparing dictionary for cutflow plots')
            flow, flow_decay = get_flows(
                procs_cat, processes,
                cuts, events,
                cat, sel,
                ecm=ecm,
                json_file=json_file,
                loc_json=out_json+f'/{sel}'
            )
            LOGGER.info('Making Cutflow plot')
            CutFlow(
                flow, outDir, cat, sel,
                procs_cat, colors, legend, cuts, cuts_label,
                ecm=ecm, lumi=lumi, outName='cutFlow',
                format=format, sig_scale=sig_scale,
                yMin=1e4 if ecm==240 else (1e2 if ecm==365 else 1)
            )

            # Combined (ee+mumu) cutflow
            flow_tot, flow_decay_tot = get_flows(
                procs_tot, processes,
                cuts, events, cat, sel,
                ecm=ecm, json_file=json_file,
                loc_json=loc_json+f'/{sel}',
                tot=True
            )
            LOGGER.info('Making Cutflow plot')
            CutFlow(
                flow_tot, outDir, cat, sel,
                procs_tot, colors, legend, cuts, cuts_label,
                ecm=ecm, lumi=lumi,
                suffix='_tot', format=format,
                sig_scale=sig_scale,
                yMin=1e4 if ecm==240 else (1e2 if ecm==365 else 1)
            )
