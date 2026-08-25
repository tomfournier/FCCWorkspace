from typing import Any

from ..logger import get_logger

LOGGER = get_logger(__name__)


def datacard_txt(
        categories: list[str],
        sig_procs: dict[str, list[str]],
        bkg_procs: dict[str, list[str]],
        systs: dict[str, dict[str, Any]],
        suffix: str = '',
        mc_stats: bool = False,
        col_w: int = 12) -> str:

    import re

    procs = list((sig_procs | bkg_procs).keys())

    nprocs = len(procs)
    ncats  = len(categories)

    cats_str =       ''.join([f'{cat:<{col_w}}'  for cat in categories])
    procs_str =      ''.join([f'{proc:<{col_w}}' for proc in procs]) * ncats
    cats_procs_str = ''.join([f'{cat:<{col_w}}'  for cat in categories for _ in range(nprocs)])

    procs_idx = list(range(-len(sig_procs) + 1, len(bkg_procs) + 1, 1))
    cats_procs_idx_str = ''.join([f'{proc_idx:<{col_w}}' for proc_idx in procs_idx] * ncats)

    rates_cats  = f'{-1:<{col_w}}' *  ncats
    rates_procs = f'{-1:<{col_w}}' * (ncats * nprocs)

    sep = '#' * (22 + len(cats_procs_str))
    dc_lines = [
        'imax *',  # Number of bins
        'jmax *',  # Number of processes minus 1
        'kmax *',  # Number of systematics
        sep,
        f'shapes *        * datacard{suffix}.root $CHANNEL_$PROCESS $CHANNEL_$PROCESS_$SYSTEMATICS',
        f'shapes data_obs * datacard{suffix}.root $CHANNEL_asimov',
        sep,
        f'bin                        {cats_str}',    # Bin names
        f'observation                {rates_cats}',  # Observed event counts
        sep,
        f'bin                        {cats_procs_str}',      # Bin for each process
        f'process                    {procs_str}',           # Processes names
        f'process                    {cats_procs_idx_str}',  # Processes indices
        f'rate                       {rates_procs}',         # Expected rates
        sep
    ]

    for systName, syst in systs.items():
        syst_type = syst['type']
        syst_val  = syst['value']
        procs_to_apply = syst['procs']

        dc_tmp = f'{systName:<15} {syst_type:<10} '
        cats = syst.get('cats', categories)
        for _ in cats:
            for proc in procs:
                apply_proc = (isinstance(procs_to_apply, list) and proc in procs_to_apply) \
                    or (isinstance(procs_to_apply, str) and re.search(procs_to_apply, proc))
                val = syst_val if apply_proc else '-'
                dc_tmp += f'{val:<{col_w}}'
        dc_lines.append(dc_tmp)

    if mc_stats: dc_lines.append('* autoMCStats 1 1')

    return '\n'.join(dc_lines) + '\n'


def datacard_root(
        inputDir: str,
        hNames: list[str],
        categories: list[str],
        sig_procs: dict[str, list],
        bkg_procs: dict[str, list],
        systs_procs: dict[str, dict[str, Any]],
        rebin: int = 1,
        intLumi: float | int = 1):

    import ROOT

    hists, hists_asimov = [], {}
    proc_dict = sig_procs | bkg_procs

    def get_hist(proc_list: list[str], name: str):
        if isinstance(proc_list, str):
            proc_list = [proc_list]

        result = None
        for proc in proc_list:
            with ROOT.TFile(f'{inputDir}/{proc}.root', 'READ') as fIn:
                source = fIn.Get(name)
                if source is None:
                    raise KeyError(f'Histogram {name!r} not found in {proc}.root')
                hist = source.Clone()
                hist.SetDirectory(0)
            if result is None:
                result = hist
            else:
                result.Add(hist)

        if result is None:
            raise ValueError('A process list must contain at least one process')
        if intLumi != 1: result.Scale(intLumi)
        if rebin != 1: result.Rebin(rebin)
        return result

    for procName, procList in proc_dict.items():
        for i, cat in enumerate(categories):
            hist = get_hist(procList, hNames[i])
            hist.SetName(f'{cat}_{procName}')
            hists.append(hist)

            if cat not in hists_asimov:
                hist_asimov = hist.Clone()
                hist_asimov.SetName(f'{cat}_asimov')
                hist_asimov.SetDirectory(0)
                hists_asimov[cat] = hists_asimov
            else: hists_asimov[cat].Add(hist)

    for systName, systDict in systs_procs.items():
        for direction in ['Up', 'Down']:
            if direction in systDict:
                syst = systDict[direction]
            elif '*' in systDict:
                syst = systDict['*']
            else:
                LOGGER.error(f"Did not found {direction} or '*' in syst_procs[{systName}], "
                             f"syst_procs[{systName}] should either contain 'Up' and 'Down' or '*' key")
                raise KeyError(f"Missing {direction!r} or '*' in syst_procs[{systName!r}]")

            source_procs = syst['processes']
            if isinstance(source_procs, str):
                source_procs = [source_procs]
            target_procs = systDict['procs']
            if isinstance(target_procs, str):
                target_procs = [target_procs]

            if isinstance(source_procs, dict):
                target_sources = {proc: source_procs[proc] for proc in target_procs}
            elif len(target_procs) == 1:
                target_sources = {target_procs[0]: source_procs}
            elif len(source_procs) == len(target_procs):
                target_sources = dict(zip(target_procs, source_procs))
            else:
                raise ValueError(
                    f"Systematic {systName!r} has {len(target_procs)} target processes "
                    f"but {len(source_procs)} source processes; use equal-length lists "
                    "or a source-process dictionary"
                )

            for i, cat in enumerate(categories):
                for proc, proc_sources in target_sources.items():
                    variation = get_hist(proc_sources, hNames[i] + f'_{systName}{direction}')
                    hist = variation.Clone(f'{cat}_{proc}_{systName}{direction}')
                    hist.SetDirectory(0)
                    hists.append(hist)

    return hists, hists_asimov


def write_datacards(
        outputDir: str,
        dc_txt: str,
        hists: list,
        hists_asimov: list,
        suffix: str = ''):
    import ROOT

    with open(f'{outputDir}/datacard{suffix}.txt', 'w') as fOut:
        fOut.write(dc_txt)

    with ROOT.TFile(f'{outputDir}/datacard{suffix}.root', 'RECREATE') as fOut:
        for hist in hists + hists_asimov:
            fOut.Write(hist)

    return None
