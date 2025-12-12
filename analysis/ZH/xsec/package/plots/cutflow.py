import os, re, copy, ROOT

import numpy as np

from tqdm import tqdm

from .root import plotter
from ..tools.utils import get_df, mkdir
from ..tools.process import getMetaInfo
from ..config import h_colors, h_labels

from .root.helper import setup_latex, mk_legend, canvas_margins
from .root.helper import style_hist, build_cfg, save_plot
from .root.plotter import setup_cutflow_hist, save_canvas

from .python.helper import find_sample_files, get_processed, is_there_events
from .python.helper import _col_from_file, get_count, get_flows, dump_json

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)



#____________________________________________
def branches_from_cuts(cuts: dict[str,
                                  dict[str, 
                                       str]], 
                       variables: list
                       ) -> list:
    used = set()
    for sel_cuts in cuts.values():
        for expr in sel_cuts.values():
            if not expr:
                continue
            for var in variables:
                if re.search(r'\b' + re.escape(var) + r'\b', expr):
                    used.add(var)
    return sorted(used)

#______________________________________________________
def write_table(file_path: str,
                file_name: str,
                headers: list[str],
                rows: list[list[str]],
                first_col_width: int = 10,
                other_col_width: int = 25,
                header_sep: bool = True,
                footer_lines: list[str] | None = None,
                file_type: str = 'txt'
                ) -> None:
    mkdir(file_path)
    ncols = len(headers)
    widths = [first_col_width] + [other_col_width] * (ncols - 1)

    fmt = '{:<%d} ' % widths[0] + ' '.join(['{:<%d}' % w for w in widths[1:]])
    with open(f'{file_path}/{file_name}.{file_type}', 'w') as f:
        f.write(fmt.format(*headers) + '\n')
        if header_sep:
            sep = ['-' * widths[0]] + ['-' * w for w in widths[1:]]
            f.write(fmt.format(*sep) + '\n')
        for row in rows:
            row_fixed = [str(r) for r in row] + [''] * (ncols - len(row))
            f.write(fmt.format(*row_fixed) + '\n')
        if footer_lines:
            f.write('\n')
            for line in footer_lines:
                f.write(str(line) + '\n')

#_____________________________________________________
def CutFlow(flow: dict[str,
                       dict[str,
                            ROOT.TH1 | dict[str,
                                            float]]], 
            outDir: str, 
            cat: str, 
            sel: str, 
            procs: list[str], 
            colors: dict[str, dict[str, ROOT.TColor]], 
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
            yMax: float = 1e10) -> None:

    leg = mk_legend(len(procs), columns=1,
                    x1=0.55, y1=0.99, 
                    x2=0.99, y2=0.9,
                    border_size=0,
                    fill_style=0,
                    text_size=0.03,
                    set_margin=0.2)

    h_sig = flow[procs[0]]['hist'][0]
    hists_yields = [copy.deepcopy(h_sig)]
    style_hist(h_sig, 
               color=colors[procs[0]],
               width=4, style=1, 
               scale=sig_scale)

    leg.AddEntry(h_sig, f'{legend[procs[0]]} (#times {int(sig_scale)})' 
                    if sig_scale!=1 else legend[procs[0]], 'L')

    st, h_bkg_tot = ROOT.THStack(), None
    st.SetName('stack')
    for bkg in procs[1:]:
        h_bkg = flow[bkg]['hist'][0]

        if h_bkg_tot is None: 
            h_bkg_tot = h_bkg.Clone('h_bkg_tot')
        else: 
            h_bkg_tot.Add(h_bkg)

        style_hist(h_bkg, 
                   color=ROOT.kBlack,
                   width=1, style=1,
                   fill_color=colors[bkg])
        leg.AddEntry(h_bkg, legend[bkg], 'F')
        st.Add(h_bkg)
        hists_yields.append(h_bkg)

    h_bkg_tot.SetLineColor(ROOT.kBlack)
    h_bkg_tot.SetLineWidth(2)

    nbins = len(cuts[sel])
    contents = np.vstack([
        np.fromiter(( float(hist.GetBinContent(i+1)) for i in range(nbins) ), dtype=float) 
        for hist in hists_yields
    ])
    errors = np.vstack([
        np.fromiter(( float(hist.GetBinError(i+1)) for i in range(nbins) ), dtype=float) 
        for hist in hists_yields
    ])

    nsig = contents[0] / sig_scale
    nbkg = contents[1:].sum(axis=0) if contents.shape[0] > 1 else np.zeros_like(nsig)
    denom = np.sqrt(nsig + nbkg)
    with np.errstate(divide='ignore', invalid='ignore'):
        sig_arr = np.where((nsig + nbkg)==0, -1.0, nsig / denom)
    significances = sig_arr.tolist()

    cfg = build_cfg(h_sig, 
                    logX=False, logY=True,
                    xmin=0,xmax=nbins,
                    ymin=yMin,ymax=yMax, 
                    xtitle='None',
                    cutflow=True)
    plotter.cfg = cfg

    cat_label = 'e' if cat=='ee' else '#mu'
    canvas, dummy = setup_cutflow_hist(nbins, labels[sel], cat_label)

    dummy.Draw('HIST')
    st.Draw('SAME HIST')
    h_bkg_tot.Draw('SAME HIST')
    h_sig.Draw('SAME HIST')
    leg.Draw('SAME')

    out = f'{outDir}/yield/{sel}/cutflow'
    save_canvas(canvas, out, outName, suffix, format)

    rows = []
    for i in range(nbins):
        row = ['Cut %d'%i, '%.3f'%significances[i]]
        for histProc in hists_yields:
            yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
            row.append('%.2e +/- %.2e' % (yield_, err))
        rows.append(row)
    write_table(out, outName+suffix,
                headers=['Cut', 'Significance'] + procs,
                rows=rows,
                first_col_width=10,
                other_col_width=25)

#__________________________________________________________
def CutFlowDecays(flow: dict[str,
                             dict[str,
                                  ROOT.TH1 | dict[str,
                                                  float]]], 
                  outDir: str, 
                  cat: str, 
                  sel: str, 
                  h_decays: list[str], 
                  cuts: dict[str, dict[str, str]], 
                  labels: dict[str, dict[str, str]], 
                  suffix: str = '', 
                  ecm: int = 240, 
                  lumi: float = 10.8, 
                  outName: str = 'cutFlow_decays', 
                  format: list[str] = ['png'], 
                  yMin: float | int = 0, 
                  yMax: float | int = 150) -> None:

    leg = mk_legend(len(h_decays), columns=4, 
                    x1=0.2, y1=0.925, 
                    x2=0.95, y2=0.925,
                    border_size=0,
                    fill_style=0,
                    text_size=0.03,
                    set_margin=0.25)

    hists, hist_yield = [], []
    nbins = len(cuts[sel])
    eff_final, eff_final_err = [], []

    contents, errors = [], []
    for h_decay in h_decays:
        h_sig = flow[h_decay]['hist'][0]
        hist_yield.append(h_sig.Clone(f'yield_{h_decay}'))
        style_hist(h_sig, 
                   color=h_colors[h_decay],
                   width=2, style=1, 
                   scale=100./h_sig.GetBinContent(1))

        leg.AddEntry(h_sig, h_labels[h_decay], 'L')
        hists.append(h_sig)

        eff_final.append(float(h_sig.GetBinContent(nbins)))
        eff_final_err.append(float(h_sig.GetBinError(nbins)))

        contents.append(np.fromiter((float(h_sig.GetBinContent(i+1)) 
                                     for i in range(nbins)), dtype=float))
        errors.append(np.fromiter((float(h_sig.GetBinError(i+1)) 
                                   for i in range(nbins)), dtype=float))
        

    hist_tot = hists[0].Clone('h_tot')
    for hist in hists[1:]:
        hist_tot.Add(hist)
    hist_tot.Scale(1.0 / len(h_decays))
    eff_avg = hist_tot.GetBinContent(nbins)
    eff_avg_err = hist_tot.GetBinError(nbins)
    eff_min, eff_max = eff_avg-min(eff_final), max(eff_final)-eff_avg

    cfg = build_cfg(hists[-1], logX=False, logY=False,
                    xmin=0, xmax=nbins,
                    ymin=yMin, ymax=yMax, 
                    xtitle='None', 
                    ytitle='Selection efficiency [%]',
                    cutflow=True)
    plotter.cfg = cfg

    cat_label = 'e' if cat=='ee' else '#mu'
    canvas, dummy = setup_cutflow_hist(nbins, labels[sel], cat_label)
    dummy.Draw('HIST')

    txt = setup_latex(0.04, 11, text_color=1, text_font=42)
    txt.DrawLatex(0.2, 0.2, f'Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %')
    txt.DrawLatex(0.2, 0.15, f'Min/max: {eff_min:.2f}/{eff_max:.2f}')
    txt.Draw('SAME')

    for hist in hists:
        hist.Draw('SAME HIST')
    leg.Draw('SAME')

    out = f'{outDir}/yield/{sel}/cutflow'
    save_canvas(canvas, out, outName, suffix, format)

    rows = []
    for i in range(nbins):
        row = ['Cut %d'%i]
        for j in range(len(hist_yield)):
            yield_, err = contents[j][i], errors[j][i]
            row.append('%.2e +/- %.2e' % (yield_, err))
        rows.append(row)
    write_table(out, outName+suffix,
                headers=['Cut'] + h_decays,
                rows=rows,
                first_col_width=10,
                other_col_width=25)
    
    del canvas
    
    out_eff = f'{outDir}/yield/{sel}/efficiency'
    Efficiency(out_eff, contents, errors, eff_final, eff_final_err, 
               eff_avg, eff_avg_err, eff_min, eff_max, h_decays,
               nbins, suffix=suffix, format=format, ecm=ecm, lumi=lumi)

#__________________________
def Efficiency(outDir: str, 
               contents: list[np.ndarray], 
               errors: list[np.ndarray], 
               eff_final: float, 
               eff_final_err: float, 
               eff_avg: float, 
               eff_avg_err: float, 
               eff_min: float, 
               eff_max: float, 
               h_decays: list[str], 
               nbins: int, 
               suffix: str = '', 
               format: list[str] = ['png'], 
               ecm: int = 240, 
               lumi: float = 10.8, 
               ) -> None:

    # make final efficiency pull plot (compact)
    xMin, xMax = int(min(eff_final))-5, int(max(eff_final))+3
    h_pulls = ROOT.TH2F('pulls', 'pulls', (xMax-xMin)*10, xMin, xMax, len(h_decays)+1, 0, len(h_decays)+1)
    g_pulls = ROOT.TGraphErrors(len(h_decays)+1)

    g_pulls.SetPoint(0, eff_avg, 0.5); g_pulls.SetPointError(0, eff_avg_err, 0.)
    h_pulls.GetYaxis().SetBinLabel(1, 'Average')

    for i,h_decay in enumerate(h_decays):
        g_pulls.SetPoint(i+1, eff_final[i], float(i+1) + 0.5)
        g_pulls.SetPointError(i+1, eff_final_err[i], 0.)
        h_pulls.GetYaxis().SetBinLabel(i+2, h_labels[h_decay])

    canvas = plotter.canvas(800, 800)
    canvas_margins(canvas, top=0.08, bottom=0.1, left=0.15, right=0.05)
    canvas.SetFillStyle(4000)
    canvas.SetGrid(1, 0)
    canvas.SetTickx(1)

    h_pulls.GetXaxis().SetTitle('Selection efficiency [%]')
    h_pulls.GetXaxis().SetTitleSize(0.04)
    h_pulls.GetXaxis().SetLabelSize(0.035)
    h_pulls.GetXaxis().SetTitleOffset(1)
    h_pulls.GetYaxis().SetLabelSize(0.055)
    h_pulls.GetYaxis().SetTickLength(0)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.SetNdivisions(506, 'XYZ')
    h_pulls.Draw('HIST 0')

    maxx = len(h_decays)+1
    line = ROOT.TLine(eff_avg, 0, eff_avg, maxx)
    line.SetLineColor(ROOT.kGray)
    line.SetLineWidth(2)
    line.Draw('SAME')

    shade = ROOT.TGraph()
    shade.SetPoint(0, eff_avg-eff_avg_err, 0); shade.SetPoint(1, eff_avg+eff_avg_err, 0)
    shade.SetPoint(2, eff_avg+eff_avg_err, maxx); shade.SetPoint(3, eff_avg-eff_avg_err, maxx)
    shade.SetPoint(4, eff_avg-eff_avg_err, 0)
    shade.SetFillColor(16); shade.SetFillColorAlpha(16, 0.35); shade.Draw('SAME F')

    g_pulls.SetMarkerSize(1.2); g_pulls.SetMarkerStyle(20); g_pulls.SetLineWidth(2)
    g_pulls.Draw('P0 SAME')

    latex = setup_latex(0.045, 30, text_color=1, text_font=42)
    latex.DrawLatex(0.95, 0.925, f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}')
    latex = setup_latex(0.045, 13, text_color=1, text_font=42)
    latex.DrawLatex(0.15, 0.96, '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}')
    txt = setup_latex(0.04, 11, text_color=1, text_font=42)
    txt.DrawLatex(0.2, 0.2, f'Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %')
    txt.DrawLatex(0.2, 0.15, f'Min/max: {eff_min:.2f}/{eff_max:.2f}')
    txt.Draw('SAME')

    save_plot(canvas, outDir, 'selection_efficiency', suffix, format)

    # efficiency table
    rows = []
    for i in range(nbins):
        row = [f'Cut {i}']
        for arr, err in zip(contents, errors):
            row.append('%.2f +/- %.2f' % (arr[i], err[i]))
        rows.append(row)
    write_table(outDir, 'selection_efficiency'+suffix,
                headers=['Cut']+h_decays,
                rows=rows,
                first_col_width=10,
                other_col_width=18,
                footer_lines=[
                    f'Average: {eff_avg:.3f} +/- {eff_avg_err:.3f}',
                    f'Min/max: {eff_min:.3f}/{eff_max:.3f}'])

#_____________________________________________________
def get_cutflow(inDir: str, 
                outDir: str, 
                cat: str, 
                sels: list[str], 
                procs: list[str], 
                procs_decays: list[str],
                processes: dict[str, list[str]], 
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
                loc_json: str = '') -> None:

    events, file_list = {}, {}
    if not tot: 
        procs[0]        = f'Z{cat}H'
        procs_decays[0] = f'z{cat}h'

    print('----->[Info] Getting processed events')
    for proc in tqdm(procs_decays):
        for sample in processes[proc]:
            events[sample] = {}

            flist = find_sample_files(inDir, sample)
            file_list[sample] = flist
            events[sample]['cross-section']   = getMetaInfo(sample, remove=True)
            events[sample]['eventsProcessed'] = get_processed(flist)

    col_map: dict[str, set] = {}
    for sample, flist in file_list.items():
        if flist and len(flist) > 0:
            col_map[sample] = _col_from_file(os.fspath(flist[0]))
        else:
            col_map[sample] = set()
    
    print('\n----->[Info] Getting cuts from dataframe')
    for proc in procs_decays:
        print(f'\n----->[Info] From {proc}')
        for sample in processes[proc]:
            print(f'------>[Info] sample {sample}')
            
            flist = file_list.get(sample, [])
            has_file = bool(flist)
            processed = events[sample]['eventsProcessed']
            xsec = events[sample]['cross-section']
            scale_sample = (lumi * 1e6 * xsec / processed) if (processed and scaled) else 1.0

            if has_file and is_there_events(sample, inDir):
                for sel in sels:
                    if not sel in cuts: continue
                    print(f'------->[Info] For selection: {sel}')
                    events[sample][sel] = {'cut': {}, 'err': {}, 'filter': {}}
                    raw_counts = {cut:0.0 for cut in cuts[sel].keys()}

                    for f in flist:
                        df = get_df(f, branches) if branches else get_df(f)
                        if df is None or df.empty:
                            for cut, filter in cuts[sel].items():
                                events[sample][sel]['filter'][cut] = filter
                            continue

                        df_mask = np.ones(len(df), dtype=bool)
                        for cut, filter in cuts[sel].items():
                            events[sample][sel]['filter'][cut] = filter
                            count, df_mask = get_count(df, df_mask, [f],
                                                       cut, filter,
                                                       columns=col_map.get(sample, None))
                            raw_counts[cut] += float(count)
                        
                        for cut, total_raw in raw_counts.items():
                            scale = scale_sample if scaled else 1.0
                            events[sample][sel]['cut'][cut] = float(total_raw) * scale
                            events[sample][sel]['err'][cut] = np.sqrt(total_raw) * scale
                        
            else:
                for sel in sels:
                    if not sel in cuts: continue
                    events[sample][sel] = {'cut': {}, 'err': {}, 'filter': {}}
                    for cut, filter in cuts[sel].items():
                        events[sample][sel]['filter'][cut] = filter
                        if cut=='cut0' and processed:
                            scale = lumi * 1e6 * xsec / processed
                            events[sample][sel]['cut'][cut] = scale * processed
                            events[sample][sel]['err'][cut] = scale * np.sqrt(processed)
                        else:
                            events[sample][sel]['cut'][cut] = 0
                            events[sample][sel]['err'][cut] = 0

    print('----->[Info] Dumping events in a json file')
    dump_json(events, loc_json, 'events')
    
    for sel in sels:
        if not sel in cuts: continue
        if not tot:
            print('----->[Info] Preparing dictionary for cutflow plots')
            flow, flow_decay = get_flows(procs, processes, 
                                         cuts, events, 
                                         cat, sel, 
                                         z_decays, H_decays, 
                                         ecm=ecm, 
                                         json_file=json_file, 
                                         loc_json=loc_json+f'/{sel}')
            print('----->[Info] Making Cutflow plot')
            CutFlow(flow, outDir, cat, sel, 
                    procs, colors, legend, cuts, cuts_label,
                    ecm=ecm, lumi=lumi, 
                    outName='cutFlow', format=format, 
                    sig_scale=sig_scale)
            print('----->[Info] Making CutflowDecays plot')
            CutFlowDecays(flow_decay, outDir, cat, sel, 
                          H_decays, cuts, cuts_label, 
                          ecm=ecm, lumi=lumi, 
                          format=format)
        else:
            procs_cat, procs_tot = copy.deepcopy(procs), copy.deepcopy(procs)
            procs_cat[0] = f'Z{cat}H'

            print('----->[Info] Preparing dictionary for cutflow plots')
            flow, flow_decay = get_flows(procs_cat, processes, 
                                         cuts, events, 
                                         cat, sel, 
                                         z_decays, H_decays, 
                                         ecm=ecm, 
                                         json_file=json_file, 
                                         loc_json=loc_json+f'/{sel}')
            print('----->[Info] Making Cutflow plot')
            CutFlow(flow, outDir, cat, sel, 
                    procs_cat, colors, legend, cuts, cuts_label, 
                    ecm=ecm, lumi=lumi, outName='cutFlow', 
                    format=format, sig_scale=sig_scale)
            print('----->[Info] Making CutflowDecays plot')
            CutFlowDecays(flow_decay, outDir, cat, sel, 
                          H_decays, cuts, cuts_label, 
                          ecm=ecm, lumi=lumi, 
                          format=format)
            
            flow_tot, flow_decay_tot = get_flows(procs_tot, processes, 
                                                 cuts, events, cat, sel, 
                                                 z_decays, H_decays, 
                                                 ecm=ecm, 
                                                 json_file=json_file, 
                                                 loc_json=loc_json+f'/{sel}', 
                                                 tot=True)
            print('----->[Info] Making Cutflow plot')
            CutFlow(flow_tot, outDir, cat, sel, 
                    procs_tot, colors, legend, cuts, cuts_label, 
                    ecm=ecm, lumi=lumi, 
                    suffix='_tot', format=format, 
                    sig_scale=sig_scale)
            print('----->[Info] Making CutflowDecays plot')
            CutFlowDecays(flow_decay_tot, outDir, cat, sel, 
                          H_decays, cuts, cuts_label, 
                          ecm=ecm, lumi=lumi, 
                          format=format, suffix='_tot', 
                          yMin=-30, yMax=160)
