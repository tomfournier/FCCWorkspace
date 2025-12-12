import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import numpy as np
import pandas as pd

from typing import Union

from ..config import h_colors, h_labels, vars_label, vars_xlabel

from .root import plotter
from .root.plotter import finalize_canvas

from .root.helper import make_cfg, configure_axis, setup_latex, mk_legend
from .root.helper import draw_latex, load_hists, style_hist
from .root.helper import build_cfg, save_plot, savecanvas

from ..tools.utils import mkdir, get_range, get_range_decay
from ..tools.process import getHist
from ..func.bias import make_pseudosignal

PLT_STYLE_SET = False

def _ensure_plt_style() -> None:
    global PLT_STYLE_SET
    if not PLT_STYLE_SET:
        from .python.plotter import set_plt_style
        set_plt_style()
        PLT_STYLE_SET = True

#____________________________________________________
def get_args(var: str, 
             sel: str,
             args: dict[str, 
                        dict[str, 
                             Union[str, float, int]]]
             ) -> dict[str, 
                       Union[str, float, int]]:
    
    arg = args[var].copy() if var in args else {}
    if 'which' in arg:
        if arg['which']=='both':
            del arg['which']
        elif arg['which']=='make':
            del arg['which']
        elif arg['which']=='decay':
            arg = {}
        else:
            print("WARNING: Wrong value given to 'which', " 
                  "acting as if 'both' were given")
        
            
    if 'sel' in arg:
        if '*' in arg['sel']:
            if arg['sel'].replace('*', '') not in sel:
                arg = {}
            else:
                del arg['sel']
        elif arg['sel']!=sel:
            arg = {}
        else: 
            del arg['sel']
    
    for key in ['xmin', 'xmax', 'ymin', 'ymax']:
        arg.setdefault(key, None)
    
    arg.setdefault('rebin',     1)
    
    arg.setdefault('suffix',  '')
    arg.setdefault('outName', '')
    arg.setdefault('format', ['png'])

    arg.setdefault('strict', True)
    arg.setdefault('logX',   False)

    arg.setdefault('stack',  False)
    arg.setdefault('sig_scale', 1.)

    return arg

#______________________________________________________
def args_decay(var: str, 
               sel: str,
               args: dict[str, 
                          dict[str, 
                               Union[str, float, int]]]
               ) -> dict[str, 
                         Union[str, float, int]]:
    
    arg = args[var].copy() if var in args else {}
    if 'which' in arg:
        if arg['which']=='both':
            del arg['which']
        elif arg['which']=='make':
            arg = {}
        elif arg['which']=='decay':
            del arg['which']
        else:
            print("WARNING: Wrong value given to 'which', " 
                  "acting as if 'both' were given")

    if 'sel' in arg:
        if '*' in arg['sel']:
            if arg['sel'].replace('*', '') not in sel:
                arg = {}
            else:
                del arg['sel']
        elif arg['sel']!=sel:
            arg = {}
        else: 
            del arg['sel']
    
    for key in ['xmin', 'xmax', 'ymin', 'ymax']:
        arg.setdefault(key, None)
    
    arg.setdefault('rebin',     1)
    
    arg.setdefault('suffix',  '')
    arg.setdefault('outName', '')
    arg.setdefault('format', ['png'])

    arg.setdefault('strict', True)
    arg.setdefault('logX',   False)

    return arg

#_____________________________________________________
def significance(variable: str, 
                 inDir: str, 
                 outDir: str, 
                 sel: str,
                 procs: list[str], 
                 processes: dict[str, 
                                 list[str]], 
                 locx: str = 'right', 
                 locy: str = 'top',
                 xMin: Union[float, int, None] = None,
                 xMax: Union[float, int, None] = None, 
                 outName: str = '', 
                 suffix: str = '', 
                 format: list[str] = ['png'], 
                 reverse: bool = False, 
                 lazy: bool = True, 
                 rebin: int = 1) -> None:
    
    from .python.plotter import set_labels, savefigs
    from matplotlib.pyplot import subplots, close

    _ensure_plt_style()

    if outName=='': outName = variable
    suff  = f'_{sel}_histo'

    h_sig = getHist(variable, 
                    processes[procs[0]], inDir, 
                    suffix=suff, rebin=rebin)
    sig_tot = h_sig.Integral()

    bkgs_procs = []
    for bkg in procs[1:]: 
        bkgs_procs.extend(processes[bkg])

    h_bkg = getHist(variable, bkgs_procs, inDir, 
                    suffix=suff, rebin=rebin, lazy=lazy)

    nbins = h_sig.GetNbinsX()

    sig_arr = np.array([h_sig.GetBinContent(i) for i in range(1, nbins+1)])
    bkg_arr = np.array([h_bkg.GetBinContent(i) for i in range(1, nbins+1)])
    centers = np.array([h_sig.GetBinCenter(i)  for i in range(1, nbins+1)])
    
    mask = np.ones(nbins, dtype=bool)
    if xMin is not None:
        mask &= (centers >= xMin)
    if xMax is not None:
        mask &= (centers <= xMax)
    
    if reverse:
        sig_cum = np.cumsum(sig_arr)
        bkg_cum = np.cumsum(bkg_arr)
    else:
        sig_cum = np.cumsum(sig_arr[::-1])[::-1]
        bkg_cum = np.cumsum(bkg_arr[::-1])[::-1]
    
    denom = sig_cum + bkg_cum
    with np.errstate(divide='ignore', invalid='ignore'):
        significance = np.where(denom > 0, sig_cum / np.sqrt(denom), 0)
    sig_loss = np.where(sig_tot > 0, sig_cum / sig_tot, 0)
    
    x, y = centers[mask], significance[mask]
    l = sig_loss[mask]

    max_index = int(np.argmax(y))
    max_y = float(y[max_index])
    max_x, max_l = float(x[max_index]), float(l[max_index])

    fig, ax1 = subplots()

    ax2 = ax1.twinx()
    ax2.plot(x, l, 
             color='red', 
             linewidth=3, 
             label='Signal efficiency')
    ax1.scatter(x, y, 
                color='blue', 
                marker='o', 
                label='Significance')
    ax1.scatter(max_x, max_y, 
                color='red', 
                marker='*', 
                s=150)
    
    ax1.axvline(max_x, 
                color='black', 
                alpha=0.8, 
                linewidth=1)
    ax1.axhline(max_y, 
                color='blue', 
                alpha=0.8, 
                linewidth=1)
    ax2.axhline(max_l, 
                color='red', 
                alpha=0.8, 
                linewidth=1)

    ax1.set_xlim(min(x), max(x))
    if variable=='H':
        GeV = ' GeV$^{2}$'
    elif 'GeV' in vars_xlabel[variable]:
        GeV = ' GeV'
    else:
        GeV = ''

    set_labels(ax1, vars_xlabel[variable], 'Significance', left=' ', locx=locx, locy=locy)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.yaxis.label.set_color('blue')

    set_labels(ax2, ylabel='Signal Efficiency', left=' ', locy=locy)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.yaxis.label.set_color('red')
    ax2.grid(False, axis='y')

    if reverse:
        ax1.set_title(rf'Max: {vars_label[variable]} $<$ {max_x:.2f}{GeV}, '
                      rf'Significance = {max_y:.2f}, '
                      rf'Signal eff = {max_l*100:.1f} \%')
    else:
        ax1.set_title(rf'Max: {vars_label[variable]} $>$ {max_x:.2f}{GeV}, '
                      rf'Significance = {max_y:.2f}, '
                      rf'Signal eff = {max_l*100:.1f} \%')
    fig.tight_layout()

    out = f'{outDir}/significance/{sel}'
    mkdir(out)

    suffix = '_reverse' if reverse else ''
    savefigs(fig, out, outName, 
             suffix=suffix, 
             format=format)
    close()

#_________________________________________________
def makePlot(variable: str, 
             inDir: str, 
             outDir: str, 
             sel: str, 
             procs: list[str], 
             processes: dict[str, 
                             list[str]], 
             colors: dict[str, 
                          str], 
             legend: dict[str, 
                          str], 
             ecm: int = 240, 
             lumi: float = 10.8, 
             suffix: str = '', 
             outName: str = '', 
             format: list[str] = ['png'],
             xmin: Union[float, int, None] = None, 
             xmax: Union[float, int, None] = None,
             ymin: Union[float, int, None] = None, 
             ymax: Union[float, int, None] = None,
             rebin: int = 1, 
             sig_scale: float = 1., 
             strict: bool = True,
             logX: bool = False, 
             logY: bool = True, 
             stack: bool = False, 
             lazy: bool = True) -> None:

    if outName == '': outName = variable
    suff = f'_{sel}_histo'

    Processes = {k:v for k, v in processes.items() if k in procs}

    leg = mk_legend(len(procs))
    raw_hists = load_hists(Processes, 
                          variable, 
                          inDir, 
                          suffix=suff, 
                          rebin=rebin, 
                          lazy=lazy)
    all_hists = {proc: hist.Clone(f'{proc}_{variable}') 
                for proc, hist in raw_hists.items() if hist}
    
    sig_key = procs[0]
    sig_hist = all_hists.get(sig_key)
    
    st, bkgs = ROOT.THStack(), []
    st.SetName('stack')
    for proc, hist in all_hists.items():
        is_sig = proc==sig_key
        scale = f' (#times {int(sig_scale)})' if is_sig and sig_scale!=1 else ''
        style_hist(hist, 
                   color=colors[proc] if is_sig else ROOT.kBlack, 
                   width=3 if is_sig else 1, 
                   fill_color=colors[proc] if not is_sig else None,
                   scale = sig_scale if is_sig else 1.)
        leg.AddEntry(hist, legend[proc]+scale, 'L' if is_sig else 'F')

        if not is_sig: 
            st.Add(hist)
            bkgs.append(hist)

    cfg = build_cfg(sig_hist, 
                    logX=logX, logY=logY, 
                    xmin=xmin, xmax=xmax, 
                    ymin=ymin, ymax=ymax, 
                    ecm=ecm, lumi=lumi,
                    strict=strict, 
                    stack=stack, 
                    hists=bkgs)

    plotter.cfg = cfg
    canvas, dummy = plotter.canvas(), plotter.dummy()
    dummy.Draw('HIST')
    if stack:
        st.Add(sig_hist)
        st.Draw('HIST SAME')
    else: 
        if bkgs:
            st.Draw('HIST SAME')
        sig_hist.Draw('HIST SAME')
    leg.Draw('SAME')

    finalize_canvas(canvas)
    out = f'{outDir}/makePlot/{sel}'
    linlog = '_log' if logY else '_lin'
    save_plot(canvas, out, outName, linlog+suffix, format)
    canvas.Close()


#___________________________________________________
def PlotDecays(variable: str, 
               inDir: str, 
               outDir: str, 
               sel: str, 
               z_decays: list[str], 
               h_decays: list[str],
               ecm: int = 240, 
               lumi: float = 10.8, 
               rebin: int = 1, 
               outName: str = '', 
               suffix: str = '',
               format: list[str] = ['png'], 
               xmin: Union[float, int, None] = None, 
               xmax: Union[float, int, None] = None,
               ymin: Union[float, int, None] = None, 
               ymax: Union[float, int, None] = None,
               logX: bool = False, 
               logY: bool = False,
               lazy: bool = True, 
               strict: bool = True,
               tot: bool = False) -> None:

    if outName == '': outName = variable
    suff = f'_{sel}_histo'

    sigs = {h: [f'wzp6_ee_{z}H_H{h}_ecm{ecm}' for z in z_decays] for h in h_decays}

    raw_hists = load_hists(sigs, variable, inDir, suff, rebin=rebin, lazy=lazy)
    hists = {h_decay: hist.Clone(f'{h_decay}_{variable}') 
             for h_decay, hist in raw_hists.items() if hist}
    leg = mk_legend(len(sigs), 
                    columns=4, 
                    x1=0.2,  y1=0.925, 
                    x2=0.95, y2=0.925)

    for h_decay, hist in hists.items():
        integral = hist.Integral()
        norm = 1.0 / integral if integral>0 else 1.0
        style_hist(hist, 
                   h_colors[h_decay],
                   width=2, 
                   scale=norm)
        leg.AddEntry(hist, h_labels[h_decay], 'L')

    ref_hist = next(iter(hists.values()))
    cfg = build_cfg(ref_hist, 
                    logX=logX, logY=logY, 
                    xmin=xmin, xmax=xmax, 
                    ymin=ymin, ymax=ymax,
                    ecm=ecm, lumi=lumi,
                    strict=strict, 
                    hists=list(hists.values()), 
                    range_func=get_range_decay,
                    decay=True)

    plotter.cfg = cfg
    canvas, dummy = plotter.canvas(), plotter.dummy(1)
    dummy.Draw('HIST') 

    for hist in hists.values(): hist.Draw('SAME HIST')
    leg.Draw('SAME')
    
    out = f'{outDir}/higgsDecays/{sel}/tot' if tot \
        else f'{outDir}/higgsDecays/{sel}/cat'
    linlog = '_log' if logY else '_lin'
    finalize_canvas(canvas)
    save_plot(canvas, out, outName, linlog+suffix, format)
    canvas.Close()


#________________________________________
def AAAyields(hName: str, 
              inDir: str, 
              outDir: str, 
              plots: dict[str, 
                          list[str]], 
              legend: dict[str, 
                           str], 
              colors: dict[str, 
                           str], 
              cat: str, sel: str, 
              ecm: int = 240, 
              lumi: float = 10.8, 
              scale_sig: float = 1., 
              scale_bkg: float = 1., 
              lazy: bool = True,
              outName: str = '', 
              format: list[str] = ['png']
              ) -> None:
    
    if outName=='': outName = 'AAAyields'
    if   cat == 'mumu': 
        ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
    elif cat == 'ee':   
        ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
    else:
        raise ValueError(f'{cat} value is not supported')
    suffix = f'_{sel}_histo'

    signal      = plots['signal']
    backgrounds = plots['backgrounds']

    leg = mk_legend(len(signal)+len(backgrounds), 
                    x1=0.6, y1=0.86, x2=0.9, y2=0.88,
                    text_font=42)

    yields = {}

    for b in backgrounds:
        hist = getHist(hName,
                       backgrounds[b], inDir,
                       suffix=suffix, lazy=lazy)
        integral = hist.Integral()
        entries  = hist.GetEntries()

        style_hist(hist, 
                   color=ROOT.kBlack,
                   fill_color=colors[b],
                   scale=scale_bkg)
        leg.AddEntry(hist, legend[b], 'F')

        yields[b] = [legend[b], 
                     integral * scale_bkg, 
                     entries]

    for s in signal:
        hist = getHist(hName, 
                       signal[s], inDir, 
                       suffix=suffix, lazy=lazy)
        integral = hist.Integral()
        entries  = hist.GetEntries()

        style_hist(hist, 
                   color=colors[s],
                   scale=scale_sig,
                   width=4)
        leg.AddEntry(hist, legend[s], 'L')
        yields[s] = [legend[s], 
                     integral * scale_sig, 
                     entries]

    canvas = plotter.canvas(top=None, 
                            bottom=None, 
                            left=0.14, 
                            right=0.08, 
                            batch=True,
                            yields=True)

    dummyh = ROOT.TH1F('', '', 1, 0, 1)
    dummyh.SetStats(0)
    configure_axis(dummyh.GetXaxis(), '', 0, 1, label_offset=999, label_size=0)
    configure_axis(dummyh.GetYaxis(), '', 0, 1, label_offset=999, label_size=0)
    dummyh.Draw('AH')
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(31)

    simu = [('#bf{FCC-ee} #scale[0.7]{#it{Simulation}}', 0.9, 0.92, 0.04)]
    draw_latex(latex, simu)


    latex.SetTextAlign(12)
    latex.SetNDC(ROOT.kTRUE)
    sqrt = [('#bf{#it{#sqrt{s} = '+f'{ecm}'+' GeV}}', 0.18, 0.83, 0.04)]
    draw_latex(latex, sqrt)

    s_tot = int( sum( [yields[s][1] for s in signal] ) )
    b_tot = int( sum( [yields[b][1] for b in backgrounds] ) )
    text_data = [
        ('#bf{#it{L = '+f'{lumi}'+' ab^{#minus1}}}', 0.18, 0.78, 0.035),
        ('#bf{#it{' + ana_tex + '}}', 0.18, 0.73, 0.04),
        ('#bf{#it{' + sel + '}}', 0.18, 0.68, 0.025),
        ('#bf{#it{Signal Scaling = ' + f'{scale_sig:.3g}' + \
            '}}', 0.18, 0.62, 0.04),
        ('#bf{#it{Background Scaling = ' + \
        f'{scale_bkg:.3g}' + '}}', 0.18, 0.57, 0.04),
        ('#bf{#it{Significance = ' +
         f'{s_tot/(s_tot+b_tot)**(0.5):.3f}' + '}}', 0.18, 0.52, 0.04),
        ('#bf{#it{Process}}', 0.18, 0.45, 0.035),
        ('#bf{#it{Yields}}', 0.5, 0.45, 0.035),
        ('#bf{#it{Raw MC}}', 0.75, 0.45, 0.035),
    ]
    draw_latex(latex, text_data)

    latex_yield = []
    for dy, (label, integral, entries) in enumerate(yields.values()):
        latex_yield.extend([
            ('#bf{#it{' + label + '}}', 0.18, 0.4-dy*0.05, 0.035),
            ('#bf{#it{' + str(int(integral)) + '}}', 0.5, 0.4-dy*0.05, 0.035),
            ('#bf{#it{' + str(int(entries)) + '}}', 0.75, 0.4-dy*0.05, 0.035)
        ])
    draw_latex(latex, latex_yield)

    out = f'{outDir}/yield/{sel}'
    mkdir(out)
    savecanvas(canvas, out, outName, format=format)
    canvas.Close()

#___________________________________
def Bias(df: pd.DataFrame, 
         nomDir: str, 
         outDir: str, 
         h_decays: list[str], 
         suffix: str = '', 
         ecm: int = 240, 
         lumi: float = 10.8, 
         outName: str = 'bias', 
         format: list[str] = ['png']
         ) -> None:

    bias_dict = dict(zip(df['mode'], df['bias'] * 100))
    unc = float(np.loadtxt(f'{nomDir}/results.txt')[-1] * 1e4)

    bias_values = list(bias_dict.values())
    max_bias = max(abs(min(bias_values)), abs(max(bias_values)))
    xMin, xMax = (-int(unc*1.2), int(unc*1.2)) if unc > max_bias \
        else (-int(max_bias*1.2), int(max_bias*1.2))

    In = sum(1 for b in bias_values if abs(b) < unc)
    Out = len(bias_values) - In

    h_pulls = ROOT.TH2F('pulls', 'pulls', 
                        (xMax-xMin)*10, 
                        xMin, xMax, 
                        len(h_decays), 
                        0, len(h_decays))
    g_in, g_out = ROOT.TGraphErrors(In), ROOT.TGraphErrors(Out)

    i, j = 0, 0
    for k, h_decay in enumerate(h_decays):
        b = bias_dict[h_decay]
        if np.abs(b) < unc: 
            g_in.SetPoint(i, b, float(k) + 0.5)
            h_pulls.GetYaxis().SetBinLabel(k+1, h_labels[h_decay])
            i += 1
        else:
            g_out.SetPoint(j, b, float(k) + 0.5)
            h_pulls.GetYaxis().SetBinLabel(k+1, h_labels[h_decay])
            j += 1

    cfg = build_cfg(h_pulls, 
                    xmin=xMin, xmax=xMax, 
                    ymin=0, ymax=len(h_decays),
                    ytitle='None',
                    ecm=ecm, lumi=lumi,
                    strict=False, 
                    hists=None,
                    cutflow=True)

    plotter.cfg = cfg
    canvas = plotter.canvas(top=0.08, bottom=0.1)
    canvas.SetGrid()

    xTitle = 'Bias (#times 100) [%]'

    h_pulls.GetXaxis().SetTitleSize(0.04), h_pulls.GetXaxis().SetLabelSize(0.035)
    h_pulls.GetXaxis().SetTitle(xTitle), h_pulls.GetXaxis().SetTitleOffset(1)
    h_pulls.GetYaxis().SetLabelSize(0.055), h_pulls.GetYaxis().SetTickLength(0)
    h_pulls.GetYaxis().LabelsOption('v'), h_pulls.SetNdivisions(506, 'XYZ')
    h_pulls.Draw('HIST 0')

    maxx, lines = len(h_decays), []
    for l, c in zip([-1, 0, 1], [ROOT.kBlack, ROOT.kGray, ROOT.kBlack]):
        line = ROOT.TLine(unc*l, 0, unc*l, maxx)
        line.SetLineColor(c)
        line.SetLineWidth(3)
        line.Draw('SAME')
        lines.append(line)

    g_in.SetMarkerSize(1.2), g_in.SetMarkerStyle(20)
    g_in.SetLineWidth(2), g_in.SetMarkerColor(ROOT.kBlack)
    g_in.Draw('P0 SAME')

    g_out.SetMarkerSize(1.2), g_out.SetMarkerStyle(20)
    g_out.SetLineWidth(2), g_out.SetMarkerColor(ROOT.kRed)
    g_out.Draw('P0 SAME')


    latex = setup_latex(text_size=0.045, text_align=30, 
                        text_color=1, text_font=42)
    latex.DrawLatex(0.95, 0.925, f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}')

    latex = setup_latex(text_size=0.045, text_align=13, text_font=42)
    latex.DrawLatex(0.15, 0.96, '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}')

    mkdir(outDir)
    savecanvas(canvas, outDir, outName, suffix, format)
    canvas.Close()

#_________________________________________
def hist_to_arrays(hist: ROOT.TH1
                   ) -> tuple[np.ndarray, 
                              np.ndarray, 
                              np.ndarray]:

    nbins = hist.GetNbinsX()
    bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, nbins+2)])
    counts = np.array([hist.GetBinContent(i)    for i in range(1, nbins+1)])
    errors = np.array([hist.GetBinError(i)      for i in range(1, nbins+1)])
    
    return counts, bin_edges, errors

#_____________________________________________
def PseudoRatio(variable: str, 
                inDir: str, 
                outDir: str, 
                cat: str, 
                target: str, 
                z_decays: list[str], 
                h_decays: list[str], 
                ecm: int = 240, 
                lumi: float = 10.8, 
                pert: float = 1.05, 
                sel: str = '', 
                rebin: int = 1,
                outName: str = 'PseudoRatio', 
                format: list[str] = ['png'], 
                proc_scales: dict[str, 
                                  float] = {},
                logX: bool = False, 
                logY: bool = False, 
                lazy: bool = True, 
                tot: bool = True, 
                density: bool = False
                ) -> None:
    
    from matplotlib.pyplot import subplots, close
    import mplhep as hep
    hep.style.use('CMS')

    from .python.plotter import set_plt_style, set_labels, savefigs
    set_plt_style()

    import warnings
    warnings.filterwarnings('ignore', message='The value of the smallest subnormal for')


    if outName == '': outName = 'PseudoRatio'
    suffix = f'_{sel}_histo'
    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays]

    h_tot = None
    for sig in sigs:
        h_sig = getHist(variable, sig, inDir, 
                        suffix=suffix, rebin=rebin, lazy=lazy)
        h_sig.Rebin(rebin)
        if h_tot==None: h_tot = h_sig.Clone('h_tot')
        else: h_tot.Add(h_sig)

    hist_pseudo = make_pseudosignal(
        variable, inDir, target, cat, z_decays, h_decays, 
        suffix=suffix, variation=pert, tot=tot, proc_scales=proc_scales
    )
    hist_pseudo.Rebin(rebin)

    if density and h_tot.Integral()>0:
        norm = 1.0 / h_tot.Integral()
        hist_pseudo.Scale(norm)
        h_tot.Scale(norm)

    scale_min, scale_max = 5e-1 if logY else 1, 1e4 if logY else 1.5
    xMin, xMax, yMin, yMax = get_range([h_tot], [hist_pseudo], logY=logY,
                                       scale_min=scale_min, 
                                       scale_max=scale_max)

    sig_vals, bin, sig_err = hist_to_arrays(h_tot)
    psd_vals, _, psd_err = hist_to_arrays(hist_pseudo) 
    
    # Compute ratio with error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_vals = np.where(sig_vals > 0, psd_vals / sig_vals, 1.0)
        psd_rel_err = np.where(psd_vals > 0, psd_err / psd_vals, 0)
        sig_rel_err = np.where(sig_vals > 0, sig_err / sig_vals, 0)
        ratio_errs = np.where(sig_vals > 0, ratio_vals * np.sqrt(psd_rel_err**2 + sig_rel_err**2), 0)
    

    fig, [ax1, ax2] = subplots(2, 1, height_ratios=[4, 1])
    
    hep.histplot(sig_vals, bins=bin, ax=ax1,
                 histtype='step', linewidth=3, 
                 label='Signal', color='tab:blue')
    hep.histplot(psd_vals, bins=bin, yerr=psd_err, ax=ax1,
                 histtype='errorbar', linewidth=2, color='black',
                 marker='o', markersize=4, linestyle='',
                 label='Pseudo-signal')
    hep.histplot(ratio_vals, bins=bin, yerr=ratio_errs, ax=ax2,
                 histtype='band')
    hep.histplot(ratio_vals, bins=bin, ax=ax2,
                 histtype='errorbar', linewidth=2, color='black',
                 marker='o', markersize=4)
    
    ax2.axhline(1.0, color='gray', linestyle='-', linewidth=2, alpha=0.6)
    ax2.axhline(pert, color='black', linestyle='--', linewidth=2, alpha=0.8)

    ax1.set_xlim(xMin, xMax)
    ax2.set_xlim(xMin, xMax)
    ax1.set_ylim(yMin, yMax)
    ax2.set_ylim(0.95, pert * 1.05)

    ax1.legend(loc='upper right', frameon=True)
    ax1.tick_params(labelbottom=False)
    
    if logX: ax1.set_xscale('log')
    if logY: ax1.set_yscale('log')
    if logX: ax2.set_xscale('log')

    lumi_text = rf'$\sqrt{{s}} = {ecm}$ GeV, {lumi} ab$^{{-1}}$'
    xtitle = r'm$_{\mathrm{recoil}}$ High [GeV]' if 'high' in sel \
        else r'm$_{\mathrm{recoil}}$ Low [GeV]'
    ytitle = 'Normalized to Unity' if density else 'Events'
    set_labels(ax2, xtitle, 'Ratio', left='None')
    set_labels(ax1, '', ytitle, right=lumi_text)
    
    out = f'{outDir}/high' if 'high' in sel else f'{outDir}/low'
    mkdir(out)
    savefigs(fig, out, outName, suffix=f'_{target}', format=format)
    close(fig)
