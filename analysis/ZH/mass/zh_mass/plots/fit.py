################################
### STANDARD LIBRARY IMPORTS ###
################################

import array, ROOT



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from path import PathObj
from plot import plotter
from plot.plotter import draw_latex, setup_latex, save_canvas
from plot.graphplot import GraphPlot
from plot.pullplot import PullPlot
from ..userConfig import param_label



########################
### HELPER FUNCTIONS ###
########################

def load_nll_curve(
        file_path: str | PathObj,
        only_unc: bool = True
) -> tuple[float, GraphPlot] | tuple[float, float, list[float], list[float], GraphPlot]:

    xv, yv = [], []
    if isinstance(file_path, str): file_path = PathObj(file_path)
    with open(file_path, 'r') as fIn:
        for i, line in enumerate(fIn.readlines()):
            line = line.rstrip()
            if i == 0:
                best_mass = float(line.split(' ')[3])
                unc_mass  = float(line.split(' ')[2])
            else:
                xv.append(float(line.split(' ')[0]))
                yv.append(float(line.split(' ')[1]))

    graph = GraphPlot.from_xy(xv, yv)
    if only_unc:
        return unc_mass, graph
    return best_mass, unc_mass, xv, yv, graph



######################
### PLOT FUNCTIONS ###
######################

def plot_mass_scan(
        outputdir,
        graph,
        label: str,
        err: float,
        outName: str = 'mass',
        suffix: str = '',
        ecm: int = 240,
        lumi: float = 10.8,
        color = ROOT.kRed,
        width: int = 2,
        marker: int = 20,
        file_formats: list[str] = ['png']
) -> None:

    plot = GraphPlot(outputdir, None, None, ecm, lumi, file_formats)
    plot.add(graph, f'{label}, #Delta(m_{{H}}) = {err:.2f} MeV', color, width, marker, draw_option='LP')
    plot.build_config(graph, xtitle='m_{H} [GeV]', ytitle='-2#DeltaNLL')
    canvas = plot.draw(outName, suffix, [1.0, 4.0])
    canvas.Close()

    return None


def plot_mass_breakdown_impacts(
        outputdir: str,
        errs: list[float],
        labels: list[str],
        xtitle: str = '#sigma_{syst.}(m_{H}) (MeV)',
        xmin: float = -3,
        xmax: float = 3,
        ecm: int = 240,
        lumi: float = 10.8,
        outName: str = 'mass_breakdown_impacts',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:
    
    display_labels = [f'{label} ({err:.2g} MeV)' for label, err in zip(labels, errs)]
    plot = PullPlot(outputdir, '', None,
                    [0.] * len(errs), list(reversed(errs)),
                    list(reversed(display_labels)), ecm=ecm, lumi=lumi, xtitle=xtitle)
    plot.draw(outName, suffix, file_formats, False, xmin, xmax)

    return None


def plot_mass_breakdown_curves(
        outputdir: str,
        curves: list[tuple[list[float], list[float], float, str, int, int]],
        labels: list[str],
        topRight: str,
        topLeft: str,
        xmin: float = 124.995,
        xmax: float = 125.005,
        ymin: float = 0,
        ymax: float = 2,
        ecm: int = 240,
        lumi: float = 10.8,
        outName: str = 'mass_breakdown',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    plot = GraphPlot(outputdir, None, None, ecm, lumi, file_formats)
    for (x_vals, y_vals, err, color, width, marker), label in zip(curves, labels):
        graph = GraphPlot.from_xy(x_vals, y_vals)
        plot.add(graph, f'{label} #delta(m_{{H}}) = {err:.2f} MeV', color, width, marker, draw_option='L')
    plot.build_config(None, xmin, xmax, ymin, ymax, 'm_{h} (GeV)', '-2#DeltaNLL', topLeft, topRight)
    canvas = plot.draw(outName, suffix, [1.0, 4.0])
    canvas.Close()

    return None


def plot_mass_multiple(
        inputdirs: list[str],
        labels: list[str],
        outputdir,
        suffix: str,
        topRight: str,
        xmin: float,
        xmax: float,
        ecm: int = 240,
        lumi: float = 10.8,
        leg_label: str = '',
        file_formats: list[str] = ['png'],
        force_stat: list[bool] | None = None
) -> None:

    if force_stat is None:
        force_stat = [False] * len(inputdirs)

    graphs, errs = [], []
    for i, inputdir in enumerate(inputdirs):
        fName = f'mass{suffix + "_stat" if force_stat[i] else suffix}.txt'
        unc_mass, graph = load_nll_curve(f'{inputdir}/{fName}')
        errs.append(unc_mass)
        graphs.append(graph)

    plot = GraphPlot(outputdir, None, None, ecm, lumi, file_formats)
    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 1]
    for index, graph in enumerate(graphs):
        plot.add(graph, f'{labels[index]} #delta(m_{{H}}) = {errs[index]*1000:.2f} MeV',
                 colors[index], draw_option='L')
    plot.build_config(graphs[0], xmin, xmax, xtitle='m_{H} [GeV]', ytitle='-2#DeltaNLL', topRight=topRight)
    canvas = plot.draw('', suffix, [1.0, 4.0], leg_label)
    canvas.Close()

    return None


def plot_xsec_multiple(
        tags: list[str],
        labels: list[str],
        outputdir: str,
        topRight: str,
        xmin: float,
        xmax: float,
        ecm: int = 240,
        lumi: float = 10.8,
        outName: str = '',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    graphs, errs = [], []
    for tag in tags:
        unc_xsec, graph = load_nll_curve(f'{tag}/xsec.txt')
        errs.append(unc_xsec)
        graphs.append(graph)

    plot = GraphPlot(outputdir, None, None, ecm, lumi, file_formats)
    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 1]
    for idx, graph in enumerate(graphs):
        plot.add(graph, f'{labels[idx]} #delta(#sigma) = {errs[idx]*100:.2f}', colors[idx], draw_option='L')
    plot.build_config(graphs[0], xmin, xmax, xtitle='#sigma(ZH#rightarrowl^{#plus}l^{#minus})/#sigma_{ref}',
                      ytitle='-2#DeltaNLL', topRight=topRight,)
    canvas = plot.draw(outName, suffix, [1.0, 4.0])
    canvas.Close()

    return None


def plot_spline_scan(
        outputdir: str,
        MH,
        x_vals,
        y_vals,
        spline,
        outName: str,
        ytitle: str,
        label: str = '',
        ecm: int = 240,
        lumi: float = 10.8,
        marker_color=ROOT.kBlack,
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    graph = ROOT.TGraphErrors(len(x_vals), array.array('d', x_vals), array.array('d', y_vals),
                              array.array('d', [0] * len(x_vals)),
                              array.array('d', [0] * len(x_vals)))
    plot = GraphPlot(outputdir, graph, None, ecm, lumi, file_formats)
    plot.build_config(xmin=124.9, xmax=125.1, xtitle='m_{H} [GeV]', ytitle=ytitle)
    
    plotter.cfg = plot.cfg
    canvas, dummy = plotter.canvas(left=0.2), plotter.dummy()
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    frame = MH.frame()
    spline.plotOn(frame)
    GraphPlot.style_graph(graph, marker_color, 2, 8, 1.5)
    graph.Draw('SAME P')
    frame.Draw('SAME')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.25, 0.92, label)

    plotter.finalize_canvas(canvas)
    save_canvas(canvas, outputdir, outName, suffix, file_formats)

    return None


def plot_decomposition(
        outputdir,
        w_tmp,
        label: str,
        mH_label: str,
        yield_nom: float | int,
        model_spec: dict,
        file_formats: list[str] = ['png']
) -> None:

    mrec = w_tmp.var('zll_recoil_m')
    plotter.cfg = plotter.make_cfg({'xmin': 120, 'xmax': 140, 'ymin': 0, 'ymax': 1,
                                    'xtitle': 'm_{recoil} [GeV]', 'ytitle': 'Events'})
    canvas, dummy = plotter.canvas(), plotter.dummy()
    dummy.Draw('HIST')

    plt = mrec.frame()
    sig_fit =  w_tmp.pdf(f"{model_spec['model_name']}_{mH_label}")
    fracs   = [w_tmp.obj(f"{frac}_{mH_label}").getVal() for frac in model_spec['fractions']]
    comps   = [w_tmp.obj(f"{comp['name']}_{mH_label}")  for comp in model_spec['components']]

    comp_data = [('Total PDF', ROOT.kBlack, sig_fit, yield_nom)]
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kCyan, ROOT.kGreen, ROOT.kMagenta, ROOT.kOrange, ROOT.kViolet]
    for idx, (comp_spec, comp) in enumerate(zip(model_spec['components'], comps)):
        if idx < len(fracs): comp_yield = yield_nom * fracs[idx]
        else:                    comp_yield = yield_nom * (1 - sum(fracs))
        comp_data.append((comp_spec['title'], colors[idx % len(colors)], comp, comp_yield))

    for _, color, comp, comp_yield in comp_data[1:]:
        comp.plotOn(plt, ROOT.RooFit.LineColor(color),       ROOT.RooFit.Normalization(comp_yield, ROOT.RooAbsReal.NumEvent))
    sig_fit.plotOn(plt,  ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.Normalization(yield_nom,  ROOT.RooAbsReal.NumEvent))

    leg = plotter.define_legend(0, 1, 0.5, 0.7, 0.95, 0.9,
                                0, 0, 0.04, 0.15)

    for name, color, _, _ in comp_data:
        tmp = ROOT.TGraph()
        tmp.SetPoint(0, 0, 0)
        tmp.SetLineColor(color)
        tmp.SetLineWidth(3)
        tmp.Draw('SAME')
        leg.AddEntry(tmp, name, 'L')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    plt.Draw('SAME')
    leg.Draw()

    plotter.finalize_canvas(canvas)
    save_canvas(canvas, outputdir, 'fit_mH', f'{mH_label}_decomposition', file_formats)

    return sig_fit


def plot_fit(
        outputdir: str,
        mrec,
        rdh_zh,
        sig_fit,
        label: str,
        nBins: int = 250,
        ecm: int = 240,
        lumi: float = 10.8,
        outName: str = 'fit_mH',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    plotter.cfg = plotter.make_cfg({
        'xmin':   120, 'xmax':  140, 'ymin': 0, 'ymax': 1, 'xtitle': 'm_{recoil} [GeV]', 'ytitle': 'Events',
        'yminR': -3.5, 'ymaxR': 3.5, 'ytitleR': 'Pull'}, ecm, lumi, True)

    canvas, padT, padB    = plotter.canvasRatio()
    dummyT, dummyB, lines = plotter.dummyRatio(1, [0])

    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetGrid()
    dummyT.Draw('HIST')

    plt = mrec.frame(ROOT.RooFit.Title('ZH signal'))
    rdh_zh.plotOn(plt, ROOT.RooFit.Binning(nBins))
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    sig_fit.paramOn(plt, ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(0.45, 0.9, 0.9))
    histpull = plt.pullHist()
    plt.Draw('SAME')

    latex = setup_latex(0.045, 13, 1, 42)
    draw_latex(latex, [(label, 0.2, 0.88, 0.045), f'#chi^2 = {plt.chiSquare():.3f}', 0.2, 0.82, 0.045])

    plotter.finalize_canvas(canvas)

    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()

    dummyB.Draw('HIST')
    for line in lines: line.Draw('SAME')
    plt.addPlotable(histpull, 'P')
    plt.Draw('SAME')

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    save_canvas(canvas, outputdir, outName, suffix, file_formats)

    # Explicitly delete objects to free memory faster
    canvas.Close()
    del canvas, dummyB, dummyT, padT, padB

    return None



def plot_fit_all(
        outDir,
        w_tmp,
        mHs,
        yield_zh,
        label,
        outName: str = 'fil_all',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    mrec = w_tmp.var('zll_recoil_m')
    plotter.cfg = plotter.make_cfg({
        'xmin': 124, 'xmax': 130, 'ymin': 0, 'ymax': 1,
        'xtitle': 'm_{recoil} [GeV]', 'ytitle': 'Events',
        'yminR': -3.5, 'ymaxR': 3.5, 'ytitleR': 'Pull',
    })

    canvas, dummy = plotter.canvas(), plotter.dummy()
    dummy.Draw('HIST')

    plt = mrec.frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, mH in enumerate(mHs):
        sig_fit = w_tmp.pdf('zh_model_'+f'{mH:.3f}'.replace('.', 'p'))
        # Need to re-normalize the pdf, as the pdf is normalized to 1
        sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[i]), ROOT.RooFit.Normalization(yield_zh, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.finalize_canvas(canvas)
    save_canvas(canvas, outDir, outName, suffix, file_formats)

    return None


def plot_params_vs_mh(
        MH: 'ROOT.RooRealVar',
        outDir: PathObj,
        param: str,
        vals: dict[str, float | int],
        spline: 'ROOT.RooSpline1D',
        label: str = '',
        outName: str = '',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    mHs = vals['mH']

    graph = ROOT.TGraphErrors(len(mHs), array.array('d', mHs), array.array('d', vals[param]),
                              array.array('d', [0]*len(mHs)),
                              array.array('d', [0]*len(mHs)))

    plot = GraphPlot(outDir, graph=graph, file_formats=file_formats)
    plot.build_config(xmin=124.9, xmax=125.1, xtitle='m_{H} [GeV]', ytitle=param_label.get(param, param))

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.cfg = plot.cfg
    canvas, dummy = plotter.canvas(left=0.2), plotter.dummy()
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    plt = MH.frame()
    spline.plotOn(plt)
    GraphPlot.style_graph(graph, ROOT.kBlack, 2, 8, 1.5)
    graph.Draw('SAME P')
    plt.Draw('SAME')

    plotter.finalize_canvas(canvas)
    outName = outName or f'fit_{param}'
    save_canvas(canvas, outDir, outName, suffix, file_formats)

    return None


def fit_plot(
    rdh_zh: 'ROOT.RooDataHist',
    pdf: 'ROOT.RooAddPdf',
    recoilmass: 'ROOT.RooRealVar',
    nBins: int,
    label: int,
    outputdir: PathObj,
    outName: str = 'fit_all',
    suffix: str = '',
    file_formats: list[str] = ['png']
) -> None:

    canvas, padT, padB     = plotter.canvasRatio()
    dummyT, dummyB, lines  = plotter.dummyRatio(1, [0], [ROOT.kBlue+2])

    canvas.cd()
    padT.Draw()
    padT.SetGrid()
    padT.cd()
    dummyT.Draw('HIST')

    plt = recoilmass.frame()
    plt.SetTitle('ZH signal')
    rdh_zh.plotOn(plt, ROOT.RooFit.Binning(nBins))  # ROOT.RooFit.Normalization(yield_zh, ROOT.RooAbsReal.NumEvent)

    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    pdf.paramOn(plt, ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(0.45, 0.9, 0.9))

    histpull = plt.pullHist()
    plt.Draw('SAME')

    latex = setup_latex(0.045, 13, 1, 42)
    draw_latex(latex, [(label, 0.2, 0.88, 0.045),
                       (f'#chi^2 = {plt.chiSquare():.3f}', 0.2, 0.82, 0.045)])

    plotter.finalize_canvas(canvas)

    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.GetXaxis().SetTitleOffset(4.0*dummyB.GetXaxis().GetTitleOffset())
    dummyB.Draw('HIST')

    plt = recoilmass.frame()
    plt.addPlotable(histpull, 'P')
    plt.Draw('SAME')
    for line in lines: line.Draw('SAME')

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    save_canvas(canvas, outputdir, outName, suffix, file_formats)

    del canvas, dummyT, dummyB, padT, padB

    return None


def plot_syst_dist(
        w_tmp: ROOT.RooWorkspace,
        yield_nom: int,
        outputdir: str,
        syst: str,
        mH_label: str,
        outName: str = 'fit_mH',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')

    plt    = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]

    for i, channel in enumerate([f'_{syst}Up', '', f'_{syst}Down']):
        sig_fit = w_tmp.pdf(f'zh_model_{mH_label}{channel}')
        sig_fit.plotOn(plt, ROOT.RooFit.Linecolor(colors[i]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')
    plotter.finalize_canvas(canvas)
    save_canvas(canvas, outputdir, outName, suffix + f'{mH_label}_{syst}', file_formats)

    return None


def plot_fit_with_pull(
    rdh: ROOT.RooDataHist,
    pdf: ROOT.RooAddPdf,
    recoilmass: ROOT.RooRealVar,
    n_bins: int,
    output_base: str,
    label_text: str,
    title: bool = None,
    fit_color: ROOT.TColor = ROOT.kRed,
    param_layout: tuple[float | int] = (0.25, 0.9, 0.9),
) -> None:

    canvas, padT, padB    = plotter.canvasRatio()
    dummyT, dummyB, lines = plotter.dummyRatio(1, [0])
    dummyB.GetXaxis().SetTitleOffset(4.0 * dummyB.GetXaxis().GetTitleOffset())

    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw('HIST')

    plt = recoilmass.frame()
    if title is not None:
        plt.SetTitle(title)
    rdh.plotOn(plt, ROOT.RooFit.Binning(n_bins))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(fit_color))

    if param_layout is not None:
        pdf.paramOn(plt, ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(*param_layout))

    histpull = plt.pullHist()
    plt.Draw('SAME')

    latex = setup_latex(0.045, 13, 1, 42)
    draw_latex(latex, [(label_text, 0.2, 0.88, 0.045),
                       f'#chi^2 = {plt.chiSquare():.3f}', 0.2, 0.82, 0.045])

    plotter.finalize_canvas(canvas)

    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()
    dummyB.Draw('HIST')
    for line in lines: line.Draw('SAME')

    plt = recoilmass.frame()
    plt.addPlotable(histpull, 'P')
    plt.Draw('SAME')

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs(f'{output_base}.png')

    del canvas, dummyB, dummyT, padT, padB

    return None


def plot_signal(
        workspace: ROOT.RooWorkspace,
        mHs: list[float | int],
        outDir: str,
        label: str,
        yield_nom: float | int,
        pdf_sigs: list[ROOT.RooAddPdf],
        outName: str = 'fil_all',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    canvas = plotter.canvas(leftMargin=0.2)
    dummy  = plotter.dummy()
    dummy.Draw('HIST')

    plt = workspace.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, _ in enumerate(mHs):

        sig_fit = pdf_sigs[i]
        # Need to re-normalize the pdf, as the pdf is normalized to 1
        sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[i]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.finalize_canvas(canvas)
    save_canvas(canvas, outDir, outName, suffix, file_formats)

    return None
