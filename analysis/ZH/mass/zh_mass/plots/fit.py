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
from plot.ratioplot import RatioPlot
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


def get_limits(
        obs: ROOT.RooRealVar,
        data_pdf: ROOT.RooDataHist | ROOT.RooAbsPdf | None = None,
        xMin: float | int | None = None,
        xMax: float | int | None = None,
        yMax: float | int | None = None,
        scale_max: float = 1.25,
        normalization: float | int | None = None
) -> tuple[float, float] |tuple[float, float, float]:

    # Get x-limits form observable
    xmin, xmax = obs.getMin(), obs.getMax()
    if xMin is not None and xmin < xMin: xmin = xMin
    if xMax is not None and xmax > xMax: xmax = xMax

    if data_pdf is None:
        return xmin, xmax

    hist = data_pdf.createHistogram(f'{data_pdf.GetName()}_plot', obs)

    # Assume a either a RooDataHist or a RooAbsPdf
    if data_pdf.ClassName() == 'RooDataHist':
        ymax = hist.GetMaximum() * scale_max
        if yMax is not None and ymax > yMax: ymax = yMax
        return xmin, xmax, ymax

    # Assume a RooAbsPdf
    if normalization is None:
        raise ValueError("Can't compute ymax from RooAbsPdf without normalization")

    integral = hist.Integral()
    if integral > 0:
        hist.Scale(normalization / integral)
    ymax = hist.GetMaximum() * scale_max
    if yMax is not None and ymax > yMax: ymax = yMax

    return xmin, xmax, ymax



######################
### PLOT FUNCTIONS ###
######################

def plot_mass_scan(
        outputdir: str,
        graph: ROOT.TGraph | ROOT.TGraphErrors,
        label: str,
        err: float,
        color = ROOT.kRed,
        width: int = 2,
        marker: int = 20,
        ecm: int = 240,
        lumi: float = 10.8,
        xtitle: str = 'm_{H} [GeV]',
        ytitle: str = '-2#DeltaNLL',
        outName: str = 'mass_scan',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    plot = GraphPlot(outputdir, None, None, ecm, lumi, file_formats)
    plot.add(graph, f'{label}, #Delta(m_{{H}}) = {err:.2f} MeV', color, width, marker, draw_option='LP')
    plot.build_config(graph, xtitle=xtitle, ytitle=ytitle)
    canvas = plot.draw(outName, suffix, [1.0, 4.0])
    canvas.Close()

    return None


def plot_mass_breakdown_impacts(
        outputdir: str,
        errs: list[float],
        labels: list[str],
        xmin: float = -3,
        xmax: float = 3,
        ecm: int = 240,
        lumi: float = 10.8,
        xtitle: str = '#sigma_{syst.}(m_{H}) (MeV)',
        outName: str = 'mass_breakdown_impact',
        topLeft: str | None = None,
        topRight: str | None = None,
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:
    
    display_labels = [f'{label} ({err:.2g} MeV)' for label, err in zip(labels, errs)]
    plot = PullPlot(outputdir, '', None,
                    [0.] * len(errs), list(reversed(errs)),
                    list(reversed(display_labels)), ecm=ecm, lumi=lumi, xtitle=xtitle)
    plot.draw(outName, suffix, file_formats, False, xmin, xmax, topLeft, topRight)

    return None


def plot_mass_breakdown_curves(
        outputdir: str,
        curves: list[tuple[list[float], list[float], float, str, int, int]],
        labels: list[str],
        xmin: float = 124.995,
        xmax: float = 125.005,
        ymin: float = 0,
        ymax: float = 2,
        ecm: int = 240,
        lumi: float = 10.8,
        xtitle: str = 'm_{H} [GeV]',
        ytitle: str = '-2#DeltaNLL',
        topLeft: str | None = None,
        topRight: str | None = None,
        outName: str = 'mass_breakdown',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    plot = GraphPlot(outputdir, None, None, ecm, lumi, file_formats)
    for (x_vals, y_vals, err, color, width, marker), label in zip(curves, labels):
        graph = GraphPlot.from_xy(x_vals, y_vals)
        plot.add(graph, f'{label} #delta(m_{{H}}) = {err:.2f} MeV', color, width, marker, draw_option='L')
    plot.build_config(None, xmin, xmax, ymin, ymax, xtitle, ytitle, topLeft, topRight)
    canvas = plot.draw(outName, suffix, [1.0, 4.0])
    canvas.Close()

    return None


def plot_mass_multiple(
        inputdirs: list[str],
        labels: list[str],
        outputdir,
        xmin: float,
        xmax: float,
        legend_header: str = '',
        ecm: int = 240,
        lumi: float = 10.8,
        xtitle: str = 'm_{H} [GeV]',
        ytitle: str = '-2#DeltaNLL',
        topRight: str | None = None,
        outName: str = '',
        suffix: str = '',
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
    plot.build_config(graphs[0], xmin, xmax, xtitle=xtitle, ytitle=ytitle, topRight=topRight)
    canvas = plot.draw(outName, suffix, [1.0, 4.0], legend_header)
    canvas.Close()

    return None


def plot_xsec_multiple(
        tags: list[str],
        labels: list[str],
        outputdir: str,
        xmin: float,
        xmax: float,
        ecm: int = 240,
        lumi: float = 10.8,
        xtitle: str = '#sigma(ZH#rightarrowl^{#plus}l^{#minus})/#sigma_{ref}',
        ytitle: str = '-2#DeltaNLL',
        topRight: str | None = None,
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
    plot.build_config(graphs[0], xmin, xmax, xtitle=xtitle, ytitle=ytitle, topRight=topRight,)
    canvas = plot.draw(outName, suffix, [1.0, 4.0])
    canvas.Close()

    return None


def plot_spline_scan(
        outputdir: str,
        obs: ROOT.RooRealVar,
        x_vals: list[float | int],
        y_vals: list[float | int],
        spline: 'ROOT.RooSpline1D',
        label: str = '',
        color: ROOT.TColor | int = ROOT.kBlack,
        ecm: int = 240,
        lumi: float = 10.8,
        xtitle: str = 'm_{H} [GeV]',
        ytitle: str = '',
        outName: str = '',
        suffix: str = '',
        file_formats: list[str] = ['png']
) -> None:

    graph = ROOT.TGraphErrors(len(x_vals), array.array('d', x_vals), array.array('d', y_vals),
                              array.array('d', [0] * len(x_vals)),
                              array.array('d', [0] * len(x_vals)))
    plot = GraphPlot(outputdir, graph, None, ecm, lumi, file_formats)
    plot.build_config(xmin=124.9, xmax=125.1, xtitle=xtitle, ytitle=ytitle)
    
    canvas, dummy = plotter.canvas(left=0.2, config=plot.cfg), plotter.dummy(config=plot.cfg)
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    frame = obs.frame()
    spline.plotOn(frame)
    GraphPlot.style_graph(graph, color, 2, 8, 1.5)
    graph.Draw('SAME P')
    frame.Draw('SAME')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.25, 0.92, label)

    plotter.finalize_canvas(canvas, config=plot.cfg)
    save_canvas(canvas, outputdir, outName, suffix, file_formats)

    return None


def plot_decomposition(
        outputdir: str,
        workspace: ROOT.RooWorkspace,
        label: str,
        mH_label: str,
        yield_nom: float | int,
        spec: dict,
        ecm: int = 240,
        lumi: float = 10.8,
        xMin: float | int | None = None,
        xMax: float | int | None = None,
        yMax: float | int | None = None,
        scale_max: float = 1.25,
        xtitle: str = 'm_{recoil} [GeV]',
        ytitle: str = 'Events',
        outName: str = 'fit_mH',
        suffix: str = '',
        file_formats: list[str] = ['png'],
) -> None:

    mrec = workspace.var('zll_recoil_m')
    pdf = workspace.pdf(f"{spec['model_name']}_{mH_label}")
    xmin, xmax, ymax = get_limits(mrec, pdf, xMin, xMax, yMax, scale_max, yield_nom)

    cfg = plotter.make_cfg({'xmin': xmin, 'xmax': xmax, 'ymin': 0, 'ymax': ymax,
                            'xtitle': xtitle, 'ytitle': ytitle}, ecm, lumi)
    canvas, dummy = plotter.canvas(config=cfg), plotter.dummy(config=cfg)
    dummy.Draw('HIST')

    plt = mrec.frame()
    fracs = [workspace.obj(f"{frac}_{mH_label}").getVal() for frac in spec['fractions']]
    comps = [workspace.obj(f"{comp['name']}_{mH_label}")  for comp in spec['components']]

    comp_data = [('Total PDF', ROOT.kBlack, pdf, yield_nom)]
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kCyan, ROOT.kGreen, ROOT.kMagenta, ROOT.kOrange, ROOT.kViolet]
    for idx, (comp_spec, comp) in enumerate(zip(spec['components'], comps)):
        if idx < len(fracs): comp_yield = yield_nom * fracs[idx]
        else:                    comp_yield = yield_nom * (1 - sum(fracs))
        comp_data.append((comp_spec['title'], colors[idx % len(colors)], comp, comp_yield))

    for _, color, comp, comp_yield in comp_data[1:]:
        comp.plotOn(plt, ROOT.RooFit.LineColor(color),  ROOT.RooFit.Normalization(comp_yield, ROOT.RooAbsReal.NumEvent))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.Normalization(yield_nom,  ROOT.RooAbsReal.NumEvent))

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

    plotter.finalize_canvas(canvas, config=cfg)
    save_canvas(canvas, outputdir, outName, f'{mH_label}_decomposition{suffix}', file_formats)

    return pdf


def plot_fit(
        outputdir: str,
        mrec: ROOT.RooRealVar,
        data: ROOT.RooDataHist,
        pdf: ROOT.RooAbsPdf,
        label: str,
        nBins: int = 250,
        ecm: int = 240,
        lumi: float = 10.8,
        xMin: float | int | None = None,
        xMax: float | int | None = None,
        yMax: float | int | None = None,
        scale_max: float = 1.25,
        xtitle: str = 'm_{recoil} [GeV]',
        ytitle: str = 'Events',
        outName: str = 'fit_mH',
        suffix: str = '',
        file_formats: list[str] = ['png'],
) -> None:

    xmin, xmax, ymax = get_limits(mrec, data, xMin, xMax, yMax, scale_max)
    cfg = plotter.make_cfg({'xmin': xmin, 'xmax': xmax, 'ymin': 0, 'ymax': max(ymax, 1e-12),
                            'xtitle': xtitle, 'ytitle': ytitle}, ecm, lumi, True)

    ratio_plot = RatioPlot(cfg, outputdir, outName, suffix, file_formats, (0,), (ROOT.kBlue + 2,))

    def annotate(frame) -> None:
        latex = setup_latex(0.045, 13, 1, 42)
        draw_latex(latex, [(label, 0.2, 0.88, 0.045), f'#chi^2 = {frame.chiSquare():.3f}', 0.2, 0.82, 0.045])

    ratio_plot.draw(data, pdf, mrec, nBins, pdf_options=(ROOT.RooFit.LineColor(ROOT.kRed),), param_pdf=pdf,
                    param_opts=(ROOT.RooFit.Format('NELU'), ROOT.RooFit.AutoPrecision(2), ROOT.RooFit.Layout(0.45, 0.9, 0.9)),
                    title='ZH signal', annotate=annotate)
    ratio_plot.save()
    ratio_plot.close()

    return None



def plot_fit_all(
        outputdir: str,
        workspace: ROOT.RooWorkspace,
        mHs: list[str],
        normalization: float | int,
        label: str,
        ecm: int = 240,
        lumi: float = 10.8,
        xMin: float | int | None = None,
        xMax: float | int | None = None,
        yMax: float | int | None = None,
        scale_max: float = 1.25,
        xtitle: str = 'm_{recoil} [GeV]',
        ytitle: str = 'Events',
        outName: str = 'fil_all',
        suffix: str = '',
        file_formats: list[str] = ['png'],
) -> None:

    mrec = workspace.var('zll_recoil_m')
    pdfs = [workspace.pdf('zh_model_' + f'{mH:.3f}'.replace('.', 'p')) for mH in mHs]
    limits = [get_limits(mrec, pdf, xMin, xMax, yMax, scale_max, normalization) for pdf in pdfs]
    xmin, xmax, ymax = limits[0][0], limits[0][1], max(limit[2] for limit in limits)
    cfg = plotter.make_cfg({'xmin': xmin, 'xmax': xmax, 'ymin': 0, 'ymax': max(ymax, 1e-12),
                            'xtitle': xtitle, 'ytitle': ytitle}, ecm, lumi)

    canvas, dummy = plotter.canvas(config=cfg), plotter.dummy(config=cfg)
    dummy.Draw('HIST')

    plt = mrec.frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, pdf in enumerate(pdfs):
        # Need to re-normalize the pdf, as the pdf is normalized to 1
        pdf.plotOn(plt, ROOT.RooFit.LineColor(colors[i]), ROOT.RooFit.Normalization(normalization, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.finalize_canvas(canvas, config=cfg)
    save_canvas(canvas, outputdir, outName, suffix, file_formats)

    return None


def plot_params_vs_mh(
        obs: ROOT.RooRealVar,
        outputdir: str | PathObj,
        param: str,
        vals: dict[str, float | int],
        spline: 'ROOT.RooSpline1D',
        label: str = '',
        ecm: int = 240,
        lumi: float = 10.8,
        xMin: float | int | None = None,
        xMax: float | int | None = None,
        yMax: float | int | None = None,
        xtitle: str = 'm_{H} [GeV]',
        ytitle: str | None = None,
        outName: str = '',
        suffix: str = '',
        file_formats: list[str] = ['png'],
) -> None:

    mHs = vals['mH']
    graph = ROOT.TGraphErrors(len(mHs), array.array('d', mHs), array.array('d', vals[param]),
                              array.array('d', [0]*len(mHs)),
                              array.array('d', [0]*len(mHs)))

    plot = GraphPlot(outputdir, graph, None, ecm, lumi, file_formats)
    plot.build_config(xmin=xMin, xmax=xMax, ymax=yMax, xtitle=xtitle, ytitle=ytitle or param_label.get(param, param))

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    canvas, dummy = plotter.canvas(left=0.2, config=plot.cfg), plotter.dummy(config=plot.cfg)
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    plt = obs.frame()
    spline.plotOn(plt)
    GraphPlot.style_graph(graph, ROOT.kBlack, 2, 8, 1.5)
    graph.Draw('SAME P')
    plt.Draw('SAME')

    plotter.finalize_canvas(canvas, config=plot.cfg)
    outName = outName or f'fit_{param}'
    save_canvas(canvas, outputdir, outName, suffix, file_formats)

    return None


def fit_plot(
    data: ROOT.RooDataHist,
    pdf: ROOT.RooAddPdf,
    obs: ROOT.RooRealVar,
    nBins: int,
    label: str,
    outputdir: str | PathObj,
    ecm: int = 240,
    lumi: float = 10.8,
    xMin: float | int | None = None,
    xMax: float | int | None = None,
    yMax: float | int | None = None,
    scale_max: float = 1.25,
    xtitle: str = 'm_{recoil} [GeV]',
    ytitle: str = 'Events',
    outName: str = 'fit_mH',
    suffix: str = '',
    file_formats: list[str] = ['png'],
) -> None:

    xmin, xmax, ymax = get_limits(obs, data, xMin, xMax, yMax, scale_max)
    cfg = plotter.make_cfg({'xmin': xmin, 'xmax': xmax, 'ymin': 0, 'ymax': max(ymax, 1e-12),
                            'xtitle': xtitle, 'ytitle': ytitle, 'yminR': -3.5, 'ymaxR': 3.5}, ecm, lumi, True)
    ratio_plot = RatioPlot(cfg, outputdir, outName, suffix, file_formats, (0,), (ROOT.kBlue + 2,))

    def annotate(frame):
        latex = setup_latex(0.045, 13, 1, 42)
        draw_latex(latex, [(label, 0.2, 0.88, 0.045), f'#chi^2 = {frame.chiSquare():.3f}', 0.2, 0.82, 0.045])

    ratio_plot.draw(data, pdf, obs, nBins, pdf_opts=(ROOT.RooFit.LineColor(ROOT.kRed),), param_pdf=pdf,
                    param_opts=(ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(0.45, 0.9, 0.9)),
                    title='ZH signal', annotate=annotate)
    ratio_plot.save()
    ratio_plot.close()

    return None


def plot_syst_dist(
        workspace: ROOT.RooWorkspace,
        normalization: int,
        outputdir: str,
        syst: str,
        mH_label: str,
        ecm: int = 240,
        lumi: float = 10.8,
        xMin: float | int | None = None,
        xMax: float | int | None = None,
        yMax: float | int | None = None,
        scale_max: float = 1.25,
        xtitle: str = 'm_{recoil} [GeV]',
        ytitle: str = 'Events',
        outName: str = 'fit_mH',
        suffix: str = '',
        file_formats: list[str] = ['png'],
) -> None:

    mrec =  workspace.var('zll_recoil_m')
    fits = [workspace.pdf(f'zh_model_{mH_label}{channel}') for channel in [f'_{syst}Up', '', f'_{syst}Down']]
    limits = [get_limits(mrec, pdf, xMin, xMax, yMax, scale_max, normalization) for pdf in fits]
    xmin, xmax, ymax = limits[0][0], limits[0][1], max(limit[2] for limit in limits)
    cfg = plotter.make_cfg({'xmin': xmin, 'xmax': xmax, 'ymin': 0, 'ymax': max(ymax, 1e-12),
                            'xtitle': xtitle, 'ytitle': ytitle}, ecm, lumi)

    canvas, dummy = plotter.canvas(config=cfg), plotter.dummy(config=cfg)
    canvas.SetGrid()
    dummy.Draw('HIST')

    plt = mrec.frame()
    colors = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]

    for i, sig_fit in enumerate(fits):
        sig_fit.plotOn(plt, ROOT.RooFit.Linecolor(colors[i]), ROOT.RooFit.Normalization(normalization, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')
    plotter.finalize_canvas(canvas, config=cfg)
    save_canvas(canvas, outputdir, outName, f'{mH_label}_{syst}{suffix}', file_formats)

    return None


def plot_fit_with_pull(
    data: ROOT.RooDataHist,
    pdf: ROOT.RooAddPdf,
    obs: ROOT.RooRealVar,
    outputdir: str,
    label_text: str,
    nBins: int = 250,
    ecm: int = 240,
    lumi: float = 10.8,
    xMin: float | int | None = None,
    xMax: float | int | None = None,
    yMax: float | int | None = None,
    scale_max: float = 1.25,
    xtitle: str = 'm_{recoil} [GeV]',
    ytitle: str = '',
    outName: str = 'fit_mH',
    suffix: str = '',
    file_formats: list[str] = ['png'],
    title: str | None = None,
    fit_color: ROOT.TColor = ROOT.kRed,
    param_layout: tuple[float | int] | None = (0.25, 0.9, 0.9),
) -> None:

    xmin, xmax, ymax = get_limits(obs, data, xMin, xMax, yMax, scale_max)
    cfg = plotter.make_cfg({'xmin': xmin, 'xmax': xmax, 'ymax': max(ymax, 1e-12), 'yminR': -3.5, 'ymaxR': 3.5,
                            'xtitle': xtitle, 'ytitle': ytitle + f' / {obs.getBinning().average():.0f}'}, ecm, lumi, True)
    ratio_plot = RatioPlot(cfg, outputdir, outName, suffix, file_formats, (0,))

    def annotate(frame):
        latex = setup_latex(0.045, 13, 1, 42)
        draw_latex(latex, [(label_text, 0.2, 0.88, 0.045), f'#chi^2 = {frame.chiSquare():.3f}', 0.2, 0.82, 0.045])

    param_opts = None
    if param_layout is not None:
        param_opts = (ROOT.RooFit.Format('NELU'), ROOT.RooFit.AutoPrecision(2), ROOT.RooFit.Layout(*param_layout))
    ratio_plot.draw(data, pdf, obs, nBins, pdf_opts=(ROOT.RooFit.LineColor(fit_color),),
                    param_pdf=pdf if param_layout is not None else None, param_opts=param_opts, title=title, annotate=annotate)
    ratio_plot.save()
    ratio_plot.close()

    return None


def plot_signal(
        workspace: ROOT.RooWorkspace,
        outputdir: str,
        label: str,
        normalization: float | int,
        pdfs: list[ROOT.RooAddPdf],
        ecm: int = 240,
        lumi: float = 10.8,
        xMin: float | int | None = None,
        xMax: float | int | None = None,
        yMax: float | int | None = None,
        scale_max: float = 1.25,
        xtitle: str = 'm_{recoil} [GeV]',
        ytitle: str = 'Events',
        outName: str = 'fil_all',
        suffix: str = '',
        file_formats: list[str] = ['png'],
) -> None:

    mrec = workspace.var('zll_recoil_m')
    limits = [get_limits(mrec, pdf, xMin, xMax, yMax, scale_max, normalization) for pdf in pdfs]
    xmin, xmax, ymax = limits[0][0], limits[0][1], max(limit[2] for limit in limits)
    cfg = plotter.make_cfg({'xmin': xmin, 'xmax': xmax, 'ymin': 0, 'ymax': max(ymax, 1e-12),
                            'xtitle': xtitle, 'ytitle': ytitle}, ecm, lumi)
    canvas, dummy = plotter.canvas(left=0.2, config=cfg), plotter.dummy(config=cfg)
    dummy.Draw('HIST')

    plt = workspace.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, pdf in enumerate(pdfs):
        # Need to re-normalize the pdf, as the pdf is normalized to 1
        pdf.plotOn(plt, ROOT.RooFit.LineColor(colors[i]), ROOT.RooFit.Normalization(normalization, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.finalize_canvas(canvas, config=cfg)
    save_canvas(canvas, outputdir, outName, suffix, file_formats)

    return None
