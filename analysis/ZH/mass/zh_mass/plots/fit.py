import array, ROOT

from ..userConfig import PathObj, plot_file
from .root import plotter
from .root.helper import mk_legend, setup_latex, draw_latex, savecanvas, build_cfg
from .root.plotter import finalize_canvas, finalize_canvasRatio



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
    'n_1':      'n_1',
    'n_2':      'n_2',
    'cb_1':     'cb_1',
    'cb_2':     'cb_2'
}


def _mass_plot_cfg(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        top_left: str,
        top_right: str,
        x_title: str,
        y_title: str) -> dict:
    return {
        'logy': False,
        'logx': False,
        'xmin': x_min,
        'xmax': x_max,
        'ymin': y_min,
        'ymax': y_max,
        'xtitle': x_title,
        'ytitle': y_title,
        'topRight': top_right,
        'topLeft': top_left,
    }


def _style_graph(graph, color, width: int = 4, style: int = 20, size: float | int = 1):
    graph.SetMarkerStyle(style)
    graph.SetMarkerColor(color)
    graph.SetMarkerSize(size)
    graph.SetLineColor(color)
    graph.SetLineWidth(width)


def _draw_reference_line(x_min: float, x_max: float, y: float = 1.0):
    line = ROOT.TLine(x_min, y, x_max, y)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.Draw('SAME')
    return line


def _load_nll_curve(file_path):
    xv, yv = [], []
    with open(file_path, 'r') as fIn:
        for i, line in enumerate(fIn.readlines()):
            line = line.rstrip()
            if i == 0:
                best_mass = float(line.split(' ')[3])
                unc_mass = float(line.split(' ')[2])
            else:
                xv.append(float(line.split(' ')[0]))
                yv.append(float(line.split(' ')[1]))

    graph = ROOT.TGraph(len(xv), array.array('d', xv), array.array('d', yv))
    return best_mass, unc_mass, xv, yv, graph


def plot_mass_scan(
        outDir,
        graph,
        label: str,
        uncertainty_mev: float,
        output_name: str = 'mass',
        ecm: int = 240,
        graph_color=ROOT.kRed,
        graph_width: int = 2,
        graph_marker: int = 20):

    cfg = build_cfg(graph.GetHistogram(), False, False,
                    xtitle='m_{H} [GeV]', ytitle='-2#DeltaNLL', ecm=ecm)
    plotter.cfg = cfg

    canvas = plotter.canvas()
    dummy  = plotter.dummy()

    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    _style_graph(graph, graph_color, graph_width, graph_marker)
    graph.Draw('SAME LP')
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    leg = mk_legend(0, 1, 0.2, 0.825, 0.9, 0.9,
                    1, 0, 0.035, 0.15)
    leg.AddEntry(graph, f'{label}, #Delta(m_{{H}}) = {uncertainty_mev:.2f} MeV', 'LP')
    leg.Draw()

    finalize_canvas(canvas)
    savecanvas(canvas, outDir, output_name, '', plot_file)
    canvas.Close()


def plot_mass_breakdown_impacts(
        outDir,
        uncertainties_mev: list[float],
        labels: list[str],
        x_title: str = '#sigma_{syst.}(m_{H}) (MeV)',
        x_min: float = -3,
        x_max: float = 3,
        output_name: str = 'mass_breakdown_impacts'):

    canvas = plotter.canvas(
        1e3, 1e3, 0.08, 0.1, 0.25, 0.05, grid=False
    )
    canvas.SetGrid(1, 0)
    canvas.SetTickx(1)

    n_params = len(uncertainties_mev)
    h_pulls = ROOT.TH2F('pulls', 'pulls', 6, x_min, x_max, n_params, 0, n_params)
    g_pulls = ROOT.TGraphAsymmErrors(n_params)

    for index, (unc, label) in enumerate(zip(uncertainties_mev, labels)):
        point_index = n_params - 1 - index
        g_pulls.SetPoint(point_index, 0, float(point_index) + 0.5)
        g_pulls.SetPointError(point_index, unc, unc, 0., 0.)
        h_pulls.GetYaxis().SetBinLabel(point_index + 1, f'#splitline{label}{{({unc:.2g} MeV)}}')

    h_pulls.GetXaxis().SetTitleSize(0.04)
    h_pulls.GetXaxis().SetLabelSize(0.03)
    h_pulls.GetXaxis().SetTitle(x_title)
    h_pulls.GetXaxis().SetTitleOffset(1)
    h_pulls.GetYaxis().SetLabelSize(0.045)
    h_pulls.GetYaxis().SetTickLength(0)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.SetNdivisions(506, 'XYZ')
    h_pulls.Draw('HIST')

    _style_graph(g_pulls, ROOT.kBlack, 2, 20, 0.8)
    g_pulls.Draw('P SAME')

    finalize_canvas(canvas, False)
    savecanvas(canvas, outDir, output_name, '', plot_file)


def plot_mass_breakdown_curves(
        outDir,
        curves: list[tuple[list[float], list[float], float, str, int, int]],
        labels: list[str],
        topRight: str,
        topLeft: str,
        x_min: float = 124.995,
        x_max: float = 125.005,
        y_min: float = 0,
        y_max: float = 2,
        output_name: str = 'mass_breakdown'):

    cfg = _mass_plot_cfg(x_min, x_max, y_min, y_max, topLeft, topRight, 'm_{h} (GeV)', '-2#DeltaNLL')
    plotter.cfg = cfg

    canvas = plotter.canvas()
    dummy  = plotter.dummy()
    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    leg = mk_legend(len(curves), 1, 0.2, 0.9, 0.9, 0.9,
                    1, 0, 0.03, 0.1)

    for (x_values, y_values, uncertainty_mev, color, width, marker), label in zip(curves, labels):
        graph = ROOT.TGraph(len(x_values), array.array('d', x_values), array.array('d', y_values))
        _style_graph(graph, color, width=width, marker=marker)
        graph.Draw('SAME L')
        leg.AddEntry(graph, f'{label} #delta(m_{{H}}) = {uncertainty_mev:.2f} MeV', 'L')

    leg.Draw()
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    finalize_canvas(canvas)
    savecanvas(canvas, outDir, output_name, '', plot_file)


def plot_mass_multiple(
        in_dirs,
        labels: list[str],
        out_dir,
        suffix: str,
        top_right: str,
        x_min: float,
        x_max: float,
        leg_label: str = '',
        force_stat: list[bool] | None = None):

    if force_stat is None:
        force_stat = [False] * len(in_dirs)

    graphs, uncertainties = [], []
    for i, in_dir in enumerate(in_dirs):
        file_name = f'mass{suffix + "_stat" if force_stat[i] else suffix}.txt'
        _, unc_mass, _, _, graph = _load_nll_curve(in_dir / file_name)
        uncertainties.append(unc_mass)
        graphs.append(graph)

    cfg = build_cfg(graphs[0].createHistogram(), xmin=x_min, xmax=x_max,
                    xtitle='m_{H} [GeV]', ytitle='-2#DeltaNLL', decay=True,
                    hists=[g.createHistogram() for g in graphs])
    cfg = {'topRight': top_right}
    plotter.cfg = cfg

    canvas = plotter.canvas()
    dummy  = plotter.dummy()
    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    n = len(graphs) + (0 if leg_label == '' else 1)
    leg = mk_legend(n, 1, 0.2, 0.9, 0.9, 0.9,
                    0, 0, 0.03, 0.15)
    if leg_label != '': leg.SetHeader(leg_label)

    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 1]
    for index, graph in enumerate(graphs):
        _style_graph(graph, colors[index])
        graph.Draw('SAME L')
        leg.AddEntry(graph, f'{labels[index]} #delta(m_{{H}}) = {uncertainties[index]*1000:.2f} MeV', 'L')

    leg.Draw()
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    finalize_canvas(canvas)
    savecanvas(canvas, out_dir, '', suffix, plot_file)


def plot_xsec_multiple(
        tags: list[str],
        labels: list[str],
        out_dir,
        top_right: str,
        x_min: float,
        x_max: float,
        output_name: str = ''):

    graphs = []
    uncertainties = []
    for tag in tags:
        _, unc_xsec, _, _, graph = _load_nll_curve(f'{tag}/xsec.txt')
        uncertainties.append(unc_xsec)
        graphs.append(graph)

    cfg = build_cfg(graphs[0].createHistogram(), xmin=x_min, xmax=x_max, decay=True,
                    xtitle='#sigma(ZH#rightarrowl^{#plus}l^{#minus})/#sigma_{ref}',
                    ytitle='-2#DeltaNLL', hists=[g.createHistogram() for g in graphs])
    cfg = {'topRight': top_right}
    plotter.cfg = cfg

    canvas = plotter.canvas()
    dummy  = plotter.dummy()
    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    leg = mk_legend(len(graphs), 1, 0.2, 0.9, 0.9, 0.9,
                    1, 0, 0.03, 0.15)

    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 1]
    for index, graph in enumerate(graphs):
        _style_graph(graph, colors[index])
        graph.Draw('SAME L')
        leg.AddEntry(graph, f'{labels[index]} #delta(#sigma) = {uncertainties[index]*100:.2f}', 'L')

    leg.Draw()
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    finalize_canvas(canvas)
    savecanvas(canvas, out_dir, output_name, '', plot_file)


def plot_spline_scan(
        outDir,
        MH,
        x_values,
        y_values,
        spline,
        output_name: str,
        y_title: str,
        label: str = '',
        marker_color=ROOT.kBlack):

    graph = ROOT.TGraphErrors(
        len(x_values),
        array.array('d', x_values),
        array.array('d', y_values),
        array.array('d', [0] * len(x_values)),
        array.array('d', [0] * len(x_values)),
    )
    cfg = build_cfg(graph.GetHistogram(), xmin=124.9, xmax=125.1,
                    xtitle='m_{H} [GeV]', ytitle=y_title)
    plotter.cfg = cfg
    canvas = plotter.canvas(left=0.2)
    dummy  = plotter.dummy()
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    frame = MH.frame()
    spline.plotOn(frame)
    _style_graph(graph, marker_color, 2, 8, 1.5)
    graph.Draw('SAME P')
    frame.Draw('SAME')

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.25, 0.92, label)

    finalize_canvas(canvas)
    savecanvas(canvas, outDir, output_name, '', plot_file)


def plot_decomposition(
        outDir,
        w_tmp,
        label: str,
        mH_label: str,
        yield_nom: float | int,
        model_spec: dict,
        yMax: float | int | None = None,
         ):

    mrec = w_tmp.var('zll_recoil_m')
    cfg = build_cfg(mrec.createHistogram('mrec', ''), xmin=120, xmax=140,
                    xtitle='m_{recoil} [GeV]', ytitle='Events')
    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy  = plotter.dummy()
    dummy.Draw('HIST')

    plt = mrec.frame()
    sig_fit    =  w_tmp.pdf(f"{model_spec['model_name']}_{mH_label}")
    fractions  = [w_tmp.obj(f"{fraction}_{mH_label}").getVal() for fraction  in model_spec['fractions']]
    components = [w_tmp.obj(f"{component['name']}_{mH_label}") for component in model_spec['components']]

    component_data = [('Total PDF', ROOT.kBlack, sig_fit, yield_nom)]
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kCyan, ROOT.kGreen, ROOT.kMagenta, ROOT.kOrange, ROOT.kViolet]
    for index, (component_spec, component) in enumerate(zip(model_spec['components'], components)):
        if index < len(fractions):
            component_yield = fractions[index] * yield_nom
        else:
            component_yield = (1 - sum(fractions)) * yield_nom
        component_data.append((component_spec['title'], colors[index % len(colors)], component, component_yield))

    for _, color, component, component_yield in component_data[1:]:
        component.plotOn(plt, ROOT.RooFit.LineColor(color), ROOT.RooFit.Normalization(component_yield, ROOT.RooAbsReal.NumEvent))
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    leg = mk_legend(0, 1, 0.5, 0.7, 0.95, 0.9,
                    0, 0, 0.04, 0.15)

    for name, color, _, _ in component_data:
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

    finalize_canvas(canvas)
    savecanvas(canvas, outDir, 'fit_mH', f'{mH_label}_decomposition', plot_file)

    return sig_fit


def plot_fit(
        outDir,
        mrec,
        rdh_zh,
        sig_fit,
        mH_label: str,
        label: str,
        nBins: int = 250,
        ecm: int = 240,
         ):

    cfg = build_cfg(sig_fit.createHistogram('dummy', mrec),
                    False, False, ecm=ecm,
                    xtitle='m_{recoil} [GeV]', ytitle='Events')
    cfg = {'yminR' : -3.5, 'ymaxR' : 3.5, 'ytitleR' : 'Pull'}
    plotter.cfg = cfg

    canvas, padT, padB    = plotter.canvasRatio()
    dummyT, dummyB, lines = plotter.dummyRatio(1, [0])

    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetGrid()
    dummyT.Draw('HIST')

    plt = mrec.frame(ROOT.RooFit.Title('ZH signal'))
    rdh_zh.plotOn(plt,   ROOT.RooFit.Binning(nBins))
    sig_fit.plotOn(plt,  ROOT.RooFit.LineColor(ROOT.kRed))
    sig_fit.paramOn(plt, ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(0.45, 0.9, 0.9))
    histpull = plt.pullHist()
    plt.Draw('SAME')

    latex = setup_latex(0.045, 13, 1, 42)
    draw_latex(latex, [(label, 0.2, 0.88, 0.045),
                       f'#chi^2 = {plt.chiSquare():.3f}', 0.2, 0.82, 0.045])

    finalize_canvasRatio(canvas)

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
    savecanvas(canvas, outDir, 'fit_mH', mH_label, plot_file)

    # Explicitly delete objects to free memory faster
    canvas.Close()
    del canvas, dummyB, dummyT, padT, padB



def plot_fit_all(
        outDir,
        w_tmp,
        mHs,
        yield_zh,
        yMax,
        label,
        topLeft: str = '',
        topRight: str = ''
         ):

    mrec = w_tmp.var('zll_recoil_m')
    cfg = build_cfg(mrec.createHistogram('mrec', ''), xmin=124, xmax=130,
                    xtitle='m_{recoil} [GeV]', ytitle='Events')
    cfg = {'yminR': -3.5, 'ymaxR': 3.5, 'ytitleR': 'Pull'}
    plotter.cfg = cfg

    canvas = plotter.canvas()
    dummy  = plotter.dummy()
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

    finalize_canvas(canvas)
    savecanvas(canvas, outDir, 'fit_all', '', plot_file)



def plot_params_vs_mh(
        MH: 'ROOT.RooRealVar',
        outDir: PathObj,
        param: str,
        vals: dict[str, float | int],
        spline: 'ROOT.RooSpline1D',
        label: str = ''):

    mHs = vals['mH']

    graph = ROOT.TGraphErrors(
        len(mHs),
        array.array('d', mHs),
        array.array('d', vals[param]),
        array.array('d', [0]*len(mHs)),
        array.array('d', [0]*len(mHs))
    )

    cfg = build_cfg(graph.GetHistogram(), xmin=124.9, xmax=125.1,
                    xtitle='m_{H} [GeV]', ytitle=param_label[param])

    latex = setup_latex(0.04, 13, 1, 42)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.cfg = cfg
    canvas = plotter.canvas(left=0.2)
    dummy  = plotter.dummy()
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    plt = MH.frame()
    spline.plotOn(plt)
    _style_graph(graph, ROOT.kBlack, 2, 8, 1.5)
    graph.Draw('SAME P')
    plt.Draw('SAME')

    finalize_canvas(canvas)
    savecanvas(canvas, outDir, f'fit_{param}', '', plot_file)


def fit_plot(
    rdh_zh: 'ROOT.RooDataHist',
    pdf: 'ROOT.RooAddPdf',
    mH_label: str,
    recoilmass: 'ROOT.RooRealVar',
    nBins: int,
    label: int,
    outDir: PathObj
     ):

    canvas, padT, padB = plotter.canvasRatio()
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

    finalize_canvasRatio(canvas)

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
    savecanvas(canvas, outDir, 'fit_mH', mH_label, plot_file)

    del canvas, dummyT, dummyB, padT, padB


def plot_syst_dist(
        w_tmp: ROOT.RooWorkspace,
        yield_nom: int,
        outDir: str,
        syst: str,
        mH_label: str
         ):

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')

    plt    = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]

    for i, channel in enumerate([f'_{syst}Up', '', f'_{syst}Down']):
        sig_fit = w_tmp.pdf(f'zh_model_{mH_label}{channel}')
        sig_fit.plotOn(plt,
                       ROOT.RooFit.Linecolor(colors[i]),
                       ROOT.RooFit.Normalization(yield_nom,
                                                 ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')
    finalize_canvas(canvas)
    savecanvas(canvas, outDir, 'fit_mH', f'{mH_label}_{syst}', plot_file)



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
     ):

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

    finalize_canvasRatio(canvas)

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


def plot_signal(
        workspace: ROOT.RooWorkspace,
        mHs: list[float | int],
        outDir: str,
        label: str,
        yield_nom: float | int,
        pdf_sigs: list[ROOT.RooAddPdf]
         ):

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

    finalize_canvas(canvas)
    savecanvas(canvas, outDir, 'fit_all', '', plot_file)
