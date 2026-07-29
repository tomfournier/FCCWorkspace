import array, ROOT

from ..userConfig import PathObj
from .root import plotter



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


def _style_graph(graph, color, width: int = 4, marker: int = 20):
    graph.SetMarkerStyle(marker)
    graph.SetMarkerColor(color)
    graph.SetMarkerSize(1)
    graph.SetLineColor(color)
    graph.SetLineWidth(width)


def _draw_reference_line(x_min: float, x_max: float, y: float = 1.0):
    line = ROOT.TLine(float(x_min), y, float(x_max), y)
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
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        topLeft: str,
        topRight: str,
        output_name: str = 'mass',
        graph_color=ROOT.kRed,
        graph_width: int = 2,
        graph_marker: int = 20,
        x_title: str = 'm_{H} [GeV]',
        y_title: str = '-2#DeltaNLL'):

    cfg = _mass_plot_cfg(x_min, x_max, y_min, y_max, topLeft, topRight, x_title, y_title)
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()

    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    _style_graph(graph, graph_color, width=graph_width, marker=graph_marker)
    graph.Draw('SAME LP')
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    leg = ROOT.TLegend(.20, 0.825, 0.90, .9)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.035)
    leg.SetMargin(0.15)
    leg.SetBorderSize(1)
    leg.AddEntry(graph, f'{label}, #Delta(m_{{H}}) = {uncertainty_mev:.2f} MeV', 'LP')
    leg.Draw()

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/{output_name}.png')


def plot_mass_breakdown_impacts(
        outDir,
        uncertainties_mev: list[float],
        labels: list[str],
        topRight: str,
        topLeft: str,
        x_title: str = '#sigma_{syst.}(m_{H}) (MeV)',
        x_min: float = -3,
        x_max: float = 3,
        output_name: str = 'mass_breakdown_impacts'):

    canvas = ROOT.TCanvas('c', 'c', 1000, 1000)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.25)
    canvas.SetRightMargin(0.05)
    canvas.SetFillStyle(4000)
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

    g_pulls.SetMarkerSize(0.8)
    g_pulls.SetMarkerStyle(20)
    g_pulls.SetLineWidth(2)
    g_pulls.Draw('P SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(30)
    latex.DrawLatex(0.95, 0.925, topRight)

    latex.SetTextAlign(13)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.25, 0.96, topLeft)

    canvas.SaveAs(f'{outDir}/{output_name}.png')


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
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    leg = ROOT.TLegend(.20, 0.9 - len(curves) * 0.05, 0.90, .9)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.1)
    leg.SetBorderSize(1)

    for (x_values, y_values, uncertainty_mev, color, width, marker), label in zip(curves, labels):
        graph = ROOT.TGraph(len(x_values), array.array('d', x_values), array.array('d', y_values))
        _style_graph(graph, color, width=width, marker=marker)
        graph.Draw('SAME L')
        leg.AddEntry(graph, f'{label} #delta(m_{{H}}) = {uncertainty_mev:.2f} MeV', 'L')

    leg.Draw()
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/{output_name}.png')


def plot_mass_multiple(
        in_dirs,
        labels: list[str],
        out_dir,
        suffix: str,
        top_right: str,
        x_min: float,
        x_max: float,
        y_min: float = 0,
        y_max: float = 2,
        leg_label: str = '',
        force_stat: list[bool] | None = None,
        leg_margin: float = 0.15):

    if force_stat is None:
        force_stat = [False] * len(in_dirs)

    graphs = []
    uncertainties = []
    for i, in_dir in enumerate(in_dirs):
        file_name = f'mass{suffix + "_stat" if force_stat[i] else suffix}.txt'
        _, unc_mass, _, _, graph = _load_nll_curve(in_dir / file_name)
        uncertainties.append(unc_mass)
        graphs.append(graph)

    cfg = {
        'logy': False,
        'logx': False,
        'xmin': x_min,
        'xmax': x_max,
        'ymin': y_min,
        'ymax': y_max,
        'xtitle': 'm_{H} [GeV]',
        'ytitle': '-2#DeltaNLL',
        'topRight': top_right,
        'topLeft': '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    }
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    n = len(graphs) + (0 if leg_label == '' else 1)
    leg = ROOT.TLegend(.20, 0.9 - n * 0.05, 0.90, .9)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(leg_margin)
    leg.SetBorderSize(1)
    if leg_label != '':
        leg.SetHeader(leg_label)

    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 1]
    for index, graph in enumerate(graphs):
        _style_graph(graph, colors[index])
        graph.Draw('SAME L')
        leg.AddEntry(graph, f'{labels[index]} #delta(m_{{H}}) = {uncertainties[index]*1000:.2f} MeV', 'L')

    leg.Draw()
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{out_dir}{suffix}.png')


def plot_xsec_multiple(
        tags: list[str],
        labels: list[str],
        out_dir,
        top_right: str,
        x_min: float,
        x_max: float,
        y_min: float = 0,
        y_max: float = 2,
        output_name: str = ''):

    graphs = []
    uncertainties = []
    for tag in tags:
        _, unc_xsec, _, _, graph = _load_nll_curve(f'{tag}/xsec.txt')
        uncertainties.append(unc_xsec)
        graphs.append(graph)

    cfg = {
        'logy': False,
        'logx': False,
        'xmin': x_min,
        'xmax': x_max,
        'ymin': y_min,
        'ymax': y_max,
        'xtitle': '#sigma(ZH#rightarrowl^{#plus}l^{#minus})/#sigma_{ref}',
        'ytitle': '-2#DeltaNLL',
        'topRight': top_right,
        'topLeft': '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    }
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    leg = ROOT.TLegend(.20, 0.9 - len(graphs) * 0.05, 0.90, .9)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.15)
    leg.SetBorderSize(1)

    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 1]
    for index, graph in enumerate(graphs):
        _style_graph(graph, colors[index])
        graph.Draw('SAME L')
        leg.AddEntry(graph, f'{labels[index]} #delta(#sigma) = {uncertainties[index]*100:.2f}', 'L')

    leg.Draw()
    _draw_reference_line(cfg['xmin'], cfg['xmax'])

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    name = output_name or out_dir
    canvas.SaveAs(f'{name}.png')


def plot_spline_scan(
        outDir,
        MH,
        x_values,
        y_values,
        spline,
        output_name: str,
        y_title: str,
        y_min: float,
        y_max: float,
        x_min: float = 124.9,
        x_max: float = 125.1,
        x_title: str = 'm_{H} (GeV)',
        label: str = '',
        marker_color=ROOT.kBlack,
        topLeft: str = '',
        topRight: str = ''):

    graph = ROOT.TGraphErrors(
        len(x_values),
        array.array('d', x_values),
        array.array('d', y_values),
        array.array('d', [0] * len(x_values)),
        array.array('d', [0] * len(x_values)),
    )

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : x_min,
        'xmax'              : x_max,
        'ymin'              : y_min,
        'ymax'              : y_max,

        'xtitle'            : x_title,
        'ytitle'            : y_title,

        'topRight'          : topRight,
        'topLeft'           : topLeft,
    }

    plotter.cfg = cfg
    canvas = plotter.canvas(leftMargin=0.2)
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    frame = MH.frame()
    spline.plotOn(frame)
    graph.SetMarkerStyle(8)
    graph.SetMarkerColor(marker_color)
    graph.SetMarkerSize(1.5)
    graph.Draw('SAME P')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.25, 0.92, label)

    frame.Draw('SAME')
    plotter.aux()
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/{output_name}.png')


def plot_decomposition(
        outDir,
        w_tmp,
        label: str,
        mH_label: str,
        yield_nom: float | int,
        model_spec: dict,
        topLeft: str = '',
        topRight: str = '',
        yMax: float | int | None = None,
         ):

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : 3 * yMax if yMax is not None else 1,

        'xtitle'            : 'm_{recoil} [GeV]',
        'ytitle'            : 'Events',

        'topRight'          : topRight,
        'topLeft'           : topLeft,

        'ratiofraction'     : 0.3,
        'ytitleR'           : 'Pull',
        'yminR'             : -3.5,
        'ymaxR'             :  3.5,
    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')
    plt = w_tmp.var('zll_recoil_m').frame()

    sig_fit = w_tmp.pdf(f"{model_spec['model_name']}_{mH_label}")
    fractions = [w_tmp.obj(f"{fraction}_{mH_label}").getVal() for fraction in model_spec['fractions']]
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

    leg = ROOT.TLegend(.50, 0.7, .95, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)
    leg.SetMargin(0.15)

    for name, color, _, _ in component_data:
        tmp = ROOT.TGraph()
        tmp.SetPoint(0, 0, 0)
        tmp.SetLineColor(color)
        tmp.SetLineWidth(3)
        tmp.Draw('SAME')
        leg.AddEntry(tmp, name, 'L')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.2, 0.92, label)

    plt.Draw('SAME')
    leg.Draw()
    plotter.aux()

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}_decomposition.png')

    return sig_fit


def plot_fit(
        outDir,
        recoilmass,
        rdh_zh,
        sig_fit,
        mH_label: str,
        yMax: float | int,
        label: str,
        nBins: int = 250,
        topLeft: str = '',
        topRight: str = ''
         ):

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : yMax,

        'xtitle'            : 'm_{recoil} [GeV]',
        'ytitle'            : 'Events',

        'topRight'          : topRight,
        'topLeft'           : topLeft,

        'ratiofraction'     :  0.3,
        'ytitleR'           : 'Pull',
        'yminR'             : -3.5,
        'ymaxR'             :  3.5,
    }

    cfg['ymax'] = 1.2 * yMax
    plotter.cfg = cfg

    canvas, padT, padB     = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(rline=0)
    dummyB.GetXaxis().SetTitleOffset(4.*dummyB.GetXaxis().GetTitleOffset())   # hack label
    dummyT.GetYaxis().SetTitleOffset(1.2*dummyT.GetYaxis().GetTitleOffset())  # hack label

    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetGrid()
    dummyT.Draw('HIST')

    plt = recoilmass.frame()
    plt.SetTitle('ZH signal')
    rdh_zh.plotOn(plt,   ROOT.RooFit.Binning(nBins))
    sig_fit.plotOn(plt,  ROOT.RooFit.LineColor(ROOT.kRed))
    sig_fit.paramOn(plt, ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(0.45, 0.9, 0.9))
    histpull = plt.pullHist()
    plt.Draw('SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.2, 0.88, label)
    latex.DrawLatex(0.2, 0.82, f'#chi^2 = {plt.chiSquare():.3f}')

    plotter.auxRatio()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()

    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()
    dummyB.Draw('HIST')
    dummyL.Draw('SAME')

    plt = recoilmass.frame()
    plt.addPlotable(histpull, 'P')
    plt.Draw('SAME')

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}.png')

    del dummyB, dummyT
    del padT, padB
    del canvas



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

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : yMax,

        'xtitle'            : 'm_{recoil} [GeV]',
        'ytitle'            : 'Events',

        'topRight'          : topRight,
        'topLeft'           : topLeft,

        'ratiofraction'     : 0.3,
        'ytitleR'           : 'Pull',
        'yminR'             : -3.5,
        'ymaxR'             : 3.5,
    }

    cfg['xmin'] = 124
    cfg['xmax'] = 130
    cfg['ymax'] = 2.5 * yMax
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')

    plt = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, mH in enumerate(mHs):
        sig_fit = w_tmp.pdf('zh_model_'+f'{mH:.3f}'.replace('.', 'p'))
        # Need to re-normalize the pdf, as the pdf is normalized to 1
        sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[i]), ROOT.RooFit.Normalization(yield_zh, ROOT.RooAbsReal.NumEvent))


    plt.Draw('SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.2, 0.92, label)


    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_all.png')



def plot_params_vs_mh(
        MH: 'ROOT.RooRealVar',
        outDir: PathObj,
        param: str,
        vals: dict[str, float | int],
        spline: 'ROOT.RooSpline1D',
        topLeft: str = '',
        topRight: str = '',
        label: str = ''):

    mHs = vals['mH']

    graph = ROOT.TGraphErrors(
        len(mHs),
        array.array('d', mHs),
        array.array('d', vals[param]),
        array.array('d', [0]*len(mHs)),
        array.array('d', [0]*len(mHs))
    )

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 124.9,
        'xmax'              : 125.1,
        'ymin'              : 0.999 * min(vals[param]),
        'ymax'              : 1.001 * max(vals[param]),

        'xtitle'            : 'm_{H} [GeV]',
        'ytitle'            : param_label[param],

        'topRight'          : topRight,
        'topLeft'           : topLeft,
    }

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.cfg = cfg
    canvas = plotter.canvas(leftMargin=0.2)
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')
    dummy.GetXaxis().SetNdivisions(305)

    plt = MH.frame()
    spline.plotOn(plt)
    graph.SetMarkerStyle(8)
    graph.SetMarkerColor(ROOT.kBlack)
    graph.SetMarkerSize(1.5)
    graph.Draw('SAME P')

    latex.DrawLatex(0.25, 0.92, label)
    plt.Draw('SAME')
    plotter.aux()
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_{param}.png')


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
    dummyT, dummyB, _  = plotter.dummyRatio(1, 0)

    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.SetGrid()
    padT.cd()
    dummyT.Draw('HIST')

    plt = recoilmass.frame()
    plt.SetTitle('ZH signal')
    rdh_zh.plotOn(plt, ROOT.RooFit.Binning(nBins))  # ROOT.RooFit.Normalization(yield_zh, ROOT.RooAbsReal.NumEvent)

    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    chisq = plt.chiSquare()
    pdf.paramOn(plt, ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(0.45, 0.9, 0.9))

    histpull = plt.pullHist()
    plt.Draw('SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.2, 0.88, label)
    latex.DrawLatex(0.2, 0.82, f'#chi^{{2}} = {chisq:.3f}')

    plotter.auxRatio()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()

    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.GetXaxis().SetTitleOffset(4.0*dummyB.GetXaxis().GetTitleOffset())
    dummyB.Draw('HIST')

    plt = recoilmass.frame()
    plt.addPlotable(histpull, 'P')
    plt.Draw('SAME')

    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw('SAME')

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}.png')

    del dummyB, dummyT
    del padT, padB
    del canvas


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

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}_{syst}.png')



def plot_fit_with_pull(
    rdh: ROOT.RooDataHist,
    pdf: ROOT.RooAddPdf,
    recoilmass: ROOT.RooRealVar,
    n_bins: int,
    output_base: str,
    label_text: str,
    title: bool = None,
    fit_color: ROOT.TColor = ROOT.kRed,
    save_pdf: bool = False,
    param_layout: tuple[float | int] = (0.25, 0.9, 0.9),
     ):

    canvas, padT, padB     = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(rline=0)
    dummyB.GetXaxis().SetTitleOffset(4.0 * dummyB.GetXaxis().GetTitleOffset())

    canvas.cd()
    padT.Draw()
    padT.SetGrid()
    padT.cd()
    dummyT.Draw('HIST')

    plt = recoilmass.frame()
    if title is not None:
        plt.SetTitle(title)
    rdh.plotOn(plt, ROOT.RooFit.Binning(n_bins))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(fit_color))
    chisq = plt.chiSquare()
    if param_layout is not None:
        pdf.paramOn(plt, ROOT.RooFit.Format('NELU', ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(*param_layout))

    histpull = plt.pullHist()
    plt.Draw('SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.2, 0.88, label_text)
    latex.DrawLatex(0.2, 0.82, f'#chi^{{2}} = {chisq:.3f}')

    plotter.auxRatio()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()

    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()
    dummyB.Draw('HIST')
    dummyL.Draw('SAME')

    plt = recoilmass.frame()
    plt.addPlotable(histpull, 'P')
    plt.Draw('SAME')

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs(f'{output_base}.png')
    if save_pdf:
        canvas.SaveAs(f'{output_base}.pdf')

    del dummyB, dummyT, dummyL
    del padT, padB
    del canvas


def plot_signal(
        w_tmp: ROOT.RooWorkspace,
        mHs: list[float | int],
        outDir: str,
        label: str,
        yield_nom: float | int,
        pdf_sigs: list[ROOT.RooAddPdf]
         ):

    canvas = plotter.canvas(leftMargin=0.2)
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')

    plt = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, _ in enumerate(mHs):

        sig_fit = pdf_sigs[i]
        # Need to re-normalize the pdf, as the pdf is normalized to 1
        sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[i]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.2, 0.92, label)

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_all.png' )
