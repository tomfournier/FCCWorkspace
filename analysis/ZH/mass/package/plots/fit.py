import array, ROOT

from ..userConfig import PathObj
from .root import plotter




def plot_params_vs_mh(
        MH: 'ROOT.RooRealVar',
        outDir: PathObj,
        param_name: str,
        param_label: str,
        vals: dict[str, float | int],
        splines: dict[str,'ROOT.RooSpline1D'],
        topLeft: str,
        topRight: str,
        label: str):
    
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
        'ytitle'            : param_label,

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
    canvas.SaveAs(f'{outDir}/fit_{param_name}.png')


def plot_fit(
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

    canvas, padT, padB = plotter.canvasRatio()
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


def plot_decomposition(
        w_tmp: ROOT.RooWorkspace,
        outDir: str,
        label: str,
        mH_label: str,
        yield_nom: float | int
         ):

    cb1_val = w_tmp.obj(f'cb1_{mH_label}').getVal()
    cb2_val = w_tmp.obj(f'cb2_{mH_label}').getVal()

    cbs_1   = w_tmp.obj(f'cbs1_{mH_label}')
    cbs_2   = w_tmp.obj(f'cbs2_{mH_label}')
    gauss   = w_tmp.obj(f'gauss_{mH_label}')
    sig_fit = w_tmp.obj(f'zh_model_{mH_label}')

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')
    plt = w_tmp.var('zll_recoil_m').frame()

    cbs_1.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kRed),   ROOT.RooFit.Normalization(cb1_val * yield_nom, ROOT.RooAbsReal.NumEvent))
    cbs_2.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kBlue),  ROOT.RooFit.Normalization(cb2_val * yield_nom, ROOT.RooAbsReal.NumEvent))
    gauss.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kCyan),  ROOT.RooFit.Normalization((1 - cb1_val - cb2_val) * yield_nom, ROOT.RooAbsReal.NumEvent))
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    leg = ROOT.TLegend(.50, 0.7, .95, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)
    leg.SetMargin(0.15)

    # Define TGraphs for legend
    legends = ([ROOT.kBlack, 'Total PDF'], [ROOT.kRed, 'CB1'], [ROOT.kBlue, 'CB2'], [ROOT.kCyan, 'Gauss'])
    for color, name in legends:
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
    for i, mH in enumerate(mHs):

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
