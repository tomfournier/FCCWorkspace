
import os, array, argparse, subprocess, ROOT

import numpy as np

import package.plots.root.plotter as plotter
from package.tools.process import getHist
from package.userConfig import loc
loc.set_default_type('Path')

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)



def plot_params_vs_mh(mHs, param, param_name, param_label, spline):

    graph = ROOT.TGraphErrors(
        len(mHs),
        array.array('d', mHs),
        array.array('d', param),
        array.array('d', [0]*len(mHs)),
        array.array('d', [0]*len(mHs))
    )

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 124.9,
        'xmax'              : 125.1,
        'ymin'              : 0.999 * min(param),
        'ymax'              : 1.001 * max(param),

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






# do plotting
def plot_fit(
    rdh_zh,
    pdf,
    mH_label
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




def _with_suffix(base, suffix):
    return base if suffix == '' else f'{base}_{suffix}'


def build_2cbg_pdf(
    recoilmass,
    suffix,
    mean,
    sigma,
    alpha_1,
    alpha_2,
    n_1,
    n_2,
    mean_gt,
    sigma_gt,
    cb_1,
    cb_2,
    yield_zh,
):

    cbs_1 = ROOT.RooCBShape(_with_suffix('CrystallBall_1', suffix), 'CrystallBall_1', recoilmass, mean,    sigma, alpha_1, n_1)
    cbs_2 = ROOT.RooCBShape(_with_suffix('CrystallBall_2', suffix), 'CrystallBall_2', recoilmass, mean,    sigma, alpha_2, n_2)
    gauss = ROOT.RooGaussian(_with_suffix('gauss', suffix),        'gauss',        recoilmass, mean_gt, sigma_gt)

    sig      = ROOT.RooAddPdf(_with_suffix('sig', suffix),       '', ROOT.RooArgList(cbs_1, cbs_2, gauss), ROOT.RooArgList(cb_1, cb_2))
    sig_norm = ROOT.RooRealVar(_with_suffix('sig_norm', suffix), '', yield_zh, 0, 1e6)
    sig_fit  = ROOT.RooAddPdf(_with_suffix('zh_model', suffix),  '', ROOT.RooArgList(sig), ROOT.RooArgList(sig_norm))

    return cbs_1, cbs_2, gauss, sig, sig_norm, sig_fit


def plot_fit_with_pull(
    rdh,
    pdf,
    recoilmass,
    n_bins,
    output_base,
    label_text,
    title=None,
    fit_color=ROOT.kRed,
    save_pdf=False,
    param_layout=(0.25, 0.9, 0.9),
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


def doSignal(normYields: bool = True):

    global h_obs, yield_nom, yMax

    mHs = [125.0, 124.95, 125.05]
    procs = [f'wzp6_ee_{flavor}H_ecm{ecm}', f'wzp6_ee_{flavor}H_mH-lower-50MeV_ecm{ecm}', f'wzp6_ee_{flavor}H_mH-higher-50MeV_ecm{ecm}']
    recoilmass = w_tmp.var('zll_recoil_m')
    MH = w_tmp.var('MH')

    param_yield, param_mh, param_mean, param_mean_gt = [], [], [], []
    param_mean_offset, param_mean_gt_offset, param_sigma, param_sigma_gt = [], [], [], []
    param_alpha_1, param_alpha_2, param_n_1 = [], [], []
    param_n_2, param_cb_1, param_cb_2 = [], [], []

    hist_norm = getHist(f'{flavor}_{hName}', [procs[1]])
    hist_norm.Scale(lumiScale)
    hist_norm = hist_norm.ProjectionX('hist_zh_norm', cat_idx_min, cat_idx_max)
    yield_nom = hist_norm.Integral()

    tmp = hist_norm.Clone()
    tmp = tmp.Rebin(int(hist_norm.GetNbinsX() / nBins))
    yMax = 1.25 * tmp.GetMaximum()

    # recoil mass plot settings
    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : yMax,

        'xtitle'            : 'm_{recoil} [GeV]',
        'ytitle'            : 'Events / 0.2 GeV',

        'topRight'          : topRight,
        'topLeft'           : topLeft,

        'ratiofraction'     :  0.3,
        'ytitleR'           : 'Pull',
        'yminR'             : -3.5,
        'ymaxR'             :  3.5,
    }

    garbage = []  # Need to store the variables for memory issues

    ## Build model
    ## linear functions for mean and mean_gt
    ## constants for all the rest

    # import values
    coeff = np.loadtxt(f'{outDir}/coeff.txt'.replace(f'lumi{lumiLabel}', 'base_parametric'))  # take the coefficients from the baseline (scaled to 1 ab-1)
    param_mean0_           = float(coeff[0])
    param_mean1_           = float(coeff[1])
    param_mean_gt_offset0_ = float(coeff[4])
    param_sigma_           = float(coeff[5])
    param_sigma_gt_        = float(coeff[6])
    param_alpha_1_         = float(coeff[7])
    param_alpha_2_         = float(coeff[8])
    param_n_1_             = float(coeff[9])
    param_n_2_             = float(coeff[10])
    param_cb_1_            = float(coeff[11])
    param_cb_2_            = float(coeff[12])

    mean_argl, sigma_argl              = ROOT.RooArgList('mean_argl'),     ROOT.RooArgList('sigma_argl')
    sigma_gt_argl, mean_gt_offset_argl = ROOT.RooArgList('sigma_gt_argl'), ROOT.RooArgList('mean_gt_offset_argl')
    alpha1_argl, n1_argl, cb1_argl     = ROOT.RooArgList('alpha1_argl'),   ROOT.RooArgList('n1_argl'), ROOT.RooArgList('cb1_argl')
    alpha2_argl, n2_argl, cb2_argl     = ROOT.RooArgList('alpha2_argl'),   ROOT.RooArgList('n2_argl'), ROOT.RooArgList('cb2_argl')

    mean0 = ROOT.RooRealVar('mean0', '', param_mean0_, 0.5, 1.5)  # slope
    mean0.setConstant(ROOT.kTRUE)
    mean1 = ROOT.RooRealVar('mean1', '', param_mean1_, -1, 1)  # offset
    mean1.setConstant(ROOT.kTRUE)
    mean_argl.add(mean0)
    mean_argl.add(mean1)

    mean_gt_offset = ROOT.RooRealVar('mean_gt_offset', '', param_mean_gt_offset0_, -2, 2)
    mean_gt_offset.setConstant(ROOT.kTRUE)
    mean_gt_offset_argl.add(mean_gt_offset)

    sigma0 = ROOT.RooRealVar('sigma0', '', param_sigma_, 0, 10)  # 0.4335
    sigma0.setConstant(ROOT.kTRUE)
    sigma_argl.add(sigma0)

    sigma_gt0 = ROOT.RooRealVar('sigma_gt0', '', param_sigma_gt_, 0, 10)
    sigma_gt0.setConstant(ROOT.kTRUE)
    sigma_gt_argl.add(sigma_gt0)

    alpha10 = ROOT.RooRealVar('alpha10', '', param_alpha_1_, -5, 5)
    alpha10.setConstant(ROOT.kTRUE)
    alpha1_argl.add(alpha10)
    n10 = ROOT.RooRealVar('n10', '', param_n_1_, -50, 50)
    n10.setConstant(ROOT.kTRUE)
    n1_argl.add(n10)
    cb10 = ROOT.RooRealVar('cb10', '', param_cb_1_, 0, 1)
    cb10.setConstant(ROOT.kTRUE)
    cb1_argl.add(cb10)

    alpha20 = ROOT.RooRealVar('alpha20', '', param_alpha_2_, -5, 5)
    alpha20.setConstant(ROOT.kTRUE)
    alpha2_argl.add(alpha20)
    n20 = ROOT.RooRealVar('n20', '', param_n_2_,  -50, 50)
    n20.setConstant(ROOT.kTRUE)
    n2_argl.add(n20)
    cb20 = ROOT.RooRealVar('cb20', '', param_cb_2_, 0, 1)
    cb20.setConstant(ROOT.kTRUE)
    cb2_argl.add(cb20)

    cats    = ROOT.RooCategory('category', '')           # For each mass bin, define category
    hists   = ROOT.std.map('string, RooDataHist*')()     # Container holding all RooDataHists
    pdf_tot = ROOT.RooSimultaneous('pdf_tot', '', cats)  # Total pdf, containing all the categories

    garbage = []
    list_alpha1, list_alpha2 = [], []
    list_n1, list_n2 = [], []
    list_cb1, list_cb2 = [], []
    list_sigma, list_sigma_gt, list_mean = [], []
    list_mean_offset, list_mean_gt, list_mean_gt_offset, list_norm = [], [], [], []

    for i, proc in enumerate(procs):

        mH = mHs[i]
        mH_label = f'{mH:.3f}'.replace('.', 'p')
        print(f'Do {mH = :.3f}')

        print(f'{proc}/{hName}')
        hist_zh = getHist(f'{flavor}_{hName}', [proc])
        hist_zh.Scale(lumiScale)
        hist_zh = hist_zh.ProjectionX(f'hist_zh_{mH_label}', cat_idx_min, cat_idx_max)
        if normYields: hist_zh.Scale(yield_nom/hist_zh.Integral())

        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{mH_label}', '', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        rdh_zh.SetName(f'rdh_zh_{mH_label}')
        yield_zh = rdh_zh.sum(False)

        catIDx = mH_label
        hists.insert(ROOT.std.pair('string, RooDataHist*')(catIDx, rdh_zh))  # Does not work with recent ROOT versions?
        cats.defineType(catIDx, i)

        mean = ROOT.RooFormulaVar(f'mean_{mH_label}', f'x[1] + x[0]*{mH}', mean_argl)
        garbage.append(mean)
        sigma = ROOT.RooFormulaVar(f'sigma_{mH_label}', 'x[0]', sigma_argl)

        mean_gt_argl = ROOT.RooArgList('mean_gt_argl')
        mean_gt_argl.add(mean)
        mean_gt_argl.add(mean_gt_offset)
        mean_gt  = ROOT.RooFormulaVar(f'mean_gt_{mH_label}',  'x[0] + x[1]', mean_gt_argl)  # mean gt is as an offset w.r.t. the mean
        sigma_gt = ROOT.RooFormulaVar(f'sigma_gt_{mH_label}', 'x[0]', sigma_gt_argl)

        alpha1 = ROOT.RooFormulaVar(f'alpha1_{mH_label}', 'x[0]', alpha1_argl)
        n1     = ROOT.RooFormulaVar(f'n1_{mH_label}',     'x[0]', n1_argl)
        cb1    = ROOT.RooFormulaVar(f'cb1_{mH_label}',    'x[0]', cb1_argl)

        alpha2 = ROOT.RooFormulaVar(f'alpha2_{mH_label}', 'x[0]', alpha2_argl)
        n2     = ROOT.RooFormulaVar(f'n2_{mH_label}',     'x[0]', n2_argl)
        cb2    = ROOT.RooFormulaVar(f'cb2_{mH_label}',    'x[0]', cb2_argl)


        # construct the 2CBG pdf = cb_1*cbs_1 + cb_2*cbs_2 + gauss (cb_1 and cb_2 are the fractions, floating)
        cbs1  = ROOT.RooCBShape(f'cbs1_{mH_label}', 'CrystallBall_1', recoilmass, mean, sigma, alpha1, n1)  # first CrystallBall
        cbs2  = ROOT.RooCBShape(f'cbs2_{mH_label}', 'CrystallBall_2', recoilmass, mean, sigma, alpha2, n2)  # second CrystallBall
        gauss = ROOT.RooGaussian(f'gauss_{mH_label}', 'gauss',        recoilmass, mean_gt, sigma_gt)        # the Gaussian

        argl = ROOT.RooArgList(cbs1, cbs2, gauss)
        argl.setName(f'argl_{mH_label}')
        norms_argl = ROOT.RooArgList(cb1, cb2)
        norms_argl.setName(f'norms_argl_{mH_label}')
        sig      = ROOT.RooAddPdf(f'sig_{mH_label}',       '', argl, norms_argl)  # half of both CB functions
        sig_norm = ROOT.RooRealVar(f'sig_norm_{mH_label}', '', yield_zh, 0, 1e8)  # fix normalization

        sig_argl = ROOT.RooArgList(sig)
        sig_argl.setName(f'sig_argl_{mH_label}')
        sig_norm_argl = ROOT.RooArgList(sig_norm)
        sig_norm_argl.setName(f'sig_norm_argl_{mH_label}')
        pdf_sig = ROOT.RooAddPdf(f'zh_model_{mH_label}', '', sig_argl, sig_norm_argl)
        pdf_sigs.append(pdf_sig)

        # must store the individual vars for later , to extract the values
        # seems not to work with workspace
        list_alpha1.append(alpha1)
        list_alpha2.append(alpha2)
        list_n1.append(n1)
        list_n2.append(n2)
        list_cb1.append(cb1)
        list_cb2.append(cb2)
        list_mean.append(mean)
        list_mean_offset.append(mean1)
        list_sigma.append(sigma)
        list_mean_gt.append(mean_gt)
        list_mean_gt_offset.append(mean_gt_offset)
        list_sigma_gt.append(sigma_gt)
        list_norm.append(sig_norm)

        garbage.append(mean_gt_argl)
        garbage.append(cbs1)
        garbage.append(cbs2)
        garbage.append(gauss)
        garbage.append(argl)
        garbage.append(norms_argl)
        garbage.append(sig)
        garbage.append(sig_argl)
        garbage.append(sig_norm_argl)
        garbage.append(pdf_sig)

        pdf_sig.Print()
        pdf_tot.addPdf(pdf_sig, catIDx)

        if mH == 125.0 and h_obs is None: h_obs = hist_zh.Clone('h_obs')  # Take 125.0 GeV to add to observed (need to add background later as well)

    rdh_tot = ROOT.RooDataHist('rdh_tot', '', ROOT.RooArgList(recoilmass), cats, hists)

    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.Minimizer('Minimizer', 'simplex'))
    print('****************************')
    print('FIT STATUS')
    print(f'Covariance Quality = {fitRes.covQual()}')
    print(f'Fit status = {fitRes.status()}')
    print('****************************')

    getattr(w_tmp, 'import')(pdf_tot)  # import after fit, to have fit values in the workspace

    cov = fitRes.covarianceMatrix()
    cov.Print()

    # Plot
    plotter.cfg = cfg
    cfg['ytitle'] = f'Events / {20_000/nBins:.0f} MeV'
    for i, proc in enumerate(procs):

        mH = mHs[i]
        mH_label = f'{mH:.3f}'.replace('.', 'p')
        cfg['ymax'] = yMax

        pdf = pdf_sigs[i]
        rdh_zh = hists[mH_label]

        plot_fit(rdh_zh, pdf, mH_label)

        cb1_val = w_tmp.obj(f'cb1_{mH_label}').getVal()
        cb2_val = w_tmp.obj(f'cb2_{mH_label}').getVal()

        cfg['ymax'] = 2.5*yMax

        plotter.cfg = cfg
        canvas = plotter.canvas()
        canvas.SetGrid()
        dummy = plotter.dummy()
        dummy.Draw('HIST')
        plt = w_tmp.var('zll_recoil_m').frame()
        colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]

        leg = ROOT.TLegend(.50, 0.7, .95, .90)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.04)
        leg.SetMargin(0.15)

        cbs_1   = w_tmp.obj(f'cbs1_{mH_label}')
        cbs_2   = w_tmp.obj(f'cbs2_{mH_label}')
        gauss   = w_tmp.obj(f'gauss_{mH_label}')
        sig_fit = w_tmp.obj(f'zh_model_{mH_label}')

        cbs_1.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kRed),   ROOT.RooFit.Normalization(cb1_val * yield_nom, ROOT.RooAbsReal.NumEvent))
        cbs_2.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kBlue),  ROOT.RooFit.Normalization(cb2_val * yield_nom, ROOT.RooAbsReal.NumEvent))
        gauss.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kCyan),  ROOT.RooFit.Normalization((1 - cb1_val - cb2_val) * yield_nom, ROOT.RooAbsReal.NumEvent))
        sig_fit.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

        # define TGraphs for legend
        tmp1 = ROOT.TGraph()
        tmp1.SetPoint(0, 0, 0)
        tmp1.SetLineColor(ROOT.kBlack)
        tmp1.SetLineWidth(3)
        tmp1.Draw('SAME')
        leg.AddEntry(tmp1, 'Total PDF', 'L')

        tmp2 = ROOT.TGraph()
        tmp2.SetPoint(0, 0, 0)
        tmp2.SetLineColor(ROOT.kRed)
        tmp2.SetLineWidth(3)
        tmp2.Draw('SAME')
        leg.AddEntry(tmp2, 'CB1', 'L')

        tmp3 = ROOT.TGraph()
        tmp3.SetPoint(0, 0, 0)
        tmp3.SetLineColor(ROOT.kBlue)
        tmp3.SetLineWidth(3)
        tmp3.Draw('SAME')
        leg.AddEntry(tmp3, 'CB2', 'L')

        tmp4 = ROOT.TGraph()
        tmp4.SetPoint(0, 0, 0)
        tmp4.SetLineColor(ROOT.kCyan)
        tmp4.SetLineWidth(3)
        tmp4.Draw('SAME')
        leg.AddEntry(tmp4, 'Gauss', 'L')

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

        # import
        getattr(w_tmp, 'import')(rdh_zh)
        getattr(w_tmp, 'import')(sig_fit)


        param_mh.append(mH)
        param_mean.append(list_mean[i].getVal())
        param_mean_offset.append(list_mean_offset[i].getVal())
        param_sigma.append(list_sigma[i].getVal())
        param_mean_gt.append(list_mean_gt[i].getVal())
        param_mean_gt_offset.append(list_mean_gt_offset[i].getVal())
        param_sigma_gt.append(list_sigma_gt[i].getVal())
        param_alpha_1.append(list_alpha1[i].getVal())
        param_alpha_2.append(list_alpha2[i].getVal())
        param_n_1.append(list_n1[i].getVal())
        param_n_2.append(list_n2[i].getVal())
        param_yield.append(list_norm[i].getVal())
        param_cb_1.append(list_cb1[i].getVal())
        param_cb_2.append(list_cb2[i].getVal())



    ##################################
    # plot all fitted signals
    ##################################
    cfg['xmin'] = 124
    cfg['xmax'] = 128
    cfg['ymax'] = 2.5 * np.average(yMax)
    plotter.cfg = cfg

    canvas = plotter.canvas(leftMargin=0.2)
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')

    plt = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, mH in enumerate(mHs):

        mH_label = f'{mH:.3f}'.replace('.', 'p')
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
    canvas.SaveAs(f'{outDir}/fit_all.pdf')

    n_param = len(param_mh)
    table = array.array('d', param_mh)
    # Make splines, to connect the fit parameters a function of the Higgs mass
    # Plot them afterwards
    spline_mean           = ROOT.RooSpline1D('spline_mean',           'spline_mean',           MH, n_param, table, array.array('d', param_mean))
    spline_mean_offset    = ROOT.RooSpline1D('spline_mean_offset',    'spline_mean_offset',    MH, n_param, table, array.array('d', param_mean_offset))
    spline_sigma          = ROOT.RooSpline1D('spline_sigma',          'spline_sigma',          MH, n_param, table, array.array('d', param_sigma))
    spline_mean_gt        = ROOT.RooSpline1D('spline_mean_gt',        'spline_mean_gt',        MH, n_param, table, array.array('d', param_mean_gt))
    spline_mean_gt_offset = ROOT.RooSpline1D('spline_mean_gt_offset', 'spline_mean_gt_offset', MH, n_param, table, array.array('d', param_mean_gt_offset))
    spline_sigma_gt       = ROOT.RooSpline1D('spline_sigma_gt',       'spline_sigma_gt',       MH, n_param, table, array.array('d', param_sigma_gt))
    spline_yield          = ROOT.RooSpline1D('spline_yield',          'spline_yield',          MH, n_param, table, array.array('d', param_yield))
    spline_alpha_1        = ROOT.RooSpline1D('spline_alpha_1',        'spline_alpha_1',        MH, n_param, table, array.array('d', param_alpha_1))
    spline_alpha_2        = ROOT.RooSpline1D('spline_alpha_2',        'spline_alpha_2',        MH, n_param, table, array.array('d', param_alpha_2))
    spline_n_1            = ROOT.RooSpline1D('spline_n_1',            'spline_n_1',            MH, n_param, table, array.array('d', param_n_1))
    spline_n_2            = ROOT.RooSpline1D('spline_n_2',            'spline_n_2',            MH, n_param, table, array.array('d', param_n_2))
    spline_cb_1           = ROOT.RooSpline1D('spline_cb_1',           'spline_cb_1',           MH, n_param, table, array.array('d', param_cb_1))
    spline_cb_2           = ROOT.RooSpline1D('spline_cb_2',           'spline_cb_2',           MH, n_param, table, array.array('d', param_cb_2))

    form_mean = ROOT.RooFormulaVar('form_mean', '@0*@1 + @2', ROOT.RooArgList(mean0, MH, mean1))
    form_mean.Print()

    plot_params_vs_mh(param_mh, param_mean,           'mean',           '#mu [GeV]',             spline_mean)
    plot_params_vs_mh(param_mh, param_mean_gt,        'mean_gt',        '#mu_{gt} [GeV]',        spline_mean_gt)
    plot_params_vs_mh(param_mh, param_mean_offset,    'mean_offset',    '#mu offset [GeV]',      spline_mean_offset)
    plot_params_vs_mh(param_mh, param_mean_gt_offset, 'mean_gt_offset', '#mu_{gt} offset [GeV]', spline_mean_gt_offset)
    plot_params_vs_mh(param_mh, param_yield,          'yield',          'Events',                spline_yield)
    plot_params_vs_mh(param_mh, param_sigma,          'sigma',          '#sigma [GeV]',          spline_sigma)
    plot_params_vs_mh(param_mh, param_sigma_gt,       'sigma_gt',       '#sigma_{gt} [GeV]',     spline_sigma_gt)
    plot_params_vs_mh(param_mh, param_alpha_1,        'alpha_1',        '#alpha_{1}',            spline_alpha_1)
    plot_params_vs_mh(param_mh, param_alpha_2,        'alpha_2',        '#alpha_{2}',            spline_alpha_2)
    plot_params_vs_mh(param_mh, param_n_1,            'n_1',            'n_{1}',                 spline_n_1)
    plot_params_vs_mh(param_mh, param_n_2,            'n_2',            'n_{2}',                 spline_n_2)
    plot_params_vs_mh(param_mh, param_cb_1,           'cb_1',           'cb_{1}',                spline_cb_1)
    plot_params_vs_mh(param_mh, param_cb_2,           'cb_2',           'cb_{2}',                spline_cb_2)


    # Was getattr(w_tmp, 'import')(spline_<variable>)
    # Should test if it works
    w_tmp.Import(spline_mean)
    w_tmp.Import(spline_sigma)
    w_tmp.Import(spline_yield)
    w_tmp.Import(spline_alpha_1)
    w_tmp.Import(spline_alpha_2)
    w_tmp.Import(spline_n_1)
    w_tmp.Import(spline_n_2)
    w_tmp.Import(spline_cb_1)
    w_tmp.Import(spline_cb_2)
    w_tmp.Import(spline_mean_gt)
    w_tmp.Import(spline_sigma_gt)

    return hist_norm


def doBackgrounds():

    global h_obs

    recoilmass = w_tmp.var('zll_recoil_m')
    hist_bkg = None


    procs = [
        f'p8_ee_WW_ecm{ecm}', f'p8_ee_ZZ_ecm{ecm}',
        f'wz3p6_ee_mumu_ecm{ecm}' if flavor=='mumu' else f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', 'wz3p6_ee_tautau_ecm240',
        f'wzp6_egamma_eZ_Z{flavor}_ecm{ecm}', f'wzp6_gammae_eZ_Z{flavor}_ecm{ecm}',
        f'wzp6_gaga_{flavor}_60_ecm{ecm}', f'wzp6_gaga_tautau_60_ecm{ecm}',
        f'wzp6_ee_nuenueZ_ecm{ecm}'
    ]

    for proc in procs:
        hist = getHist(f'{flavor}_{hName}', [proc])
        hist.Scale(lumiScale)
        hist = hist.ProjectionX(f'hist_{proc}', cat_idx_min, cat_idx_max)

        # Add to total background
        if hist_bkg is None: hist_bkg = hist
        else: hist_bkg.Add(hist)

        # Add to observed
        if h_obs is None: h_obs = hist.Clone('h_obs')
        else: h_obs.Add(hist)

    hist_bkg.SetName('total_bkg')
    rdh_bkg    = ROOT.RooDataHist('rdh_bkg', 'rdh_bkg', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_bkg))
    yield_bkg_ = rdh_bkg.sum(False)

    tmp = hist_bkg.Clone()
    tmp = tmp.Rebin(int(hist_bkg.GetNbinsX() / nBins))
    yMax_bkg = 1.5*tmp.GetMaximum()

    # construct background as 4th order Bernstein polynomial
    b0  = ROOT.RooRealVar('bern0', 'bern_coeff', 1, -2,  2)
    b1  = ROOT.RooRealVar('bern1', 'bern_coeff', 1, -10, 10)
    b2  = ROOT.RooRealVar('bern2', 'bern_coeff', 1, -10, 10)
    b3  = ROOT.RooRealVar('bern3', 'bern_coeff', 1, -10, 10)
    bkg = ROOT.RooBernsteinFast(3)('bkg', 'bkg', recoilmass, ROOT.RooArgList(b0, b1, b2))

    bkg_norm = ROOT.RooRealVar('bkg_norm_tmp', 'bkg_norm_tmp', yield_bkg_, 0, 1e6)
    bkg_fit  = ROOT.RooAddPdf('bkg_fit', '', ROOT.RooArgList(bkg), ROOT.RooArgList(bkg_norm))
    bkg_fit.fitTo(rdh_bkg, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(sumw2err))


    ########### PLOTTING ###########
    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : yMax_bkg,

        'xtitle'            : 'm_{recoil} [GeV]',
        'ytitle'            : 'Events',

        'topRight'          : topRight,
        'topLeft'           : topLeft,

        'ratiofraction'     : 0.3,
        'ytitleR'           : 'Pull',
        'yminR'             : -3.5,
        'ymaxR'             : 3.5,
    }

    plotter.cfg = cfg
    plot_fit_with_pull(
        rdh_bkg,
        bkg_fit,
        recoilmass,
        nBins,
        f'{outDir}/fit_bkg',
        label,
        title=None,
        fit_color=ROOT.kRed,
        save_pdf=True,
        param_layout=(0.5, 0.9, 0.9),
    )

    # import background parameterization to the workspace
    bkg_norm.setVal(yield_bkg_)  # not constant
    b0.setConstant(True)         # set as constant
    b1.setConstant(True)         # set as constant
    b2.setConstant(True)         # set as constant
    b3.setConstant(True)         # set as constant
    getattr(w_tmp, 'import')(bkg)
    getattr(w_tmp, 'import')(bkg_norm)

    return hist_bkg





def get_hist():
    pass







def setup_syst(proc, syst, mH=125.0):

    mH_label = f'{mH:.3f}'.replace('.', 'p')
    recoilmass = w_tmp.var('zll_recoil_m')
    MH = w_tmp.var('MH')

    cfg = {

        'logX':             False,
        'logY':             False,

        'xmin':             120,
        'xmax':             140,
        'ymin':             0,
        'ymax':             yMax,

        'xtitle':           'm_{recoil} [GeV]',
        'ytitle':           'Events',

        'topLeft':          topLeft,
        'topRight':         topRight,

        'ratiofraction':    0.3,
        'ytitleR':         'Pull',
        'yminR':           -3.5,
        'ymaxR':            3.5
    }

    MH.setVal(125.0)
    spline_mean     = w_tmp.obj('spline_mean')
    spline_sigma    = w_tmp.obj('spline_sigma')
    spline_mean_gt  = w_tmp.obj('spline_mean_gt')
    spline_sigma_gt = w_tmp.obj('spline_sigma_gt')
    spline_alpha_1  = w_tmp.obj('spline_alpha_1')
    spline_alpha_2  = w_tmp.obj('spline_alpha_2')
    spline_n_1      = w_tmp.obj('spline_n_1')
    spline_n_2      = w_tmp.obj('spline_n_2')
    spline_cb_1     = w_tmp.obj('spline_cb_1')
    spline_cb_2     = w_tmp.obj('spline_cb_2')

    mean_ud,  mean_gt_ud  = [], []
    sigma_ud, sigma_gt_ud = [], []

    for s in ['Up', 'Down']:
        suffix = f'{mH_label}_{syst}{s}'

        hist_zh = get_hist()
        hist_zh.Scale(lumiScale)
        hist_zh = hist_zh.ProjectionX(f'hist_zh_{suffix}', cat_idx_min, cat_idx_max)
        hist_zh.SetName(f'hist_zh_{suffix}')
        hist_zh.Scale(yield_nom / hist_zh.Integral())

        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{suffix}', 'rdh_zh', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.Sum(False)

        # Should define it in a syst dependent way
        mean     = ROOT.RooRealVar(f'mean_{suffix}',     '', spline_mean.getVal())
        sigma    = ROOT.RooRealVar(f'sigma_{suffix}',    '', spline_sigma.getVal())
        alpha_1  = ROOT.RooRealVar(f'alpha_1_{suffix}',  '', spline_alpha_1.getVal())
        alpha_2  = ROOT.RooRealVar(f'alpha_2_{suffix}',  '', spline_alpha_2.getVal())
        n_1      = ROOT.RooRealVar(f'n_1_{suffix}',      '', spline_n_1.getVal())
        n_2      = ROOT.RooRealVar(f'n_2_{suffix}',      '', spline_n_2.getVal())
        mean_gt  = ROOT.RooRealVar(f'mean_gt_{suffix}',  '', spline_mean_gt.getVal())
        sigma_gt = ROOT.RooRealVar(f'sigma_gt_{suffix}', '', spline_sigma_gt.getVal())
        cb_1     = ROOT.RooRealVar(f'cb_1_{suffix}',     '', spline_cb_1.getVal())
        cb_2     = ROOT.RooRealVar(f'cb_2_{suffix}',     '', spline_cb_2.getVal())

        cbs_1, cbs_2, gauss, sig, sig_norm, sig_fit = build_2cbg_pdf(
            recoilmass,
            suffix,
            mean, sigma,
            alpha_1, alpha_2,
            n_1, n_2,
            mean_gt, sigma_gt,
            cb_1, cb_2,
            yield_zh
        )
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTrue), ROOT.RooFit.SumW2Error(sumw2err))

        mean_ud.append(mean.getVal())
        mean_gt_ud.append(mean_gt.getVal())

        sigma_ud.append(sigma.getVal())
        sigma_gt_ud.append(sigma_gt.getVal())

        cfg['ymax'] = yMax
        plotter.cfg = cfg
        plot_fit_with_pull(
            rdh_zh,
            sig_fit,
            recoilmass,
            nBins,
            f'{outDir}/fit_mh{suffix}',
            label,
            'ZH signal',
            ROOT.kRed
        )

        # Import
        w_tmp.Import(rdh_zh)
        w_tmp.Import(sig_fit)

    # Plot all fitted signals
    cfg['ymax'] = 2.5 * yMax
    cfg['xmin'] = 124
    cfg['xmax'] = 127
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')

    plt    = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]

    for i, channel in enumerate([f'_{syst}Up', '', f'_{syst}Down']):
        sig_fit = w_tmp.pdf(f'zh_model_{mH_label}_{channel}')
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

    # Should continue later










def doBES():

    pct = 1 if ecm==240 else (10 if ecm==365 else 0)

    ## Only consider variation for 125 GeV
    ## Assume variations to be indentical for other mass points
    mH = 125.0
    mH_label = f'{mH:.3f}'.replace('.', 'p')

    recoilmass = w_tmp.var('zll_recoil_m')
    MH = w_tmp.var('MH')


    # recoil mass plot settings
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

    ## For BES, consider only variations in norm, mean and sigma
    ## Assume others to be identical to nominal sample
    MH.setVal(125.0)  # evaluate all at 125 GeV
    spline_alpha_1  = w_tmp.obj('spline_alpha_1')
    spline_alpha_2  = w_tmp.obj('spline_alpha_2')
    spline_n_1      = w_tmp.obj('spline_n_1')
    spline_n_2      = w_tmp.obj('spline_n_2')
    spline_cb_1     = w_tmp.obj('spline_cb_1')
    spline_cb_2     = w_tmp.obj('spline_cb_2')
    spline_mean_gt  = w_tmp.obj('spline_mean_gt')
    spline_mean     = w_tmp.obj('spline_mean')
    spline_sigma_gt = w_tmp.obj('spline_sigma_gt')
    spline_sigma    = w_tmp.obj('spline_sigma')

    sigma_ud, sigma_gt_ud = [], []

    for s in ['Up', 'Down']:

        if s == 'Up':   proc = f'wzp6_ee_{flavor}H_BES-higher-{pct}pc_ecm{ecm}'
        if s == 'Down': proc = f'wzp6_ee_{flavor}H_BES-lower-{pct}pc_ecm{ecm}'
        if ecm == 365 and flavor == 'ee':
            proc = 'wzp6_ee_eeH_ecm365'  # no BES for electrons at 365
        suffix = f'{mH_label}_BES{s}'

        hist_zh = getHist(f'{flavor}_{hName}', [proc])
        hist_zh.Scale(lumiScale)
        hist_zh = hist_zh.ProjectionX(f'hist_zh_{suffix}', cat_idx_min, cat_idx_max)
        hist_zh.SetName(f'hist_zh_{suffix}')
        hist_zh.Scale(yield_nom/hist_zh.Integral())
        rdh_zh   = ROOT.RooDataHist(f'rdh_zh_{suffix}', 'rdh_zh', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)


        mean     = ROOT.RooRealVar(f'mean_{suffix}',     '', spline_mean.getVal())
        sigma    = ROOT.RooRealVar(f'sigma_{suffix}',    '', spline_sigma.getVal(), 0, 5)  # float
        alpha_1  = ROOT.RooRealVar(f'alpha_1_{suffix}',  '', spline_alpha_1.getVal())
        alpha_2  = ROOT.RooRealVar(f'alpha_2_{suffix}',  '', spline_alpha_2.getVal())
        n_1      = ROOT.RooRealVar(f'n_1_{suffix}',      '', spline_n_1.getVal())
        n_2      = ROOT.RooRealVar(f'n_2_{suffix}',      '', spline_n_2.getVal())
        mean_gt  = ROOT.RooRealVar(f'mean_gt_{suffix}',  '', spline_mean_gt.getVal())
        sigma_gt = ROOT.RooRealVar(f'sigma_gt_{suffix}', '', spline_sigma_gt.getVal(), 0, 5)  # float
        cb_1     = ROOT.RooRealVar(f'cb_1_{suffix}',     '', spline_cb_1.getVal())
        cb_2     = ROOT.RooRealVar(f'cb_2_{suffix}',     '', spline_cb_2.getVal())

        cbs_1, cbs_2, gauss, sig, sig_norm, sig_fit = build_2cbg_pdf(
            recoilmass,
            suffix,
            mean, sigma,
            alpha_1, alpha_2,
            n_1, n_2,
            mean_gt, sigma_gt,
            cb_1, cb_2,
            yield_zh,
        )
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(sumw2err))

        sigma_ud.append(sigma.getVal())
        sigma_gt_ud.append(sigma_gt.getVal())

        # Do plotting
        cfg['ymax'] = yMax
        plotter.cfg = cfg
        plot_fit_with_pull(
            rdh_zh,
            sig_fit,
            recoilmass,
            nBins,
            f'{outDir}/fit_mH{suffix}',
            label,
            'ZH signal',
            ROOT.kRed,
        )

        # Import
        w_tmp.Import(rdh_zh)
        w_tmp.Import(sig_fit)

    # plot all fitted signals
    cfg['ymax'] = 2.5 * yMax
    cfg['xmin'] = 124
    cfg['xmax'] = 127
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy  = plotter.dummy()
    dummy.Draw('HIST')
    plt    = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]

    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}_BESUp')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[0]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))
    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[1]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))
    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}_BESDown')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[2]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}_BES.png')

    # Construct BES uncertainty
    # Nominals, w/o the BES uncertainty
    spline_mean  = w_tmp.obj('spline_mean')
    spline_sigma = w_tmp.obj('spline_sigma')
    MH.setVal(125.0)  # Evaluate all at 125 GeV
    sigma_nominal    = spline_sigma.getVal()
    sigma_gt_nominal = spline_sigma_gt.getVal()

    # Sigma param
    delta = 0.5 * abs(sigma_ud[0] - sigma_ud[1])
    sig_sigma_BES_ = delta / sigma_nominal  # 1 sigma value  such that (1 + bkg_norm_BES) * sigma_nominal = sigma_nominal + delta
    sig_sigma_BES  = ROOT.RooRealVar('sig_sigma_BES', 'sig_sigma_BES', sig_sigma_BES_)  # Constant
    getattr(w_tmp, 'import')(sig_sigma_BES)
    print(sigma_nominal, delta, sig_sigma_BES_)

    # Sigma_gt param
    delta = 0.5 * abs(sigma_gt_ud[0] - sigma_gt_ud[1])
    sig_sigma_gt_BES_ = delta / sigma_gt_nominal  # 1 sigma value  such that (1 + bkg_norm_BES) * sigma_nominal = sigma_nominal + delta
    sig_sigma_gt_BES  = ROOT.RooRealVar('sig_sigma_gt_BES', 'sig_sigma_gt_BES', sig_sigma_gt_BES_)  # Constant
    getattr(w_tmp, 'import')(sig_sigma_gt_BES)
    print(sigma_gt_nominal, delta, sig_sigma_BES_)



def doSQRTS():

    # scale_BES = ROOT.RooRealVar('scale_SQRTS', 'SQRTS scale parameter', 0, -1, 1)

    ## only consider variation for 125 GeV
    ## assume variations to be indentical for other mass points

    proc = f'wzp6_ee_{flavor}H_ecm{ecm}'
    mH = 125.0
    mH_label = f'{mH:.3f}'.replace('.', 'p')

    recoilmass = w_tmp.var('zll_recoil_m')
    MH = w_tmp.var('MH')

    # recoil mass plot settings
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


    ## for SQRTS, consider only variations in norm, CB mean and CB sigma
    ## assume others to be identical to nominal sample
    MH.setVal(125.0)  # evaluate all at 125 GeV
    spline_alpha_1  = w_tmp.obj('spline_alpha_1')
    spline_alpha_2  = w_tmp.obj('spline_alpha_2')
    spline_n_1      = w_tmp.obj('spline_n_1')
    spline_n_2      = w_tmp.obj('spline_n_2')
    spline_cb_1     = w_tmp.obj('spline_cb_1')
    spline_cb_2     = w_tmp.obj('spline_cb_2')
    spline_mean_gt  = w_tmp.obj('spline_mean_gt')
    spline_mean     = w_tmp.obj('spline_mean')
    spline_sigma_gt = w_tmp.obj('spline_sigma_gt')
    spline_sigma    = w_tmp.obj('spline_sigma')

    mean_ud, mean_gt_ud = [], []

    for s in ['Up', 'Down']:

        if s == 'Up':   s_ = 'up'
        if s == 'Down': s_ = 'dw'
        suffix = f'{mH_label}_SQRTS{s}'

        hist_zh = getHist(f'{flavor}_{hName}_sqrts{s_}', [proc])
        hist_zh.Scale(lumiScale)
        hist_zh = hist_zh.ProjectionX(f'hist_zh_{suffix}', cat_idx_min, cat_idx_max)
        hist_zh.SetName(f'hist_zh_{suffix}')  # Was named BES, should confirm with Jan
        hist_zh.Scale(yield_nom/hist_zh.Integral())
        rdh_zh   = ROOT.RooDataHist(f'rdh_zh_{suffix}', 'rdh_zh', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        mean     = ROOT.RooRealVar(f'mean_{suffix}',     '', spline_mean.getVal(), mH - 5, mH + 5)
        sigma    = ROOT.RooRealVar(f'sigma_{suffix}',    '', spline_sigma.getVal())
        alpha_1  = ROOT.RooRealVar(f'alpha_1_{suffix}',  '', spline_alpha_1.getVal())
        alpha_2  = ROOT.RooRealVar(f'alpha_2_{suffix}',  '', spline_alpha_2.getVal())
        n_1      = ROOT.RooRealVar(f'n_1_{suffix}',      '', spline_n_1.getVal())
        n_2      = ROOT.RooRealVar(f'n_2_{suffix}',      '', spline_n_2.getVal())
        mean_gt  = ROOT.RooRealVar(f'mean_gt_{suffix}',  '', spline_mean_gt.getVal(), mH - 5, mH + 5)
        sigma_gt = ROOT.RooRealVar(f'sigma_gt_{suffix}', '', spline_sigma_gt.getVal())

        cb_1  = ROOT.RooRealVar(f'cb_1_{suffix}', '', spline_cb_1.getVal())
        cb_2  = ROOT.RooRealVar(f'cb_2_{suffix}', '', spline_cb_2.getVal())

        cbs_1, cbs_2, gauss, sig, sig_norm, sig_fit = build_2cbg_pdf(
            recoilmass,
            suffix,
            mean, sigma,
            alpha_1, alpha_2,
            n_1, n_2,
            mean_gt, sigma_gt,
            cb_1, cb_2,
            yield_zh,
        )
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(sumw2err))

        mean_ud.append(mean.getVal())
        mean_gt_ud.append(mean_gt.getVal())

        # Do plotting
        cfg['ymax'] = yMax
        plotter.cfg = cfg

        plot_fit_with_pull(
            rdh_zh,
            sig_fit,
            recoilmass,
            nBins,
            f'{outDir}/fit_mH{suffix}',
            label,
            'ZH signal',
            ROOT.kRed,
        )

        # Import
        w_tmp.Import(rdh_zh)
        w_tmp.Import(sig_fit)


    # plot all fitted signals
    cfg['ymax'] = 2.5 * yMax
    cfg['xmin'] = 124
    cfg['xmax'] = 127
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()

    dummy.Draw('HIST')

    plt = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]

    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}_SQRTSUp')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[0]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))
    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[1]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))
    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}_SQRTSDown')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[2]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}_SQRTS.png')

    # Construct SQRTS uncertainty
    # Nominals, w/o the SQRTS uncertainty
    MH.setVal(125.0)  # evaluate all at 125 GeV
    mean_nominal     = spline_mean.getVal()
    mean_gt__nominal = spline_mean_gt.getVal()

    # Mean param
    delta = 0.5 * abs(mean_ud[0] - mean_ud[1])
    sig_mean_SQRTS_ = delta / mean_nominal  # 1 sigma value  such that (1 + bkg_norm_BES) * mean_nominal = mean_nominal + delta
    sig_mean_SQRTS  = ROOT.RooRealVar('sig_mean_SQRTS', 'sig_mean_SQRTS', sig_mean_SQRTS_)  # Constant
    getattr(w_tmp, 'import')(sig_mean_SQRTS)
    print(mean_nominal, delta, sig_mean_SQRTS_)


    # mean_gt param
    delta = 0.5 * abs(mean_gt_ud[0] - mean_gt_ud[1])
    sig_mean_gt_SQRTS_ = (delta)/mean_gt__nominal  # 1 sigma value  such that (1+bkg_norm_BES)*mean__nominal = mean__nominal+delta
    sig_mean_gt_SQRTS  = ROOT.RooRealVar('sig_mean_gt_SQRTS', 'sig_mean_gt_SQRTS', sig_mean_gt_SQRTS_)  # constant
    getattr(w_tmp, 'import')(sig_mean_gt_SQRTS)
    print(mean_gt__nominal, delta, sig_mean_gt_SQRTS_)



def doLEPSCALE():

    # scale_BES = ROOT.RooRealVar('scale_LEPSCALE', 'LEPSCALE scale parameter', 0, -1, 1)

    ## only consider variation for 125 GeV
    ## assume variations to be indentical for other mass points

    proc = f'wzp6_ee_{flavor}H_ecm{ecm}'
    mH = 125.0
    mH_label = f'{mH:.3f}'.replace('.', 'p')

    recoilmass = w_tmp.var('zll_recoil_m')
    MH = w_tmp.var('MH')

    # recoil mass plot settings
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


    ## For LEPSCALE, consider only variations in norm, CB mean and CB sigma
    ## Assume others to be identical to nominal sample
    MH.setVal(125.0)  # evaluate all at 125 GeV
    spline_alpha_1  = w_tmp.obj('spline_alpha_1')
    spline_alpha_2  = w_tmp.obj('spline_alpha_2')
    spline_n_1      = w_tmp.obj('spline_n_1')
    spline_n_2      = w_tmp.obj('spline_n_2')
    spline_cb_1     = w_tmp.obj('spline_cb_1')
    spline_cb_2     = w_tmp.obj('spline_cb_2')
    spline_mean_gt  = w_tmp.obj('spline_mean_gt')
    spline_mean     = w_tmp.obj('spline_mean')
    spline_sigma_gt = w_tmp.obj('spline_sigma_gt')
    spline_sigma    = w_tmp.obj('spline_sigma')

    mean_ud = []

    for s in ['Up', 'Down']:

        if s == 'Up':   s_ = 'up'
        if s == 'Down': s_ = 'dw'
        suffix = f'{mH_label}_LEPSCALE{s}'

        hist_zh = getHist(f'{flavor}_{hName}_scale{s_}', [proc])
        hist_zh.Scale(lumiScale)
        hist_zh = hist_zh.ProjectionX(f'hist_zh_{suffix}', cat_idx_min, cat_idx_max)
        hist_zh.SetName(f'hist_zh_{suffix}')
        hist_zh.Scale(yield_nom/hist_zh.Integral())
        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{suffix}', 'rdh_zh', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        mean     = ROOT.RooRealVar(f'mean_{suffix}',     '', spline_mean_gt.getVal(), mH - 5, mH + 5)
        sigma    = ROOT.RooRealVar(f'sigma_{suffix}',    '', spline_sigma.getVal())
        alpha_1  = ROOT.RooRealVar(f'alpha_1_{suffix}',  '', spline_alpha_1.getVal())
        alpha_2  = ROOT.RooRealVar(f'alpha_2_{suffix}',  '', spline_alpha_2.getVal())
        n_1      = ROOT.RooRealVar(f'n_1_{suffix}',      '', spline_n_1.getVal())
        n_2      = ROOT.RooRealVar(f'n_2_{suffix}',      '', spline_n_2.getVal())
        mean_gt  = ROOT.RooRealVar(f'mean_gt_{suffix}',  '', spline_mean_gt.getVal())
        sigma_gt = ROOT.RooRealVar(f'sigma_gt_{suffix}', '', spline_sigma_gt.getVal())

        cb_1  = ROOT.RooRealVar(f'cb_1_{suffix}', '', spline_cb_1.getVal())
        cb_2  = ROOT.RooRealVar(f'cb_2_{suffix}', '', spline_cb_2.getVal())

        cbs_1, cbs_2, gauss, sig, sig_norm, sig_fit = build_2cbg_pdf(
            recoilmass,
            suffix,
            mean, sigma,
            alpha_1, alpha_2,
            n_1, n_2,
            mean_gt, sigma_gt,
            cb_1, cb_2,
            yield_zh,
        )
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(sumw2err))

        mean_ud.append(mean.getVal())

        # Do plotting
        cfg['ymax'] = yMax
        plotter.cfg = cfg

        plot_fit_with_pull(
            rdh_zh,
            sig_fit,
            recoilmass,
            nBins,
            f'{outDir}/fit_mH{mH_label}_LEPSCALE{s}',
            label,
            title='ZH signal',
            fit_color=ROOT.kRed,
        )

        # Import
        w_tmp.Import(rdh_zh)
        w_tmp.Import(sig_fit)


    # plot all fitted signals
    cfg['ymax'] = 2.5 * yMax
    cfg['xmin'] = 124
    cfg['xmax'] = 127
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()

    dummy.Draw('HIST')

    plt    = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]

    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}_LEPSCALEUp')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[0]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[1]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    sig_fit = w_tmp.pdf(f'zh_model_{mH_label}_LEPSCALEDown')
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(colors[2]), ROOT.RooFit.Normalization(yield_nom, ROOT.RooAbsReal.NumEvent))

    plt.Draw('SAME')

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}_LEPSCALE.png')

    # Construct LEPSCALE uncertainty
    # Nominals, w/o the LEPSCALE uncertainty
    spline_mean  = w_tmp.obj('spline_mean')
    spline_sigma = w_tmp.obj('spline_sigma')
    MH.setVal(125.0)  # Evaluate all at 125 GeV
    mean_nominal = spline_mean.getVal()

    # Mean param
    delta = 0.5 * abs(mean_ud[0] - mean_ud[1])
    sig_mean_LEPSCALE_ = delta / mean_nominal  # 1 sigma value  such that (1 + bkg_norm_BES) * mean_nominal = mean_nominal + delta
    sig_mean_LEPSCALE  = ROOT.RooRealVar('sig_mean_LEPSCALE', 'sig_mean_LEPSCALE', sig_mean_LEPSCALE_)  # constant
    getattr(w_tmp, 'import')(sig_mean_LEPSCALE)
    print(mean_nominal, delta, sig_mean_LEPSCALE_)


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--flavor',    type=str,   help='Flavor (mumu or ee)', default='mumu')
    parser.add_argument('--mode',      type=str,   help='Detector mode', choices=['IDEA', 'IDEA_MC', 'IDEA_3T', 'CLD', 'CLD_FullSim', 'IDEA_noBES', 'IDEA_2E', 'IDEA_BES6pct'], default='IDEA')
    parser.add_argument('--cat',       type=int,   help='Category (0, 1, 2 or 3)', choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--lumi',      type=float, help='Luminosity scale', default=1.0)
    parser.add_argument('--ecm',       type=str,   help='Center-of-mass', choices=['240', '365'],     default='240')
    parser.add_argument('--tag',       type=str,   help='Analysis tag for versioning, optional',      default='')
    parser.add_argument('--recoilMin', type=int,   help='Lower mass value for the fit (default 120)', default=120)
    parser.add_argument('--recoilMax', type=int,   help='Upper mass value for the fit (default 140)', default=140)
    parser.add_argument('--nBins',     type=int,   help='Number of bins for the plots (default 250)', default=250)
    parser.add_argument('--doSyst',    type=bool,  help='Include systematic uncertainties in th fit (default True)', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()


    sumw2err = ROOT.kTRUE

    ROOT.Math.MinimizerOptions.SetDefaultMinimizer('Fumili2')
    # ROOT.Math.MinimizerOptions.SetMinimizerAlgorithm('Simplex') # Migrad Minimize Simplex Fumili2
    ROOT.Math.MinimizerOptions.PrintDefault('Minuit2')
    ROOT.Math.MinimizerOptions.SetDefaultPrecision(1e-15)
    ROOT.Math.MinimizerOptions.SetDefaultMaxIterations(200)
    # ROOT.Math.MinimizerOptions.PrintDefault()


    flavor, ecm, sel = args.flavor, args.ecm, args.sel
    cat, tag, mode = args.cat, args.tag, args.mode
    flavorLabel = '#mu^{#plus}#mu^{#minus}' if flavor == 'mumu' else 'e^{#plus}e^{#minus}'
    lumiScale = args.lumi

    # re-normalize lumis, by default histograms are scaled to 10.8 and 3.12 ab-1
    if ecm == 240:
        lumiScale /= 10.8
    elif ecm == 365:
        lumiScale /= 3.12
    else:
        raise ValueError(f'{ecm = } is not supported, choose between [240, 365]')
    lumiStr   = str(int(args.lumi)) if args.lumi.is_integer() else str(args.lumi)
    lumiLabel = lumiStr.replace('.', 'p')

    topRight = f'#sqrt{{s}} = {ecm} GeV, {lumiStr} ab^{{-1}}'
    topLeft  = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'
    label    = f'{flavorLabel}, category {cat}'
    inDir    = loc.get('HIST_PROCESSED', flavor, ecm, sel)
    outDir   = loc.get('DATACARD',       flavor, ecm, sel)
    runDir   = loc.get('RUNDIR',         flavor, ecm, sel)

    outDir.mkdir(exist_ok=True, parents=True)
    runDir.mkdir(exist_ok=True, parents=True)
    subprocess.run(['cp', '/home/submit/jaeyserm/public_html/fccee/h_mass/index.php', f'{outDir}'])

    parId = f'{flavor}_cat{cat}_ecm{ecm}'
    hName = 'zll_recoil_m_cat'
    if cat == 0: cat_idx_min, cat_idx_max = 0,   5
    else:        cat_idx_min, cat_idx_max = cat, cat

    nBins = args.nBins  # total number of bins, for plotting
    recoilMin, recoilMax = args.recoilMin, args.recoilMax
    h_obs = None  # should hold the data_obs = sum of signal and backgrounds

    recoilmass = ROOT.RooRealVar('zll_recoil_m', 'm_{recoil} [GeV]', 125, recoilMin, recoilMax)
    MH = ROOT.RooRealVar('MH', 'Higgs mass [GeV]', 125, 124.95, 125.05)

    pdf_sigs = []

    # define temporary output workspace
    w     = ROOT.RooWorkspace('w',     'workspace')  # final workspace for combine
    w_tmp = ROOT.RooWorkspace('w_tmp', 'workspace')

    w_tmp.Import(recoilmass)
    w_tmp.Import(MH)

    hist_sig = doSignal()
    hist_bkg = doBackgrounds()

    # Build signal model
    spline_mean     = w_tmp.obj('spline_mean')
    spline_sigma    = w_tmp.obj('spline_sigma')
    spline_yield    = w_tmp.obj('spline_yield')
    spline_alpha_1  = w_tmp.obj('spline_alpha_1')
    spline_alpha_2  = w_tmp.obj('spline_alpha_2')
    spline_n_1      = w_tmp.obj('spline_n_1')
    spline_n_2      = w_tmp.obj('spline_n_2')
    spline_cb_1     = w_tmp.obj('spline_cb_1')
    spline_cb_2     = w_tmp.obj('spline_cb_2')
    spline_mean_gt  = w_tmp.obj('spline_mean_gt')
    spline_sigma_gt = w_tmp.obj('spline_sigma_gt')

    sig_alpha_1 = ROOT.RooFormulaVar('sig_alpha_1', '@0', ROOT.RooArgList(spline_alpha_1))
    sig_alpha_2 = ROOT.RooFormulaVar('sig_alpha_2', '@0', ROOT.RooArgList(spline_alpha_2))
    sig_n_1     = ROOT.RooFormulaVar('sig_n_1',     '@0', ROOT.RooArgList(spline_n_1))
    sig_n_2     = ROOT.RooFormulaVar('sig_n_2',     '@0', ROOT.RooArgList(spline_n_2))
    sig_cb_1    = ROOT.RooFormulaVar('sig_cb_1',    '@0', ROOT.RooArgList(spline_cb_1))
    sig_cb_2    = ROOT.RooFormulaVar('sig_cb_2',    '@0', ROOT.RooArgList(spline_cb_2))
    sig_norm    = ROOT.RooFormulaVar('sig_norm',    '@0', ROOT.RooArgList(spline_yield))

    if args.doSyst:

        doBES()  # 1 or 6 pct BES variation
        doSQRTS()
        doLEPSCALE()

        # Systematic strenghts
        flav = 'MU' if flavor=='mumu' else ('EL' if flavor=='ee' else flavor)
        ISR      = ROOT.RooRealVar(f'ISR_ecm{ecm}',             'ISR',      0)         # ISR      uncertainty parameter
        BES      = ROOT.RooRealVar(f'BES_ecm{ecm}',             'BES',      0, -5, 5)  # BES      uncertainty parameter
        SQRTS    = ROOT.RooRealVar(f'SQRTS_ecm{ecm}',           'SQRTS',    0, -5, 5)  # SQRTS    uncertainty parameter
        LEPSCALE = ROOT.RooRealVar(f'LEPSCALE_{flav}_ecm{ecm}', 'LEPSCALE', 0, -5, 5)  # LEPSCALE uncertainty parameter

        # BES
        sigma_BES    = w_tmp.obj('sig_sigma_BES')
        sigma_gt_BES = w_tmp.obj('sig_sigma_gt_BES')

        # SQRTS
        mean_SQRTS    = w_tmp.obj('sig_mean_SQRTS')
        mean_gt_SQRTS = w_tmp.obj('sig_mean_gt_SQRTS')

        # LEPSCALE
        mean_LEPSCALE = w_tmp.obj('sig_mean_LEPSCALE')


        sig_mean     = ROOT.RooFormulaVar('sig_mean',     '@0*(1+@1*@2)*(1+@3*@4)', ROOT.RooArgList(spline_mean,     LEPSCALE, mean_LEPSCALE, SQRTS, mean_SQRTS))
        sig_sigma    = ROOT.RooFormulaVar('sig_sigma',    '@0*(1+@1*@2)',           ROOT.RooArgList(spline_sigma,    BES,      sigma_BES))
        sig_mean_gt  = ROOT.RooFormulaVar('sig_mean_gt',  '@0*(1+@1*@2)',           ROOT.RooArgList(spline_mean_gt,  SQRTS,    mean_SQRTS))
        sig_sigma_gt = ROOT.RooFormulaVar('sig_sigma_gt', '@0*(1+@1*@2)*(1+@3*@4)', ROOT.RooArgList(spline_sigma_gt, BES,      sigma_gt_BES,  SQRTS, mean_gt_SQRTS))

    else:
        sig_mean     = ROOT.RooFormulaVar('sig_mean',     '@0', ROOT.RooArgList(spline_mean))
        sig_sigma    = ROOT.RooFormulaVar('sig_sigma',    '@0', ROOT.RooArgList(spline_sigma))
        sig_mean_gt  = ROOT.RooFormulaVar('sig_mean_gt',  '@0', ROOT.RooArgList(spline_mean_gt))
        sig_sigma_gt = ROOT.RooFormulaVar('sig_sigma_gt', '@0', ROOT.RooArgList(spline_sigma_gt))

    # construct final signal pdf
    cbs_1 = ROOT.RooCBShape('CrystallBall_1', 'CrystallBall_1', recoilmass, sig_mean, sig_sigma, sig_alpha_1, sig_n_1)
    cbs_2 = ROOT.RooCBShape('CrystallBall_2', 'CrystallBall_2', recoilmass, sig_mean, sig_sigma, sig_alpha_2, sig_n_2)
    gauss = ROOT.RooGaussian('gauss', 'gauss', recoilmass, sig_mean_gt, sig_sigma_gt)
    sig   = ROOT.RooAddPdf('sig', 'sig', ROOT.RooArgList(cbs_1, cbs_2, gauss), ROOT.RooArgList(sig_cb_1, sig_cb_2))

    w.Import(sig)

    # Construct background model
    bkg_yield = w_tmp.obj('bkg_norm_tmp').getVal()
    bkg_norm  = ROOT.RooRealVar('bkg_norm', 'bkg_norm', bkg_yield)  # Nominal background yield (automatically done by Combine with pdfName_norm, floating)
    bkg_norm.setVal(bkg_yield)  # Not constant!
    bkg = w_tmp.obj('bkg')
    w.Import(bkg, ROOT.RooFit.RenameAllVariablesExcept(parId, 'zll_recoil_m'))

    data_obs = ROOT.RooDataHist('data_obs', 'data_obs', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(h_obs))
    w.Import(data_obs)

    w.writeToFile(f'{runDir}/datacard.root')
    w.Print()

    poi = ROOT.RooArgSet(MH)
    w.defineSet('POI', poi)

    del w, w_tmp

    if   ecm == 240 and flavor == 'mumu': bkg_id = 1
    elif ecm == 240 and flavor == 'ee':   bkg_id = 2
    elif ecm == 365 and flavor == 'mumu': bkg_id = 3
    elif ecm == 365 and flavor == 'ee':   bkg_id = 4

    # Make datacard
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/datacard_template.txt', 'r') as file:
        dc = file.read()
        dc = dc.replace('$rate_sig', f'{yield_nom}')
        dc = dc.replace('$rate_bkg', f'{bkg_yield}')
        dc = dc.replace('$flavor',   f'{flavor}')
        dc = dc.replace('$ecm',      f'{ecm}')
        dc = dc.replace('$cat',      f'{cat}')
        dc = dc.replace('$bkg_id',   f'{bkg_id}')

    with open(f'{runDir}/datacard.txt', 'w') as file:
        file.write(dc)

    # remove lepton scale uncertainty for other flavor
    if flavor == 'mumu':
        cmd = "sed -i '/LEPSCALE_EL/d' datacard.txt"
    else:
        cmd = "sed -i '/LEPSCALE_MU/d' datacard.txt"
    subprocess.call(cmd, shell=True, cwd=runDir)

    # Build the Combine workspace based on the datacard, save it to ws.root
    subprocess.call(['text2workspace.py', 'datacard.txt' '-o', 'ws.root', '-v', '10', '--X-allow-no-background'], shell=True, cwd=runDir)
