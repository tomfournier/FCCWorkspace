
import array, argparse, subprocess,  ROOT

import package.plots.root.plotter as plotter
from package.tools.process import getHist
from package.userConfig import loc
loc.set_default_type('Path')

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)



parser = argparse.ArgumentParser()
parser.add_argument('--flavor', type=str, help='Flavor (mumu or ee)', default='mumu')
parser.add_argument('--mode', type=str, help='Detector mode', choices=['IDEA', 'IDEA_MC', 'IDEA_3T', 'CLD', 'CLD_FullSim',
                                                                       'IDEA_noBES', 'IDEA_2E', 'IDEA_BES6pct'], default='IDEA')
parser.add_argument('--cat', type=int, help='Category (0, 1, 2 or 3) (default 0)', choices=[0, 1, 2, 3], default=0)
parser.add_argument('--ecm', type=int, help='Center-of-mass', choices=[240, 365], default=240)
parser.add_argument('--sel', type=str, help='Selection', default='Baseline')
parser.add_argument('--tag', type=str, help='Analysis tag for versioning, optional', default='')
parser.add_argument('--recoilMin', type=int, help='Lower mass limit for fit (default 120)',    default=120)
parser.add_argument('--recoilMax', type=int, help='Upper mass limit for fit (default 140)',    default=140)
parser.add_argument('--nBins',     type=int, help='Number of bins for plotting (default 250)', default=250)
args = parser.parse_args()


config = {}


def apply_mode_suffix(proc: str, mode: str, flavor: str) -> str:
    if mode == 'IDEA_3T':
        return f'{proc}_3T'
    if mode == 'CLD':
        return f'{proc}_CLD'
    if mode == 'CLD_FullSim':
        proc = f'{proc}_CLD_FullSim'
        return proc.replace('-50MeV', '').replace('mumuH_mH', 'mumuH-mH').replace('eeH_mH', 'eeH-mH')
    if mode == 'IDEA_noBES':
        return proc.replace('_ecm240', '_noBES_ecm240')
    if mode == 'IDEA_2E' and flavor == 'ee':
        return f'{proc}_E2'
    return proc










def plot_spline_scan(
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
        marker_color=ROOT.kBlack):

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



def plot_fit(
        recoilmass,
        rdh_zh,
        sig_fit,
        mH_label: str,
        yMax
         ):

    # recoil mass plot settings
    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : yMax,

        'xtitle'            : 'm_{recoil} (GeV)',
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



def plot_decomposition(
        cbs_1,
        cbs_2,
        gauss,
        sig_fit,
        cb1_val,
        cb2_val,
        yield_zh,
        mH_label: str,
        yMax
         ):

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : yMax,

        'xtitle'            : 'm_{rec} (GeV)',
        'ytitle'            : 'Events',

        'topRight'          : topRight,
        'topLeft'           : topLeft,

        'ratiofraction'     : 0.3,
        'ytitleR'           : 'Pull',
        'yminR'             : -3.5,
        'ymaxR'             : 3.5,
    }

    cfg['ymax'] = 3 * yMax
    plotter.cfg = cfg
    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')
    plt = w_tmp.var('zll_recoil_m').frame()

    leg = ROOT.TLegend(.50, 0.7, .95, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)
    leg.SetMargin(0.15)

    cbs_1.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kRed),   ROOT.RooFit.Normalization(cb1_val*yield_zh,             ROOT.RooAbsReal.NumEvent))
    cbs_2.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kBlue),  ROOT.RooFit.Normalization(cb2_val*yield_zh,             ROOT.RooAbsReal.NumEvent))
    gauss.plotOn(plt,   ROOT.RooFit.LineColor(ROOT.kCyan),  ROOT.RooFit.Normalization((1-cb1_val-cb2_val)*yield_zh, ROOT.RooAbsReal.NumEvent))
    sig_fit.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlack), ROOT.RooFit.Normalization(yield_zh,                     ROOT.RooAbsReal.NumEvent))

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
    canvas.Modify()
    canvas.Update()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/fit_mH{mH_label}_decomposition.png')



def plot_fit_all(
        mHs,
        yield_zh,
        yMax
         ):

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 120,
        'xmax'              : 140,
        'ymin'              : 0,
        'ymax'              : yMax,

        'xtitle'            : 'm_{rec} (GeV)',
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
    cfg['ymax'] = yMax*2.5
    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()
    dummy.Draw('HIST')

    plt = w_tmp.var('zll_recoil_m').frame()
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack, ROOT.kGreen, ROOT.kCyan]
    for i, mH in enumerate(mHs):
        mH_ = f'{mH:.3f}'.replace('.', 'p')
        sig_fit = w_tmp.pdf('zh_model_%s' % mH_)
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











def make_params(flavor: str, ecm: int, cat: str, mH: float | int, config: dict[int, dict[str, dict[str, dict[str, list[float | int]]]]]):
    conf = config[ecm][flavor][cat]

    mean_cb0 = ROOT.RooRealVar(f'mean_cb0_{mH}',     '', mH)                 # Slope
    mean_cb1 = ROOT.RooRealVar(f'mean_cb1_{mH}',     '', *conf['mean_cb1'])  # Offset
    mean_cb  = ROOT.RooRealVar(f'mean_cb_{mH}', '@0+@1', ROOT.RooArgList(mean_cb0, mean_cb1))

    sigma_cb = ROOT.RooRealVar(f'sigma_cb_{mH}', '', *conf['sigma_cb'])
    sigma_gt = ROOT.RooRealVar(f'sigma_gt_{mH}', '', *conf['sigma_gt'])

    alpha_1 = ROOT.RooRealVar(f'alpha_1_{mH}', '', *conf['alpha_1'])
    alpha_2 = ROOT.RooRealVar(f'alpha_2_{mH}', '', *conf['alpha_2'])
    n_1 = ROOT.RooRealVar(f'n_1_{mH}', '', *conf['n_1'])
    n_2 = ROOT.RooRealVar(f'n_2_{mH}', '', *conf['n_2'])

    mean_gt1 = ROOT.RooRealVar(f'mean_gt1_{mH}',        '', *conf['mean_gt1'])  # Offset
    mean_gt  = ROOT.RooFormulaVar(f'mean_gt_{mH}', '@0+@1', ROOT.RooArgList(mean_cb, mean_gt1))

    cb_1 = ROOT.RooRealVar(f'cb_1_{mH}', '', *conf['cb_1'])
    cb_2 = ROOT.RooRealVar(f'cb_2_{mH}', '', *conf['cb_2'])

    params = {
        'mean_cb0':  mean_cb0, 'mean_cb1':  mean_cb1, 'mean_cb':   mean_cb,
        'sigma_cb':  sigma_cb, 'sigma_gt':  sigma_gt,
        'alpha_1':   alpha_1,  'alpha_2':   alpha_2,
        'n_1':       n_1,      'n_2':       n_2,
        'mean_gt1':  mean_gt1, 'mean_gt':   mean_gt,
        'cb_1':      cb_1,     'cb_2':      cb_2
    }

    return params



def doSignal(normYields = True):

    global h_obs

    mHs = [124.95, 125.0, 125.05]
    procs = [f'wzp6_ee_{flavor}H_ecm{ecm}', f'wzp6_ee_{flavor}H_mH-lower-50MeV_ecm{ecm}', f'wzp6_ee_{flavor}H_mH-higher-50MeV_ecm{ecm}']
    recoilmass = w_tmp.var('zll_recoil_m')
    val_mh = []
    val_yield, val_mean_cb0, val_mean_cb1, val_mean_gt, val_mean_gt1, val_sigma_cb, val_sigma_gt, val_alpha_1, val_alpha_2, val_n_1, val_n_2, val_cb_1, val_cb_2 = [], [], [], [], [], [], [], [], [], [], [], [], []
    err_yield, err_mean_cb0, err_mean_cb1, err_mean_gt, err_mean_gt1, err_sigma_cb, err_sigma_gt, err_alpha_1, err_alpha_2, err_n_1, err_n_2, err_cb_1, err_cb_2 = [], [], [], [], [], [], [], [], [], [], [], [], []

    hist_norm  = getHist(f'{flavor}_{hName}', [procs[0]])
    hist_norm  = hist_norm.ProjectionX('hist_zh_norm', cat_idx_min, cat_idx_max)
    yield_norm = hist_norm.Integral()

    tmp = hist_norm.Clone()
    print(hist_norm.GetNbinsX() / nBins)
    tmp  = tmp.Rebin(int(hist_norm.GetNbinsX() / nBins))
    yMax = tmp.GetMaximum()


    for i, proc in enumerate(procs):
        proc = apply_mode_suffix(proc, mode, flavor)

        mH = mHs[i]
        mH_label = f'{mH:.3f}'.replace('.', 'p')
        print(f'Do {mH = :.3f}')

        hist_zh = getHist(f'{flavor}_{hName}', [proc])
        hist_zh = hist_zh.ProjectionX(f'hist_zh_{mH_label}', cat_idx_min, cat_idx_max)
        if normYields: hist_zh.Scale(yield_norm/hist_zh.Integral())
        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{mH_label}', 'rdh_zh', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)
        if mH == 125.0 and h_obs is None: h_obs = hist_zh.Clone('h_obs')  # take 125.0 GeV to add to observed (need to add background later as well)



        params = make_params(flavor, ecm, cat, mH_label, config)
        mean_slope  = params['mean_slope']
        mean_offset = params['mean_offset']
        mean_cb     = params['mean_cb']

        sigma_cb = params['sigma_cb']
        sigma_gt = params['sigma_gt']

        alpha_1 = params['alpha_1']
        alpha_2 = params['alpha_2']

        n_1 = params['n_1']
        n_2 = params['n_2']

        mean_gt_offset = params['mean_gt_offset']
        mean_gt = params['mean_gt']

        cb_1 = params['cb_1']
        cb_2 = params['cb_2']


        # construct the 2CBG and perform the fit: pdf = cb_1*cbs_1 + cb_2*cbs_2 + gauss (cb_1 and cb_2 are the fractions, floating)
        cbs_1 = ROOT.RooCBShape(f'CrystallBall_1_{mH_label}', 'CrystallBall_1', recoilmass, mean_cb, sigma_cb, alpha_1, n_1)  # 1st CrystallBall
        cbs_2 = ROOT.RooCBShape(f'CrystallBall_2_{mH_label}', 'CrystallBall_2', recoilmass, mean_cb, sigma_cb, alpha_2, n_2)  # 2nd CrystallBall
        gauss = ROOT.RooGaussian(f'gauss_{mH_label}', 'gauss', recoilmass, mean_gt, sigma_gt)  # Gaussian

        sig      = ROOT.RooAddPdf(f'sig_{mH_label}',       '', ROOT.RooArgList(cbs_1, cbs_2, gauss), ROOT.RooArgList(cb_1, cb_2))  # half of both CB functions
        sig_norm = ROOT.RooRealVar(f'sig_{mH_label}_norm', '', yield_zh, 0, 1e8)  # Fix normalization
        sig_fit  = ROOT.RooAddPdf(f'zh_model_{mH_label}',  '', ROOT.RooArgList(sig), ROOT.RooArgList(sig_norm))
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(sumw2err))

        cb1_val = cb_1.getVal()
        cb2_val = cb_2.getVal()

        plot_fit(recoilmass, rdh_zh, sig_fit, mH_label, yMax)
        plot_decomposition(cbs_1, cbs_2, gauss, sig_fit,
                           cb1_val, cb2_val, yield_zh, mH_label, yMax)

        # import
        getattr(w_tmp, 'import')(rdh_zh)
        getattr(w_tmp, 'import')(sig_fit)

        val_mh.append(mH)

        val_mean_cb0.append(mean_slope.getVal())
        val_mean_cb1.append(mean_offset.getVal())
        val_sigma_cb.append(sigma_cb.getVal())
        val_mean_gt.append(mean_gt.getVal())
        val_mean_gt1.append(mean_gt_offset.getVal())
        val_sigma_gt.append(sigma_gt.getVal())
        val_alpha_1.append(alpha_1.getVal())
        val_alpha_2.append(alpha_2.getVal())
        val_n_1.append(n_1.getVal())
        val_n_2.append(n_2.getVal())
        val_yield.append(sig_norm.getVal())
        val_cb_1.append(cb_1.getVal())
        val_cb_2.append(cb_2.getVal())

        err_mean_cb0.append(mean_slope.getError())
        err_mean_cb1.append(mean_offset.getError())
        err_sigma_cb.append(sigma_cb.getError())
        err_mean_gt.append(0)
        err_mean_gt1.append(mean_gt_offset.getError())
        err_sigma_gt.append(sigma_cb.getError())
        err_alpha_1.append(alpha_1.getError())
        err_alpha_2.append(alpha_2.getError())
        err_n_1.append(n_1.getError())
        err_n_2.append(n_2.getError())
        err_yield.append(sig_norm.getError())
        err_cb_1.append(cb_1.getError())
        err_cb_2.append(cb_2.getError())

    plot_fit_all(mHs, yield_zh, yMax)

    # Export values
    idx = 1  # take values at central mass 125 GeV (not average using np.average(param_mean_offset))
    values = [1.0, val_mean_cb1[idx], 1.0, val_mean_cb1[idx], val_mean_gt1[idx], val_sigma_cb[idx],
              val_sigma_gt[idx], val_alpha_1[idx], val_alpha_2[idx], val_n_1[idx], val_n_2[idx],
              val_cb_1[idx], val_cb_2[idx]]
    with open(f'{outDir}/coeff.txt', 'w') as fOut:
        msg = '\n'.join(str(v) for v in values)
        fOut.write(msg)


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':

    sumw2err = ROOT.kTRUE

    ROOT.Math.MinimizerOptions.SetDefaultMinimizer('Fumili2')
    ROOT.Math.MinimizerOptions.PrintDefault('Minuit2')
    ROOT.Math.MinimizerOptions.SetDefaultPrecision(1e-15)
    ROOT.Math.MinimizerOptions.SetDefaultMaxIterations(200)

    mode = args.mode  # detector mode
    flavor, ecm, sel, cat, tag = args.flavor, args.ecm, args.sel, args.cat, args.tag
    flavorLabel = '#mu^{#plus}#mu^{#minus}' if flavor == 'mumu' else 'e^{#plus}e^{#minus}'

    topRight = f'#sqrt{{s}} = {ecm} GeV, 1 ab^{{#minus1}}'
    topLeft  = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'
    label  = f'{flavorLabel}, category {cat}'
    inDir  = loc.get('HIST_PROCESSED', flavor, ecm, sel)
    outDir = loc.get('PARAMETRIC', flavor, ecm, sel)

    outDir.mkdir(exist_ok=True, parents=True)
    subprocess.run(['cp', '/home/submit/jaeyserm/public_html/fccee/h_mass/index.php', f'{outDir}'])

    hName = 'zll_recoil_m_cat'

    if cat == 0: cat_idx_min, cat_idx_max = 0, 5
    else: cat_idx_min, cat_idx_max = cat, cat

    nBins = args.nBins  # total number of bins, for plotting
    h_obs = None  # should hold the data_obs = sum of signal and backgrounds

    recoilmass = ROOT.RooRealVar('zll_recoil_m', 'Recoil mass (GeV)', 125, args.recoilMin, args.recoilMax)
    MH = ROOT.RooRealVar('MH', 'Higgs mass (GeV)', 125, 124.95, 125.05)  # name Higgs mass as MH to be compatible with combine

    # define temporary output workspace
    w     = ROOT.RooWorkspace('w',     'workspace')  # final workspace for combine
    w_tmp = ROOT.RooWorkspace('w_tmp', 'workspace')

    getattr(w_tmp, 'import')(recoilmass)
    getattr(w_tmp, 'import')(MH)

    doSignal()

    # delete workspaces to avoid segfault
    del w, w_tmp
