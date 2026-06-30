
import os, sys, math, array, argparse, subprocess, ROOT

sys.path.insert(0, f'{os.path.dirname(os.path.realpath(__file__))}/../../../python')
import package.plots.root.plotter as plotter

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='Detector mode', choices=['IDEA', 'IDEA_MC', 'IDEA_3T', 'CLD', 'CLD_FullSim',
                                                                       'IDEA_noBES', 'IDEA_2E', 'IDEA_BES6pct'], default='IDEA')
parser.add_argument('--lumi', type=float, help='Luminosity scale', default=-1.0)
parser.add_argument('--statOnly', action='store_true', help='Stat-only fit')
parser.add_argument('--ecm', type=int, help='Center-of-mass', choices=[-1, 240, 365], default=-1)
parser.add_argument('--tag', type=str, help='Analysis tag', default='')
parser.add_argument('--combination', action='store_true', help='Do full combination')
args = parser.parse_args()

def findCrossing(
        xv: list[float | int],
        yv: list[float | int],
        left: bool = True,
        flip: float | int = 125,
        cross: float | int = 1.):

    closestPoint, idx = 1e9, -1
    for i in range(0, len(xv)):
        if     left and xv[i] > flip: continue
        if not left and xv[i] < flip: continue
        dy = abs(yv[i] - cross)
        if dy < closestPoint:
            closestPoint = dy
            idx = i
    # Find correct indices around crossing
    if left:
        if yv[idx] > cross: idx_ = idx + 1
        else: idx_ = idx - 1
    else:
        if yv[idx] > cross: idx_ = idx -1
        else: idx_ = idx + 1

    # Do interpolation
    omega = (yv[idx] - yv[idx_]) / (xv[idx] - xv[idx_])
    return (cross - yv[idx]) / omega + xv[idx]

def analyzeMass(
        runDir: str,
        outDir: str,
        xMin: float | int = -1,
        xMax: float | int = -1,
        yMin: float | int = 0,
        yMax: float | int = 2,
        label: str = 'label'):

    if not os.path.exists(outDir): os.makedirs(outDir, exist_ok=True)

    fIn = ROOT.TFile(f'{runDir}/higgsCombinemass.MultiDimFit.mH125.root', 'READ')
    t = fIn.Get('limit')

    str_out = ''

    xv, yv = [], []
    for i in range(0, t.GetEntries()):

        t.GetEntry(i)

        if t.quantileExpected < -1.5: continue
        if t.deltaNLL > 1000: continue
        if t.deltaNLL > 20: continue
        xv.append(getattr(t, 'MH'))
        yv.append(t.deltaNLL*2.)

    xv, yv = zip(*sorted(zip(xv, yv)))
    g = ROOT.TGraph(len(xv), array.array('d', xv), array.array('d', yv))

    # bestfit = minimum
    mass = 1e9
    for i in range(g.GetN()):
        if g.GetY()[i] == 0.: mass = g.GetX()[i]

    # extract uncertainties at crossing = 1
    unc_m = findCrossing(xv, yv, True,  mass)
    unc_p = findCrossing(xv, yv, False, mass)
    unc = (abs(mass - unc_m) + abs(unc_p - mass)) / 2

    ########### PLOTTING ###########
    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : min(xv) if xMin < 0 else xMin,
        'xmax'              : max(xv) if xMax < 0 else xMax,
        'ymin'              : yMin,
        'ymax'              : yMax ,  # max(yv)

        'xtitle'            : 'm_{h} (GeV)',
        'ytitle'            : '-2#DeltaNLL',

        'topRight'          : topRight,
        'topLeft'           : '#bf{FCC-ee} #scale[0.7]{#it{Internal}}',
    }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()

    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    g.SetMarkerStyle(20)
    g.SetMarkerColor(ROOT.kRed)
    g.SetMarkerSize(1)
    g.SetLineColor(ROOT.kRed)
    g.SetLineWidth(2)
    g.Draw('SAME LP')

    line = ROOT.TLine(float(cfg['xmin']), 1, float(cfg['xmax']), 1)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.Draw('SAME')

    leg = ROOT.TLegend(.20, 0.825, 0.90, .9)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.035)
    leg.SetMargin(0.15)
    leg.SetBorderSize(1)
    leg.AddEntry(g, '%s, #delta(m_{h}) = %.2f MeV' % (label, unc*1000.), 'LP')
    leg.Draw()

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{outDir}/mass{suffix}.png')
    canvas.SaveAs(f'{outDir}/mass{suffix}.pdf')


    # Write values to text file
    str_out = f'{unc_m} {unc_p} {unc} {mass}\n'
    for i in range(0, len(xv)): str_out += f'{xv[i]} {yv[i]}\n'
    tFile = open(f'{outDir}/mass{suffix}.txt', 'w')
    tFile.write(str_out)
    tFile.close()
    tFile = open(f'{runDir}/mass{suffix}.txt', 'w')
    tFile.write(str_out)
    tFile.close()


def doFit_mass(
        runDir: str,
        mhMin: float | int = 124.99,
        mhMax: float | int = 125.01,
        npoints: int = 50,
        combineOptions: str = ''):

    cmd = f'combine -M MultiDimFit -t -1 --setParameterRanges MH={mhMin},{mhMax} --points={npoints} --algo=grid ws.root --expectSignal=1 -m 125' + \
        f' --redefineSignalPOIs MH --X-rtd TMCSO_AdaptivePseudoAsimov -v 10 --X-rtd ADDNLL_CBNLL=0 -n mass {combineOptions}'
    cmd_tot = f"singularity exec --bind /work:/work /work/submit/jaeyserm/software/docker/combine-standalone_v9.2.1.sif bash -c '{cmd}'"
    subprocess.call(cmd_tot, shell=True, cwd=runDir)


def doFitDiagnostics_mass(runDir, mhMin=124.99, mhMax=125.01, combineOptions = ''):

    # cmd = f'combine -M FitDiagnostics -t -1 --setParameterRanges MH={mhMin},{mhMax} ws.root --expectSignal=1 -v 10  -m 125' + \
    #     f' --redefineSignalPOIs MH --floatParameters MH --X-rtd TMCSO_AdaptivePseudoAsimov --X-rtd ADDNLL_CBNLL=0 -n mass {combineOptions}'
    cmd = f'combine -M MultiDimFit --algo singles -t -1 --setParameterRanges MH={mhMin},{mhMax} ws.root --expectSignal=1 -v 10  -m 125' + \
        f' --redefineSignalPOIs MH --floatParameters MH --X-rtd TMCSO_AdaptivePseudoAsimov --X-rtd ADDNLL_CBNLL=0 -n mass {combineOptions}'

    cmd_tot = f"singularity exec --bind /work:/work /work/submit/jaeyserm/software/docker/combine-standalone_v9.2.1.sif bash -c '{cmd}'"
    subprocess.call(cmd_tot, shell=True, cwd=runDir)

    # Get the uncertainty
    with ROOT.TFile(f'{runDir}/higgsCombinemass.MultiDimFit.mH125.root') as fIn:
        fIn.ls()
        tt = fIn.Get('limit')
        vals = []
        for i in range(tt.GetEntries()):
            tt.GetEntry(i)
            vals.append(float(tt.MH))

        vals = sorted(vals)
        lo, best, hi   = vals[0], vals[1], vals[2]

        err_down, err_up = best - lo, hi - best
        err_avg = (err_up + err_down) / 2
    return err_avg



def plotMultiple(
        tags: list[str],
        labels: list[str],
        fOut: str,
        xMin: float | int = -1,
        xMax: float | int = -1,
        yMin: float | int = 0,
        yMax: float | int = 2,
        legLabel: str = '',
        forceStat: list[bool] = [],
        legMargin: float | int = 0.15):

    best_mass, unc_mass, g_mass = [], [], []
    if len(forceStat) == 0:
        forceStat = [False]*len(tags)

    for i, tag in enumerate(tags):
        xv, yv = [], []
        fIn = open(f'{tag}/mass{suffix+"_stat" if forceStat[i] else suffix}.txt', 'r')
        for i,line in enumerate(fIn.readlines()):

            line = line.rstrip()
            if i == 0:
                best_mass.append(float(line.split(' ')[3]))
                unc_mass.append(float(line.split(' ')[2]))
            else:
                xv.append(float(line.split(' ')[0]))
                yv.append(float(line.split(' ')[1]))

        g = ROOT.TGraph(len(xv), array.array('d', xv), array.array('d', yv))
        g_mass.append(g)


    ########### PLOTTING ###########
    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,

        'xtitle'            : 'm_{h} (GeV)',
        'ytitle'            : '-2#DeltaNLL',

        'topRight'          : topRight,
        'topLeft'           : '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()

    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    n = len(g_mass) + (0 if legLabel=='' else 1)
    leg = ROOT.TLegend(.20, 0.9-n*0.05, 0.90, .9)
    leg.SetBorderSize(0)
    # leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(legMargin)
    leg.SetBorderSize(1)
    if legLabel != '':
        leg.SetHeader(legLabel)

    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+1]
    for i,g in enumerate(g_mass):
        g.SetMarkerStyle(20)
        g.SetMarkerColor(colors[i])
        g.SetMarkerSize(1)
        g.SetLineColor(colors[i])
        g.SetLineWidth(4)
        g.Draw('SAME L')
        leg.AddEntry(g, f'{labels[i]} #delta(m_{{H}}) = {unc_mass[i]*1000:.2f} MeV', 'L')

    leg.Draw()
    line = ROOT.TLine(float(cfg['xmin']), 1, float(cfg['xmax']), 1)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.Draw('SAME')

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()

    canvas.SaveAs(f'{fOut}{suffix}.png' % (fOut, suffix))


def plotMultiple_xsec(
        tags: list[str],
        labels: list[str],
        fOut: str,
        xMin: float | int = -1,
        xMax: float | int = -1,
        yMin: float | int = 0,
        yMax: float | int = 2):

    best_xsec, unc_xsec, g_xsec = [], [], []
    for tag in tags:
        xv, yv = [], []
        with open(f'{tag}/xsec.txt', 'r') as fIn:
            for i,line in enumerate(fIn.readlines()):
                line = line.rstrip()
                if i == 0:
                    best_xsec.append(float(line.split(' ')[3]))
                    unc_xsec.append(float(line.split(' ')[2]))
                else:
                    xv.append(float(line.split(' ')[0]))
                    yv.append(float(line.split(' ')[1]))

        g = ROOT.TGraph(len(xv), array.array('d', xv), array.array('d', yv))
        g_xsec.append(g)

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,

        'xtitle'            : '#sigma(ZH#rightarrowl^{#plus}l^{#minus})/#sigma_{ref}',
        'ytitle'            : '-2#DeltaNLL',

        'topRight'          : topRight,
        'topLeft'           : '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()

    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    totEntries = len(g_xsec)
    leg = ROOT.TLegend(.20, 0.9-totEntries*0.05, 0.90, .9)
    leg.SetBorderSize(0)
    # leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.15)
    leg.SetBorderSize(1)

    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+1]
    for i,g in enumerate(g_xsec):
        g.SetMarkerStyle(20)
        g.SetMarkerColor(colors[i])
        g.SetMarkerSize(1)
        g.SetLineColor(colors[i])
        g.SetLineWidth(4)
        g.Draw('SAME L')
        leg.AddEntry(g, f'{labels[i]} #delta(#sigma) = {unc_xsec[i]*100:.2f}', 'L')

    leg.Draw()
    line = ROOT.TLine(float(cfg['xmin']), 1, float(cfg['xmax']), 1)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.Draw('SAME')

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f'{fOut}.png' % (fOut))


def breakDown(outDir):
    def getUnc(tag, type_):

        with open(f'{outDir}/{type_}{tag}.txt', 'r') as fIn:
            for i,line in enumerate(fIn.readlines()):
                line = line.rstrip()
                if i == 0:
                    best = float(line.split(' ')[3])
                    unc = float(line.split(' ')[2])
                    break

            if type_ == 'mass': unc*= 1000.  # convert to MeV
            if type_ == 'xsec': unc*= 100.  # convert to %
        return best, unc

    ############# mass #############
    canvas = ROOT.TCanvas('c', 'c', 1000, 1000)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.25)
    canvas.SetRightMargin(0.05)
    canvas.SetFillStyle(4000)  # transparency?
    canvas.SetGrid(1, 0)
    canvas.SetTickx(1)


    xMin, xMax = -3, 3
    xTitle = '#sigma_{syst.}(m_{H}) (MeV)'

    _, unc_ref = getUnc('_stat', 'mass')
    params = ['_BES', '_SQRTS', '_LEPSCALE_MU', '_LEPSCALE_EL', '']
    labels = ['BES', '#sqrt{s} #pm 2 MeV', 'Muon scale (~10^{-5})', 'El. scale (~10^{-5})', 'Syst. combined']

    n_params = len(params)
    h_pulls = ROOT.TH2F('pulls', 'pulls', 6, xMin, xMax, n_params, 0, n_params)
    g_pulls = ROOT.TGraphAsymmErrors(n_params)

    i = n_params
    for p in range(n_params):
        i -= 1
        _, unc = getUnc(params[p], 'mass')
        # unc = math.sqrt(unc_ref**2 - unc**2)
        print(unc_ref, unc)
        unc = math.sqrt(unc**2 - unc_ref**2)
        g_pulls.SetPoint(i, 0, float(i) + 0.5)
        g_pulls.SetPointError(i, unc, unc, 0., 0.)
        h_pulls.GetYaxis().SetBinLabel(i + 1, f'#splitline{labels[p]}{{({unc:.2g} MeV)}}')


    h_pulls.GetXaxis().SetTitleSize(0.04)
    h_pulls.GetXaxis().SetLabelSize(0.03)
    h_pulls.GetXaxis().SetTitle(xTitle)
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
    latex.SetTextAlign(30)  # 0 special vertical aligment with subscripts
    latex.DrawLatex(0.95, 0.925, topRight)

    latex.SetTextAlign(13)
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.25, 0.96, '#bf{FCCee} #scale[0.7]{#it{Simulation}}')

    canvas.SaveAs(f'{outDir}/mass_breakdown_impacts.png' % (outDir))
    del canvas, g_pulls, h_pulls

    ###############################
    ########## lln plots ##########

    _, unc_ref = getUnc('_stat', 'mass')
    params = ['_stat', '_BES', '_LEPSCALE', '_SQRTS', '']
    labels = ['Stat. only', 'Beam energy spread', 'Lepton scale', 'Center-of-mass energy', 'Stat. + syst. combined']

    best_mass, unc_mass, g_mass = [], [], []

    tags = [f'{outDir}/mass{p}.txt' for p in params]
    for i, tag in enumerate(tags):
        xv, yv = [], []
        with open(tag, 'r') as fIn:
            for i,line in enumerate(fIn.readlines()):

                line = line.rstrip()
                if i == 0:
                    best_mass.append(float(line.split(' ')[3]))
                    unc_mass.append(float(line.split(' ')[2]))
                else:
                    xv.append(float(line.split(' ')[0]))
                    yv.append(float(line.split(' ')[1]))

        g = ROOT.TGraph(len(xv), array.array('d', xv), array.array('d', yv))
        g_mass.append(g)


    ########### PLOTTING ###########
    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 124.995,
        'xmax'              : 125.005,
        'ymin'              : 0,
        'ymax'              : 2,

        'xtitle'            : 'm_{h} (GeV)',
        'ytitle'            : '-2#DeltaNLL',

        'topRight'          : topRight,
        'topLeft'           : '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    dummy = plotter.dummy()

    dummy.GetXaxis().SetNdivisions(507)
    dummy.Draw('HIST')

    legLabel = ''
    legMargin = 0.1
    n = len(g_mass) + (0 if legLabel=='' else 1)
    leg = ROOT.TLegend(.20, 0.9 - n * 0.05, 0.90, .9)
    leg.SetBorderSize(0)
    # leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(legMargin)
    leg.SetBorderSize(1)
    if legLabel != '':
        leg.SetHeader(legLabel)


    line = ROOT.TLine(float(cfg['xmin']), 1, float(cfg['xmax']), 1)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(2)
    line.Draw('SAME')

    colors = [
        ROOT.kRed+1,     # stat only
        ROOT.kBlue-4,    # beam energy spread
        ROOT.kOrange-3,  # lepton scale
        ROOT.kGreen+3,   # center-of-mass energy
        ROOT.kBlack      # stat + syst combined
    ]

    widths = [4, 2, 2, 2, 4]
    styles = [1, 2, 2, 2, 1]
    for i,g in enumerate(g_mass):
        g.SetMarkerStyle(styles[i])
        g.SetMarkerColor(colors[i])
        g.SetMarkerSize(1)
        g.SetLineColor(colors[i])
        g.SetLineWidth(widths[i])
        g.Draw('SAME L')
        leg.AddEntry(g, f'{labels[i]} #delta(m_{{H}}) = {unc_mass[i]*1000:.2f} MeV', 'L')

    leg.Draw()

    plotter.aux()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()

    canvas.SaveAs(f'{outDir}/mass_breakdown.png' % (outDir))


def text2workspace(runDir):

    cmd = 'text2workspace.py datacard.txt -o ws.root  -v 10 --X-allow-no-background'
    cmd_tot = f"singularity exec --bind /work:/work /work/submit/jaeyserm/software/docker/combine-standalone_v9.2.1.sif bash -c '{cmd}'"
    subprocess.call(cmd_tot, shell=True, cwd=runDir)


def combineCards(runDir, input_=[]):

    if not os.path.exists(runDir): os.makedirs(runDir, exist_ok=True)

    input_ = [f'{os.getcwd()}/{i}' for i in input_]
    cards = ' '.join(input_)

    cmd = f'combineCards.py   --force-shape {cards} > datacard.txt'
    cmd_tot = f"singularity exec --bind /work:/work /work/submit/jaeyserm/software/docker/combine-standalone_v9.2.1.sif bash -c '{cmd}'"
    subprocess.call(cmd_tot, shell=True, cwd=runDir)
    text2workspace(runDir)


if __name__ == '__main__':
    mode = args.mode  # Detector mode
    ecm = str(args.ecm)
    tag = args.tag
    lumiScale = args.lumi
    lumiStr = str(int(args.lumi)) if args.lumi.is_integer() else str(args.lumi)
    lumiLabel = lumiStr.replace('.', 'p')

    topRight = f'#sqrt{{s}} = {ecm} GeV, {lumiStr} ab^{{#minus1}}'
    topLeft = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'

    combineDir = f'output/h_mass/combine/{tag}/{mode}/lumi{lumiLabel}/'  # baseline combine dir
    outDir = f'/work/submit/jaeyserm/public_html/fccee/h_mass/{tag}/combine/{mode}/lumi{lumiLabel}/'



    def mHrange(mh_err):
        if 1.5 * mh_err > 0.05:  # bound to 50 MeV
            return 124.95, 125.05
        return 125 - 1.5 * mh_err, 125 + 1.5 * mh_err

    combineOptions, suffix = '', ''
    freezeParameters, setParameters = [], []

    doSyst = not args.statOnly
    if not doSyst:
        suffix = '_stat'
        # freezeParameters.extend(['BES', 'ISR', 'SQRTS', 'LEPSCALE_MU', 'LEPSCALE_EL'])
        freezeParameters.extend(['BES_ecm240', 'SQRTS_ecm240', 'LEPSCALE_MU_ecm240', 'LEPSCALE_EL_ecm240'])
        freezeParameters.extend(['bkg_norm_mumu_ecm240', 'bkg_norm_ee_ecm240', 'bkg_norm_mumu_ecm365', 'bkg_norm_ee_ecm365'])
        freezeParameters.extend(['BES_ecm365', 'SQRTS_ecm365', 'LEPSCALE_MU_ecm365', 'LEPSCALE_EL_ecm365'])




    freezeBkg = False
    if freezeBkg:
        suffix = f'_freezeBkg{suffix}'
        freezeParameters.extend(['bkg_norm_mumu_ecm240', 'bkg_norm_ee_ecm240', 'bkg_norm_mumu_ecm365', 'bkg_norm_ee_ecm365'])


    noBkg = False
    if noBkg:
        suffix = f'_noBkg{suffix}' %suffix
        freezeParameters.extend(['bkg_norm_mumu_ecm240', 'bkg_norm_ee_ecm240', 'bkg_norm_mumu_ecm365', 'bkg_norm_ee_ecm365', 'r'])  # r to be frozen to avoid issues with no-bkg fit
        setParameters.extend(['bkg_norm_mumu_ecm240=0', 'bkg_norm_ee_ecm240=0', 'bkg_norm_mumu_ecm365=0', 'bkg_norm_ee_ecm365=0'])


    # Systematic variations, unfreeze them
    systs = ['BES_ecm240', 'SQRTS_ecm240', 'LEPSCALE_MU_ecm240', 'LEPSCALE_EL_ecm240', 'BES_ecm365', 'SQRTS_ecm365', 'LEPSCALE_MU_ecm365', 'LEPSCALE_EL_ecm365']

    # suffix+='_BES'
    # systs.remove('BES_ecm240')
    # systs.remove('BES_ecm365')

    # suffix+='_SQRTS'
    # systs.remove('SQRTS_ecm240')
    # systs.remove('SQRTS_ecm365')

    # suffix+='_LEPSCALE_MU'
    # systs.remove('LEPSCALE_MU_ecm240')
    # systs.remove('LEPSCALE_MU_ecm365')

    # suffix+='_LEPSCALE_EL'
    # systs.remove('LEPSCALE_EL_ecm240')
    # systs.remove('LEPSCALE_EL_ecm365')

    suffix += '_LEPSCALE'
    systs.remove('LEPSCALE_EL_ecm240')
    systs.remove('LEPSCALE_EL_ecm365')
    systs.remove('LEPSCALE_MU_ecm240')
    systs.remove('LEPSCALE_MU_ecm365')

    freezeParameters.extend(systs)


    doBreakDown = True
    if doBreakDown:
        breakDown(f'{outDir}/mumu_ee_combined_categorized_ecm240')
        quit()
        breakDown('mumu_ee_combined_inclusive')
        breakDown('mumu_cat0')
        breakDown('mumu_cat1')
        breakDown('mumu_cat2')
        breakDown('mumu_cat3')
        breakDown('mumu_combined')
        breakDown('ee_cat0')
        breakDown('ee_cat1')
        breakDown('ee_cat2')
        breakDown('ee_cat3')
        breakDown('ee_combined')
        quit()

    doSummary = False
    if doSummary:
        outDir__ = f'/work/submit/jaeyserm/public_html/fccee/higgs_mass_xsec/{args.tag}/combine/'
        outDir   = f'/work/submit/jaeyserm/public_html/fccee/higgs_mass_xsec/{args.tag}/combine/summaryPlots/'
        plotMultiple([f'{outDir__}/IDEA/lumi10p8/mumu_combined_ecm240/',
                      f'{outDir__}/IDEA_MC/lumi10p8/mumu_combined_ecm240/',
                      f'{outDir__}/IDEA_3T/lumi10p8/mumu_combined_ecm240/',
                      f'{outDir__}/CLD/lumi10p8/mumu_combined_ecm240/'],
                     ['IDEA', 'IDEA perfect resolution', 'IDEA 3T', 'IDEA CLD silicon tracker'],
                     f'{outDir}/IDEA_IDEAL_2T_3T_CLD_mumu',
                     124.99, 125.01, legLabel='Muon final state Z(#mu^{#plus}#mu^{#minus})H (stat. + syst.)')
        plotMultiple([f'{outDir__}/IDEA/lumi10p8/mumu_combined_ecm240/',
                      f'{outDir__}/IDEA_MC/lumi10p8/mumu_combined_ecm240/',
                      f'{outDir__}/IDEA_3T/lumi10p8/mumu_combined_ecm240/',
                      f'{outDir__}/CLD/lumi10p8/mumu_combined_ecm240/'],
                     ['IDEA', 'IDEA perfect resolution', 'IDEA 3T', 'IDEA CLD silicon tracker'],
                     f'{outDir}/IDEA_IDEAL_2T_3T_CLD_mumu_stat',
                     124.99, 125.01,
                     legLabel='Muon final state Z(#mu^{#plus}#mu^{#minus})H (stat. only)',
                     forceStat=[True, True, True, True])
        plotMultiple([f'{outDir__}/IDEA/lumi10p8/mumu_ee_combined_categorized_ecm240/',
                      f'{outDir__}/IDEA/lumi10p8/mumu_ee_combined_categorized_ecm240/'],
                     ['Statistical', 'Statistical+systematic'],
                     f'{outDir}/IDEA_stat_syst',
                     124.995, 125.005,
                     legLabel='Combined muon and electron final states',
                     forceStat=[True, False])
        quit()


    ##################################
    if len(freezeParameters) > 0:
        combineOptions += ' --freezeParameters ' + ','.join(freezeParameters)
    if len(setParameters) > 0:
        combineOptions += ' --setParameters ' + ','.join(setParameters)


    ############### MUON
    if True:
        for cat, tag_base in [('ee', 'e^{+}e^{-}, '), ('mumu', '#mu^{+}#mu^{-}, ')]:
            for tag, pos in [('cat0', 'inclusive'),
                             ('cat1', 'central-central'),
                             ('cat2', 'central-forward'),
                             ('cat3', 'forward-forward'),
                             ('combined', 'combined')]:
                tag_cat, label = f'{cat}_{tag}_ecm{ecm}', tag_base + pos

        if tag=='combined':
            combineCards(f'{combineDir}/combined',
                         [f'{combineDir}/{cat}_cat1_ecm{ecm}/datacard.txt',
                          f'{combineDir}/{cat}_cat2_ecm{ecm}/datacard.txt',
                          f'{combineDir}/{cat}_cat3_ecm{ecm}/datacard.txt'])

        mh_err = doFitDiagnostics_mass(f'{combineDir}/{tag_cat}', 124.95, 125.05, combineOptions)
        mhMin, mhMax = mHrange(mh_err)
        doFit_mass(f'{combineDir}/{tag_cat}', mhMin, mhMax, 50, combineOptions)
        analyzeMass(f'{combineDir}/{tag_cat}', f'{outDir}/{tag_cat}/', mhMin, mhMax, label=label)

    ############### MUON+ELECTRON
    if True:

        # check if lumi and ecm are defined
        tag, label = f'mumu_ee_combined_inclusive_ecm{ecm}', '#mu^{#plus}#mu^{#minus}+e^{#plus}e^{#minus}, inclusive'
        combineCards(f'{combineDir}/{tag}',
                     [f'{combineDir}/mumu_cat0_ecm{ecm}/datacard.txt', f'{combineDir}/ee_cat0_ecm{ecm}/datacard.txt'])

        mh_err = doFitDiagnostics_mass(f'{combineDir}/{tag}', 124.95, 125.05, combineOptions)
        mhMin, mhMax = mHrange(mh_err)
        doFit_mass(f'{combineDir}/{tag}', mhMin, mhMax, 50, combineOptions)
        analyzeMass(f'{combineDir}/{tag}', f'{outDir}/{tag}/', mhMin, mhMax, label=label)

        tag, label = f'mumu_ee_combined_categorized_ecm{ecm}', '#mu^{#plus}#mu^{#minus}+e^{#plus}e^{#minus}, categorized'
        combineCards(f'{combineDir}/{tag}',
                     [f'{combineDir}/mumu_combined_ecm{ecm}/datacard.txt',
                      f'/{combineDir}/ee_combined_ecm{ecm}/datacard.txt'])
        mh_err = doFitDiagnostics_mass(f'{combineDir}/{tag}', 124.95, 125.05, combineOptions)
        mhMin, mhMax = mHrange(mh_err)
        doFit_mass(f'{combineDir}/{tag}', mhMin, mhMax, 50, combineOptions)
        analyzeMass(f'{combineDir}/{tag}', f'{outDir}/{tag}/', mhMin, mhMax, label=label)

        plotMultiple([f'{outDir}/mumu_combined_ecm{ecm}/',
                      f'{outDir}/ee_combined_ecm{ecm}/',
                      f'{outDir}/mumu_ee_combined_categorized_ecm{ecm}/'],
                     ['#mu^{#plus}#mu^{#minus}', 'e^{#plus}e^{#minus}', '#mu^{#plus}#mu^{#minus} + e^{#plus}e^{#minus}'],
                     f'{outDir}/mumu_ee_combined_categorized_ecm{ecm}', 124.99, 125.01)


    # 240+365
    if args.combination:

        lumi_240, lumi_365 = '10p8', '3p12'
        fit_tag = 'mumu_ee_combined_categorized'
        label = '#mu^{#plus}#mu^{#minus}+e^{#plus}e^{#minus}, categorized'

        tag = f'combined_ecm_{lumi_240}_{lumi_365}'
        topRight = f'#sqrt{{s}} = 240/365 GeV, {lumi_240.replace('p','.')}/{lumi_365.replace('p','.')} ab^{{#minus1}}'

        base_dir = f'combine/higgs_mass_{args.tag}/{mode}/'
        combineDir = f'{base_dir}/{tag}/{fit_tag}/'
        combineDir_240 = f'{base_dir}/lumi{lumi_240}/{fit_tag}_ecm240'
        combineDir_365 = f'{base_dir}/lumi{lumi_365}/{fit_tag}_ecm365'

        outDir_    = f'/work/submit/jaeyserm/public_html/fccee/higgs_mass_xsec/{args.tag}/combine/{mode}/{tag}/{fit_tag}/'
        outDir_240 = f'/work/submit/jaeyserm/public_html/fccee/higgs_mass_xsec/{args.tag}/combine/{mode}/lumi{lumi_240}/{fit_tag}_ecm240/'
        outDir_365 = f'/work/submit/jaeyserm/public_html/fccee/higgs_mass_xsec/{args.tag}/combine/{mode}/lumi{lumi_240}/{fit_tag}_ecm365/'

        doBreakDown = True
        if doBreakDown:
            breakDown(outDir_)
            quit()

        combineCards(combineDir, [combineDir_240+'/datacard.txt', combineDir_365+'/datacard.txt'])
        mh_err = doFitDiagnostics_mass(combineDir, mhMin=124.95, mhMax=125.05, combineOptions=combineOptions)
        mhMin, mhMax = mHrange(mh_err)
        doFit_mass(combineDir, mhMin=mhMin, mhMax=mhMax, npoints=50, combineOptions=combineOptions)
        analyzeMass(combineDir, outDir_, label=label, xMin=mhMin, xMax=mhMax)
        plotMultiple([outDir_240, outDir_365, outDir_], ['#sqrt{s} = 240 GeV', '#sqrt{s} = 365 GeV', 'Combination'], outDir_, 124.98, 125.02)
