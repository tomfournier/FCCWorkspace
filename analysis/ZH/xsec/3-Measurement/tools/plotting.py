import os, math, json
import matplotlib.pyplot as plt

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from . import plotter

h_decays_labels = {
    "bb": "H#rightarrowb#bar{b}", "cc": "H#rightarrowc#bar{c}", "ss": "H#rightarrows#bar{s}", "gg": "H#rightarrowgg", 
    "mumu": "H#rightarrow#mu^{#plus}#mu^{#minus}", "tautau": "H#rightarrow#tau^{#plus}#tau^{#minus}", 
    "ZZ": "H#rightarrowZZ*", "WW": "H#rightarrowWW*", "Za": "H#rightarrowZ#gamma", 
    "aa": "H#rightarrow#gamma#gamma", "inv": "H#rightarrowInv"
}

h_decays_colors = {
    "bb": ROOT.kBlack, "cc": ROOT.kBlue , "ss": ROOT.kRed, "gg": ROOT.kGreen+1, "mumu": ROOT.kOrange, 
    "tautau": ROOT.kCyan, "ZZ": ROOT.kGray, "WW": ROOT.kGray+2, "Za": ROOT.kGreen+2, "aa": ROOT.kRed+2, 
    "inv": ROOT.kBlue+2
}

#__________________________________________________________
def getMetaInfo(proc, info='crossSection', remove=False,
                fcc='cvmfs/fcc.cern.ch/FCCDicts',
                procFile="FCCee_procDict_winter2023_IDEA.json"):
    if not ('eos' in procFile):
        procFile = os.path.join(os.getenv('FCCDICTSDIR').split(':')[0], '') + procFile 
    with open(procFile, 'r') as f:
        procDict=json.load(f)
    xsec = procDict[proc][info]
    if remove:
        if 'HZZ' in proc:
            xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'))
            xsec = getMetaInfo(proc) - xsec_inv
        if 'p8_ee_WW_ecm' in proc:
            xsec_ee   = getMetaInfo(proc.replace('WW', 'WW_ee'))
            xsec_mumu = getMetaInfo(proc.replace('WW', 'WW_mumu'))
            xsec      = getMetaInfo(proc) - xsec_ee - xsec_mumu
    return xsec

#__________________________________________________________
def getHist(hName, procs, inputDir, suffix='', rebin=1, lazy=True):
    hist = None
    for proc in procs:
        fInName = f"{inputDir}/{proc}{suffix}.root"
        if os.path.exists(fInName):
            fIn = ROOT.TFile(fInName)
        else:
            if lazy: continue
            else:
                print(f"ERROR: input file {fInName} not found")
                quit()
        h = fIn.Get(hName)
        h.SetDirectory(0)
        if 'HZZ' in proc:
            xsec_old = getMetaInfo(proc)
            xsec_new = getMetaInfo(proc, remove=True)
            h.Scale( xsec_new/xsec_old )
        if 'p8_ee_WW_ecm' in proc:
            xsec_old = getMetaInfo(proc)
            xsec_new = getMetaInfo(proc, remove=True)
            h.Scale( xsec_new/xsec_old )
        if hist == None: hist = h
        else: hist.Add(h)
        fIn.Close()
    hist.Rebin(rebin)
    return hist

#__________________________________________________________
def significance(variable, variables, inputDir, outDir, procs_cfg, procs, xMin=None, xMax=None, sel='', 
                 outName="", reverse=False, plot_file=['png'], rebin=1, lazy=True):

    if outName == "": outName = variable
    suffix = f'_{sel}_histo'

    h_sig = getHist(variable, procs_cfg[procs[0]], inputDir, suffix=suffix, rebin=rebin)
    sig_tot = h_sig.Integral()

    bkgs_procs = []
    for i,bkg in enumerate(procs[1:]):
        bkgs_procs.extend(procs_cfg[bkg])

    h_bkg = getHist(variable, bkgs_procs, inputDir, 
                    suffix=suffix, rebin=rebin, lazy=lazy)
    x, y, l = [], [], []

    for i in range(1, h_sig.GetNbinsX()+1):
        if reverse: iStart, iEnd = 1, i
        else:       iStart, iEnd = i, h_sig.GetNbinsX()+1
        center = h_sig.GetBinCenter(i)

        if xMin is not None and xMax is not None:
            if center > xMax or center < xMin:
                continue
        elif xMin is not None: 
            if center < xMin: continue
        elif xMax is not None:
            if center > xMax: continue

        sig = h_sig.Integral(iStart, iEnd)
        bkg = h_bkg.Integral(iStart, iEnd)
        if (sig+bkg) <= 0 or sig_tot <= 0:
            significance = 0
            sig_loss = 0
        else:
            significance = sig / (sig + bkg)**0.5
            sig_loss = sig / sig_tot
        x.append(center), y.append(significance), l.append(sig_loss)

    max_y = max(y)
    max_index = y.index(max_y)
    max_x, max_l = x[max_index], l[max_index]

    fig, ax1 = plt.subplots(figsize=[12,10])
    ax1.scatter(x, y, marker='o', color='blue', label='Significance')
    ax1.scatter(max_x, max_y, marker='*', color='red', s=150)
    ax1.axvline(max_x, color='black', alpha=0.8, linewidth=1)
    ax1.axhline(max_y, color='blue', alpha=0.8, linewidth=1)

    gev = ' [GeV]' if '[GeV]' in variables[variable]['xlabel'] else ''
    GeV = ' GeV'   if '[GeV]' in variables[variable]['xlabel'] else ''
    cut = variables[variable]['xlabel'].replace(' [GeV]', '').replace('#', "\\")

    ax1.set_xlabel(f'${cut}${gev}', fontsize=22, loc='right')
    ax1.set_ylabel('Significance', fontsize=22, loc='top')
    ax1.set_xlim(min(x), max(x))
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.yaxis.label.set_color('blue')
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(x, l, color='red', linewidth=3, label='Signal efficiency')
    ax2.axhline(max_l, color='red', alpha=0.8, linewidth=1)
    ax2.set_ylabel('Signal efficiency', fontsize=22, loc='top')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=20)
    ax2.yaxis.label.set_color('red')

    if reverse:
        ax1.set_title(f"Max: ${cut}$ < {max_x:.2f}{GeV}, Significance = {max_y:.2f}, Signal eff = {max_l*100:.1f} %", fontsize=23)
    else:
        ax1.set_title(f"Max: ${cut}$ > {max_x:.2f}{GeV}, Significance = {max_y:.2f}, Signal eff = {max_l*100:.1f} %", fontsize=23)
    fig.tight_layout()

    out = f'{outDir}/significance/{sel}'
    if not os.path.isdir(out):
            os.system(f'mkdir -p {out}')

    suffix = "_reverse" if reverse else ""
    for pl in plot_file:
        plt.savefig(f"{out}/{outName}{suffix}.{pl}", bbox_inches='tight', dpi=fig.dpi)
        print(f'----->[Info] Plotted signficance plot in {pl} format at {out}/{outName}{suffix}.{pl}')
    plt.close()

#__________________________________________________________
def makePlot(variable, variables, inputDir, outDir, procs, procs_cfg, 
             colors, legend, sel='', ecm=240, lumi=10.8, outName="", 
             logX=False, rebin=1, sig_scale=1, 
             plot_file=['png'], stack=False, lazy=True):

    if outName == "": outName = variable
    suffix = f'_{sel}_histo'

    leg = ROOT.TLegend(.55, 0.99-(len(procs))*0.06, .99, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    h_sig = getHist(variable, procs_cfg[procs[0]], inputDir, 
                    suffix=suffix, rebin=rebin, lazy=lazy)
    h_sig.SetLineColor(colors[procs[0]])
    h_sig.SetLineWidth(3)
    h_sig.SetLineStyle(1)
    h_sig.Scale(sig_scale)
    if sig_scale != 1:
        leg.AddEntry(h_sig, f"{legend[procs[0]]} (#times {int(sig_scale)})", "L")
    else:
        leg.AddEntry(h_sig, legend[procs[0]], "L")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    for bkg in procs[1:]:
        h_bkg = getHist(variable, procs_cfg[bkg], inputDir, 
                        suffix=suffix, rebin=rebin, lazy=lazy)

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(colors[bkg])
        h_bkg.SetLineColor(ROOT.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)

        leg.AddEntry(h_bkg, legend[bkg], "F")
        st.Add(h_bkg)

    
    xMin, xMax, yMin, yMax = variables[variable]["lim"]

    if yMax < 0:
        if variables[variable]['logY']:
            yMax = math.ceil(max([h_bkg_tot.GetMaximum(), h_sig.GetMaximum()])*10000)/10.
        else:
            yMax = 1.4*max([h_bkg_tot.GetMaximum(), h_sig.GetMaximum()])

    cfg = {

        'logx' : logX, 'logy' : variables[variable]['logY'],
        'xmin' : xMin, 'xmax' : xMax, 'ymin' : yMin, 'ymax' : yMax,
        'xtitle' : variables[variable]['xlabel'], 'ytitle' : 'Events',
            
        'topRight' : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'  : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    dummy.Draw("HIST")
    st.Draw("HIST SAME")

    if stack:
        h_bkg_tot.Add(h_sig)
        h_bkg_tot.SetLineColor(colors[variable])
    
    else:
        h_bkg_tot.SetLineColor(ROOT.kBlack)
        h_sig.Draw('HIST SAME')
    
    h_bkg_tot.SetFillColor(0)
    h_bkg_tot.SetLineWidth(2)
    h_bkg_tot.Draw("HIST SAME")
    leg.Draw("SAME")
    
    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

    out = f'{outDir}/makePlot/{sel}'
    if not os.path.isdir(out):
        os.system(f'mkdir -p {out}')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/{outName}.{pl}")
    canvas.Close()


#__________________________________________________________
def PlotDecays(variable, variables, inputDir, outDir, z_decays, h_decays, sel='',
               ecm=240, lumi=10.8, outName="", plot_file=['png'], logX=False, rebin=1, lazy=True):

    if outName == "": outName = variable
    sigs, suffix = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays], f'_{sel}_histo'

    leg = ROOT.TLegend(.2, .925-(len(sigs)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists = []
    for i, sig in enumerate(sigs):
        h_decay = h_decays[i]
        h_sig = getHist(variable, sig, inputDir, 
                        suffix=suffix, rebin=rebin, lazy=lazy)
        h_sig.Rebin(rebin)
        if h_sig.Integral() > 0:
            h_sig.Scale(1./h_sig.Integral())

        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)

        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)

    xMin, xMax, yMin, yMax = variables[variable]["limH"]

    cfg = {

        'logx' : logX, 'logy' : variables[variable]['logY'],
        'xmin' : xMin, 'xmax' : xMax, 'ymin' : yMin, 'ymax' : yMax,
        'xtitle' : variables[variable]['xlabel'], 'ytitle' : 'Normalized to Unity',
            
        'topRight' : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft' : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy(1)
    dummy.Draw("HIST") 

    for i,hist in enumerate(hists):
        hist.Draw("SAME HIST")
    leg.Draw("SAME")
    
    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()
    
    out = f'{outDir}/higgsDecays/{sel}'
    if not os.path.isdir(out):
        os.system(f'mkdir -p {out}')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/{outName}.{pl}")
    canvas.Close()


#__________________________________________________________
def AAAyields(hName, inputdir, outputdir, plots, legend, colors, cat, sel,
              ecm=240, lumi=10.8, scale_sig=1, scale_bkg=1,
              plot_file=['png'], outName='', lazy=True):
    
    if outName=='': outName = 'AAAyields'
    if   cat == 'mumu': ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
    elif cat == 'ee':   ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
    suffix = f'_{sel}_histo'

    signal      = plots['signal']
    backgrounds = plots['backgrounds']

    legsize = 0.04 * (len(signal)+ len(backgrounds))
    leg = ROOT.TLegend(0.6, 0.86 - legsize, 0.9, 0.88)
    leg.SetLineColor(0)
    leg.SetTextFont(42)

    hbackgrounds = {}
    for b in backgrounds:
        hist = getHist(hName, backgrounds[b], inputdir,
                        suffix=suffix, lazy=lazy)
        hist.Scale(scale_bkg)
        hist.SetLineWidth(1)
        hist.SetLineColor(ROOT.kBlack)
        hist.SetFillColor(colors[b])
        leg.AddEntry(hist, legend[b], "F")

        hbackgrounds[b] = [hist]

    hsignal = {}
    for s in signal:
        hist = getHist(hName, signal[s], inputdir, 
                        suffix=suffix, lazy=lazy)
        hist.Scale(scale_sig)
        hist.SetLineWidth(4)
        hist.SetLineStyle(1)
        hist.SetLineColor(colors[s])
        leg.AddEntry(hist, legend[s], "L")
        hsignal[s] = [hist]

    yields = {}
    for s in hsignal:
        yields[s] = [legend[s], 
                     hsignal[s][0].Integral(0, -1),
                     hsignal[s][0].GetEntries()]
    for b in backgrounds:
        yields[b] = [legend[b],
                     hbackgrounds[b][0].Integral(0, -1),
                     hbackgrounds[b][0].GetEntries()]

    canvas = ROOT.TCanvas('yield', '', 800, 800)
    canvas.SetTicks(1, 1)
    canvas.SetLeftMargin(0.14)
    canvas.SetRightMargin(0.08)

    dummyh = ROOT.TH1F("", "", 1, 0, 1)
    dummyh.SetStats(0)
    dummyh.GetXaxis().SetLabelOffset(999)
    dummyh.GetXaxis().SetLabelSize(0)
    dummyh.GetYaxis().SetLabelOffset(999)
    dummyh.GetYaxis().SetLabelSize(0)
    dummyh.Draw("AH")
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(31)
    latex.SetTextSize(0.04)

    text = "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}"
    latex.DrawLatex(0.90, 0.92, text)

    text = '#bf{#it{#sqrt{s} = '+f'{ecm}'+' GeV}}'
    latex.SetTextAlign(12)
    latex.SetNDC(ROOT.kTRUE)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.18, 0.83, text)

    text = '#bf{#it{L = '+f'{lumi}'+' ab^{#minus1}}}'
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.18, 0.78, text)

    text = '#bf{#it{' + ana_tex + '}}'
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.18, 0.73, text)

    text = '#bf{#it{' + sel + '}}'
    latex.SetTextSize(0.025)
    latex.DrawLatex(0.18, 0.68, text)

    text = '#bf{#it{Signal Scaling = ' + f'{scale_sig:.3g}' + \
            '}}'
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.18, 0.57, text)

    text = '#bf{#it{Background Scaling = ' + \
        f'{scale_bkg:.3g}' + '}}'
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.18, 0.52, text)

    text = '#bf{#it{' + 'Process' + '}}'
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.18, 0.45, text)

    text = '#bf{#it{' + 'Yields' + '}}'
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.5, 0.45, text)

    text = '#bf{#it{' + 'Raw MC' + '}}'
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.75, 0.45, text)

    for dy, y in enumerate(yields):
        text = '#bf{#it{' + yields[y][0] + '}}'
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.18, 0.4-dy*0.05, text)

        stry = str(yields[y][1])
        stry = stry.split('.', maxsplit=1)[0]
        text = '#bf{#it{' + stry + '}}'
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.5, 0.4-dy*0.05, text)

        stry = str(yields[y][2])
        stry = stry.split('.', maxsplit=1)[0]
        text = '#bf{#it{' + stry + '}}'
        latex.SetTextSize(0.035)
        latex.DrawLatex(0.75, 0.4-dy*0.05, text)

    out = f'{outputdir}/yield/{sel}'
    if not os.path.exists(out):
        os.system(f'mkdir -p {out}')
    
    for pl in plot_file:
        canvas.SaveAs(f'{out}/{outName}.{pl}')
    canvas.Close()

