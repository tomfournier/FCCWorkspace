import sys, os, math, copy, json
import numpy as np
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
def getMetaInfo(proc, info='crossSection',
                procFile="FCCee_procDict_winter2023_IDEA.json"):
    if not ('eos' in procFile):
        procFile = os.path.join(os.getenv('FCCDICTSDIR').split(':')[0], '') + procFile 
    with open(procFile, 'r') as f:
        procDict=json.load(f)
    xsec = procDict[proc][info]
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
            xsec     = getMetaInfo(proc)
            xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'))
            h.Scale( (xsec-xsec_inv)/xsec )
        if 'p8_ee_WW_ecm' in proc:
            xsec      = getMetaInfo(proc)
            xsec_ee   = getMetaInfo(proc.replace('WW', 'WW_ee'))
            xsec_mumu = getMetaInfo(proc.replace('WW', 'WW_mumu'))
            h.Scale( (xsec-xsec_ee-xsec_mumu)/xsec )
        if hist == None: hist = h
        else: hist.Add(h)
        fIn.Close()
    hist.Rebin(rebin)
    return hist

#__________________________________________________________
def CutFlow(inputDir, outDir, procs, procs_cfg, procs_labels, procs_colors, plot_file=['png'], ecm=240, lumi=10.8,
            outName="cutFlow", hName="cutFlow", cuts=[], labels=[], sig_scale=1.0, yMin=1e4, yMax=1e10):
    
    if outName=="": outName = hName

    leg = ROOT.TLegend(.55, 0.99-(len(procs))*0.06, .99, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    hists_yields = []
    significances = []
    h_sig = getHist(hName, procs_cfg[procs[0]], inputDir)

    hists_yields.append(copy.deepcopy(h_sig))
    h_sig.Scale(sig_scale)
    h_sig.SetLineColor(procs_colors[procs[0]])
    h_sig.SetLineWidth(4)
    h_sig.SetLineStyle(1)
    if sig_scale != 1:
        leg.AddEntry(h_sig, f"{procs_labels[procs[0]]} (#times {int(sig_scale)})", "L")
    else:
        leg.AddEntry(h_sig, procs_labels[procs[0]], "L")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    for i,bkg in enumerate(procs[1:]):
        h_bkg = getHist(hName, procs_cfg[bkg], inputDir)

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(procs_colors[bkg])
        h_bkg.SetLineColor(ROOT.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)

        leg.AddEntry(h_bkg, procs_labels[bkg], "F")
        st.Add(h_bkg)
        hists_yields.append(h_bkg)

    h_bkg_tot.SetLineColor(ROOT.kBlack)
    h_bkg_tot.SetLineWidth(2)

    for i,cut in enumerate(cuts):
        nsig = h_sig.GetBinContent(i+1) / sig_scale ## undo scaling
        nbkg = 0
        for histProc in hists_yields:
            nbkg = nbkg + histProc.GetBinContent(i+1)
        if (nsig+nbkg) == 0:
            print(f"Cut {cut} zero yield sig+bkg")
            s = -1
        else:
            s = nsig / (nsig + nbkg)**0.5
        # print(f'Cut {i}:\tSignificance = {s:.2f}')
        significances.append(s)

    ########### PLOTTING ###########
    cfg = {
        'logy'              : True,
        'logx'              : False,

        'xmin'              : 0,
        'xmax'              : len(cuts),
        'ymin'              : yMin,
        'ymax'              : yMax ,

        'xtitle'            : "",
        'ytitle'            : "Events",

        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",
        }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    canvas.SetTicks()
    dummy = plotter.dummy(len(cuts))
    dummy.GetXaxis().SetLabelSize(0.75*dummy.GetXaxis().GetLabelSize())
    dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
    for i,label in enumerate(labels): dummy.GetXaxis().SetBinLabel(i+1, label)
    dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST")

    st.Draw("SAME HIST")
    h_bkg_tot.Draw("SAME HIST")
    h_sig.Draw("SAME HIST")
    leg.Draw("SAME")

    if not os.path.isdir(f'{outDir}/cutflow'):
            os.system(f'mkdir -p {outDir}/cutflow')

    plotter.aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/{outName}.{pl}")

    out_orig = sys.stdout
    with open(f"{outDir}/cutflow/{outName}.txt", 'w') as f:
        sys.stdout = f

        formatted_row = '{:<10} {:<25} ' + ' '.join(['{:<25}']*len(procs))
        print(formatted_row.format(*(["Cut", "Significance"]+procs)))
        print(formatted_row.format(*(["----------"]+["-----------------------"]*(len(procs)+1))))
        for i,cut in enumerate(cuts):
            row = ["Cut %d"%i, "%.3f"%significances[i]]
            for histProc in hists_yields:
                yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
                row.append("%.4e +/- %.2e" % (yield_, err))

            print(formatted_row.format(*row))
    sys.stdout = out_orig

#__________________________________________________________
def CutFlowDecays(inputDir, outDir, z_decays, h_decays, plot_file=['png'], suffix='', ecm=240, lumi=10.8, 
                  hName="cutFlow", outName="cutFlow", cuts=[], cut_labels=[], yMin=0, yMax=150):

    if outName == "":
        outName = hName

    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays]
    leg = ROOT.TLegend(.2, .925-(len(sigs)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists = []
    hist_tot = None
    eff_final, eff_final_err = [], []
    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        h_sig = getHist(hName, sig, inputDir)
        h_sig.Scale(100./h_sig.GetBinContent(1))

        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)

        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)

        eff_final.append(h_sig.GetBinContent(len(cuts)))
        eff_final_err.append(h_sig.GetBinError(len(cuts)))
        
        if hist_tot == None:
            hist_tot = h_sig.Clone("h_tot")
        else:
            hist_tot.Add(h_sig)

    hist_tot.Scale(1./len(sigs))
    eff_avg = sum(eff_final) / float(len(eff_final))
    eff_avg, eff_avg_err = hist_tot.GetBinContent(len(cuts)), hist_tot.GetBinError(len(cuts))
    eff_min, eff_max = eff_avg-min(eff_final), max(eff_final)-eff_avg

    ########### PLOTTING ###########
    cfg = {
        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : len(cuts),
        'ymin'              : yMin,
        'ymax'              : yMax ,

        'xtitle'            : "",
        'ytitle'            : "Selection efficiency [%]",

        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",
        }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    canvas.SetTicks()
    dummy = plotter.dummy(len(cuts))
    dummy.GetXaxis().SetLabelSize(0.8*dummy.GetXaxis().GetLabelSize())
    dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
    for i,label in enumerate(cut_labels): dummy.GetXaxis().SetBinLabel(i+1, label)
    dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST")

    txt = ROOT.TLatex()
    txt.SetTextSize(0.04)
    txt.SetTextColor(1)
    txt.SetTextFont(42)
    txt.SetNDC()
    txt.DrawLatex(0.2, 0.2, f"Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %")
    txt.DrawLatex(0.2, 0.15, f"Min/max: {eff_min:.2f}/{eff_max:.2f}")
    txt.Draw("SAME")

    for hist in hists:
        hist.Draw("SAME HIST")
    leg.Draw("SAME")

    plotter.aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()

    if not os.path.isdir(f'{outDir}/cutflow'):
            os.system(f'mkdir -p {outDir}/cutflow')

    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/{outName}_decays.{pl}")

    out_orig = sys.stdout
    with open(f"{outDir}/cutflow/{outName}_decays.txt", 'w') as f:
        sys.stdout = f

        formatted_row = '{:<10} ' + ' '.join(['{:<18}']*len(hists))
        print(formatted_row.format(*(["Cut"]+h_decays)))
        print(formatted_row.format(*(["----------"]+["----------------"]*len(hists))))
        for i,cut in enumerate(cuts):
            row = ["Cut %d"%i]
            for j,histProc in enumerate(hists):
                yield_, err = hists[j].GetBinContent(i+1), hists[j].GetBinError(i+1)
                row.append("%.2f +/- %.2f" % (yield_, err))
            print(formatted_row.format(*row))
        print("\n")
        print(f"Average: {eff_avg:.3f} +/- {eff_avg_err:.3f}")
        print(f"Min/max: {eff_min:.3f}/{eff_max:.3f}")
    sys.stdout = out_orig
    del canvas

    # make final efficiency plot eff_final, eff_final_err
    xMin, xMax = int(np.min(eff_final))-3, int(np.max(eff_final))+3

    h_pulls = ROOT.TH2F("pulls", "pulls", (xMax-xMin)*10, xMin, xMax, len(sigs)+1, 0, len(sigs)+1)
    g_pulls = ROOT.TGraphErrors(len(sigs)+1)

    ip = 0 # counter for TGraph
    g_pulls.SetPoint(ip, eff_avg, float(ip) + 0.5)
    g_pulls.SetPointError(ip, eff_avg_err, 0.)
    h_pulls.GetYaxis().SetBinLabel(ip+1, "Average")
    ip += 1

    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        g_pulls.SetPoint(ip, eff_final[i], float(ip) + 0.5)
        g_pulls.SetPointError(ip, eff_final_err[i], 0.)
        h_pulls.GetYaxis().SetBinLabel(ip+1, h_decays_labels[h_decay])
        ip += 1

    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.05)
    canvas.SetFillStyle(4000) # transparency?
    canvas.SetGrid(1, 0)
    canvas.SetTickx(1)

    xTitle = "Selection efficiency [%]"

    h_pulls.GetXaxis().SetTitleSize(0.04)
    h_pulls.GetXaxis().SetLabelSize(0.035)
    h_pulls.GetXaxis().SetTitle(xTitle)
    h_pulls.GetXaxis().SetTitleOffset(1)
    h_pulls.GetYaxis().SetLabelSize(0.055)
    h_pulls.GetYaxis().SetTickLength(0)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.SetNdivisions(506, 'XYZ')
    h_pulls.Draw("HIST 0")

    maxx = len(sigs)+1
    line = ROOT.TLine(eff_avg, 0, eff_avg, maxx)
    line.SetLineColor(ROOT.kGray)
    line.SetLineWidth(2)
    line.Draw("SAME")


    shade = ROOT.TGraph()
    shade.SetPoint(0, eff_avg-eff_avg_err, 0)
    shade.SetPoint(1, eff_avg+eff_avg_err, 0)
    shade.SetPoint(2, eff_avg+eff_avg_err, maxx)
    shade.SetPoint(3, eff_avg-eff_avg_err, maxx)
    shade.SetPoint(4, eff_avg-eff_avg_err, 0)
    #shade.SetFillStyle(3013)
    shade.SetFillColor(16)
    shade.SetFillColorAlpha(16, 0.35)
    shade.Draw("SAME F")

    g_pulls.SetMarkerSize(1.2)
    g_pulls.SetMarkerStyle(20)
    g_pulls.SetLineWidth(2)
    g_pulls.Draw('P0 SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(30) # 0 special vertical aligment with subscripts
    latex.DrawLatex(0.95, 0.925, f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}")

    latex.SetTextAlign(13)
    latex.SetTextFont(42)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.15, 0.96, "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}")

    txt = ROOT.TLatex()
    txt.SetTextSize(0.04)
    txt.SetTextColor(1)
    txt.SetTextFont(42)
    txt.SetNDC()
    txt.DrawLatex(0.2, 0.2, f"Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %")
    txt.DrawLatex(0.2, 0.15, f"Min/max: {eff_min:.2f}/{eff_max:.2f}")
    txt.Draw("SAME")

    if not os.path.isdir(f'{outDir}/cutflow'):
            os.system(f'mkdir -p {outDir}/cutflow')
    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/selection_efficiency{suffix}.{pl}")


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

