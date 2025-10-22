import os
import numpy as np
import pandas as pd

import ROOT as root
root.gROOT.SetBatch(True)
root.gStyle.SetOptStat(0)
root.gStyle.SetOptTitle(0)

from . import plotter
import atlasplots as aplt

h_decays_labels = {
    "bb": "H#rightarrow b#bar{b}", "cc": "H#rightarrow c#bar{c}", "ss": "H#rightarrow s#bar{s}", 
    "gg": "H#rightarrow gg", "mumu": "H#rightarrow#mu^{#plus}#mu^{#minus}", "tautau": "H#rightarrow#tau^{#plus}#tau^{#minus}", 
    "ZZ": "H#rightarrow ZZ*", "WW": "H#rightarrow WW*", "Za": "H#rightarrow Z#gamma", 
    "aa": "H#rightarrow#gamma#gamma", "inv": "H#rightarrow Inv"
}

h_decays_labels_pyplot = {
    "bb": r"$H\rightarrow b\bar{b}$", "cc": r"$H\rightarrow c\bar{c}$", "ss": r"$H\rightarrow s\bar{s}$", 
    "gg": r"$H\rightarrow gg$", "mumu": r"$H\rightarrow\mu^{+}\mu^{-}$", "tautau": r"$H\rightarrow\tau^{+}\tau^{-}$", 
    "ZZ": r"$H\rightarrow ZZ*$", "WW": r"$H\rightarrow WW^*$", "Za": r"$H\rightarrow Z\gamma$", 
    "aa": r"$H\rightarrow\gamma\gamma$", "inv": r"$H\rightarrow$ Inv"
}

h_decays_colors = {
    "bb": root.kBlack, "cc": root.kBlue , "ss": root.kRed, "gg": root.kGreen+1, "mumu": root.kOrange, 
    "tautau": root.kCyan, "ZZ": root.kGray, "WW": root.kGray+2, "Za": root.kGreen+2, "aa": root.kRed+2, 
    "inv": root.kBlue+2
}

procs_labels = {
    "ZH"        : "ZH",
    "ZmumuH"    : "Z(#mu^{+}#mu^{#minus})H",
    "ZeeH"      : "Z(e^{+}e^{#minus})H",
    "WW"        : "WW",
    "ZZ"        : "ZZ",
    "Zgamma"    : "Z/#gamma^{*} #rightarrow f#bar{f}+#gamma(#gamma)",
    "Rare"      : "Rare"
}

# colors from https://github.com/mpetroff/accessible-color-cycles
procs_colors = {
    "ZH"        : root.TColor.GetColor("#e42536"),
    "ZqqH"      : root.TColor.GetColor("#e42536"),
    "ZmumuH"    : root.TColor.GetColor("#e42536"),
    "ZnunuH"    : root.TColor.GetColor("#e42536"),
    "ZeeH"      : root.TColor.GetColor("#e42536"),
    "WW"        : root.TColor.GetColor("#f89c20"),
    "ZZ"        : root.TColor.GetColor("#5790fc"),
    "Zgamma"    : root.TColor.GetColor("#964a8b"),
    "Zqqgamma"  : root.TColor.GetColor("#964a8b"),
    "Rare"      : root.TColor.GetColor("#9c9ca1")
}

#__________________________________________________________
def getHist(hName, inputDir, target='', rebin=-1):
    
    tar = f'_{target}' if target!='' else ''
    fInName = f"{inputDir}/datacard{tar}.root"
    if os.path.exists(fInName):
        fIn = root.TFile(fInName)
    elif os.path.exists(fInName.replace("wzp6", "wz3p6")):
        fIn = root.TFile(fInName.replace("wzp6", "wz3p6"))
    else:
        print(f"ERROR: input file {fInName} not found")
        quit()
    h = fIn.Get(hName)
    h.SetDirectory(0)
    fIn.Close()
    if rebin!=-1:
        h.Rebin(rebin)
    return h

#__________________________________________________________
def range_hist(hist_original, x_min, x_max):

    # Get the bin numbers corresponding to the range
    bin_min = hist_original.FindBin(x_min)
    bin_max = hist_original.FindBin(x_max)
    

    hist_selected = root.TH1D(hist_original.GetName()+"new", "", bin_max - bin_min, x_min, x_max)

    for bin in range(bin_min, bin_max + 1):
        new_bin = bin - bin_min + 1  # Adjust for new histogram bin indexing
        hist_selected.SetBinContent(new_bin, hist_original.GetBinContent(bin))
        hist_selected.SetBinError(new_bin, hist_original.GetBinError(bin))  # Preserve errors

    # print(bin_min, bin_max, hist_original.Integral(), hist_selected.Integral())
    return hist_selected

#__________________________________________________________
def PlotDecays(hName, inputDir, outDir, h_decays, ecm=240, lumi=10.8, outName="", xMin=0, xMax=200, 
               yMin=0, yMax=-1, plot_file=['png'], xLabel="", yLabel="Events", logX=False, logY=True, rebin=-1, xLabels=[]):

    if outName == "": outName = hName

    leg = root.TLegend(.2, .925-(len(h_decays)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists = []
    h_base = getHist(f'{hName}_{h_decays[0]}', inputDir, target=h_decays[0], rebin=rebin)
    if rebin!=-1:
        h_base.Rebin(rebin)
    if xMin!=0 or xMax!=200:
        h_base = range_hist(h_base, xMin, xMax)
    if h_base.Integral() > 0:
        h_base.Scale(1./h_base.Integral())
    
    for h_decay in h_decays:
        h_sig = getHist(f'{hName}_{h_decay}', inputDir, target=h_decay, rebin=rebin)
        if rebin!=-1:
            h_sig.Rebin(rebin)
        if xMin!=0 or xMax!=200:
            h_sig = range_hist(h_sig, xMin, xMax)
        if h_sig.Integral() > 0:
            h_sig.Scale(1./h_sig.Integral())
        
        h_sig.Divide(h_base)

        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)
    
        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)

    cfg = {

        'logy'              : logY,
        'logx'              : logX,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : xLabel,
        'ytitle'            : yLabel,
            
        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy(1 if len(xLabels) == 0 else len(xLabels))
    if len(xLabels) > 0:
        dummy.GetXaxis().SetLabelSize(0.8*dummy.GetXaxis().GetLabelSize())
        dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
        for i,label in enumerate(xLabels): dummy.GetXaxis().SetBinLabel(i+1, label)
        dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST") 

    for i,hist in enumerate(hists):
        hist.Draw("SAME HIST")
    leg.Draw("SAME")
    
    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    root.gPad.SetTicks()
    root.gPad.RedrawAxis()

    if 'High-mass'  in xLabel: out = f'{outDir}/higgsDecays/high'
    elif 'Low-mass' in xLabel: out = f'{outDir}/higgsDecays/low'
    else :                     out = f'{outDir}/higgsDecays/nominal'

    if not os.path.isdir(out):
            os.system(f'mkdir -p {out}')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/{outName}.{pl}")



#__________________________________________________________
def makePlot(inputDir, outDir, cat, procs, target, sig='ZH', ecm=240, lumi=10.8, outName="", xMin=None, xMax=None, yMin=None, yMax=None, 
             ymin=0.9, ymax=1.1, xLabel="xlabel", yLabel="Events", logX=False, logY=True, rebin=-1, xLabels=[], plot_file=['png']):

    # aplt.set_atlas_style()

    fig, (ax1, ax2) = aplt.ratio_plot(figsize=(1000, 1000), hspace=0.05)
    ax1.set_pad_margins(left=0.15, right=0.05, top=0.055)
    ax2.set_pad_margins(left=0.15, right=0.05, bottom=0.25)
    leg = ax1.legend(loc=(.75, .99-len(procs)*0.06, .99, .90))
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    h_sig = getHist(f'{cat}_{sig}', inputDir, target=target, rebin=rebin)
    h_sig.SetLineColor(procs_colors[procs[0]])
    h_sig.SetLineWidth(3)
    h_sig.SetLineStyle(1)
    leg.AddEntry(h_sig, procs_labels[procs[0]], "L")

    h_asi = getHist(f'{cat}_data_{target}', inputDir, target=target, rebin=rebin)
    h_asi.SetMarkerStyle(8)
    h_asi.SetMarkerColor(root.kBlack)
    h_asi.SetMarkerSize(1)
    leg.AddEntry(h_asi, "Pseudo-data", "EP")

    st = root.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    for bkg in procs[1:]:
        h_bkg = getHist(f'{cat}_{bkg}', inputDir, target=target, rebin=rebin)

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(procs_colors[bkg])
        h_bkg.SetLineColor(root.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)
        leg.AddEntry(h_bkg, procs_labels[bkg], "F")

        st.Add(h_bkg)

    h_bkg_tot.Add(h_sig)
    h_bkg_tot.SetLineColor(root.kRed)
    h_bkg_tot.SetFillColor(0)
    h_bkg_tot.SetLineWidth(2)

    ax1.plot(st)
    asi = aplt.root_helpers.hist_to_graph(h_asi)
    ax1.plot(h_bkg_tot, "HIST")
    ax1.plot(asi, "EP", markerstyle=8)
    ax1.add_margins(top=0.20)

    ax1.set_xlim(xMin, xMax)
    ax1.set_ylim(yMin, yMax)

    line = root.TLine(ax1.get_xlim()[0], 1, ax1.get_xlim()[1], 1)
    ax2.plot(line)

    ratio = h_asi.Clone('ratio')
    ratio.Divide(h_bkg_tot)
    ratio_graph = aplt.root_helpers.hist_to_graph(ratio)
    ax2.plot(ratio_graph, "P0", markerstyle=8)

    # ax2.add_margins(top=0.1, bottom=0.1)
    ax2.set_xlim(xMin, xMax)
    ax2.set_ylim(ymin, ymax)

    ax2.set_xlabel(xLabel, titlesize=40, titlefont=43, labelfont=43, labelsize=35)
    ax1.set_ylabel(yLabel, titlesize=40, titlefont=43, labelfont=43, labelsize=35)
    ax2.set_ylabel('Data / MC', loc='centre', titlesize=40, titlefont=43, labelfont=43, labelsize=35)
    
    if logX: ax1.set_xscale('log')
    if logY: ax1.set_yscale('log')

    ax1.text(0.95, 0.945, f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}", size=0.04, font=42, align=30)
    ax1.text(0.15, 0.95, "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}", size=0.04, font=42, align=10)

    root.gPad.RedrawAxis("g")

    if   'High-mass' in xLabel: out = f'{outDir}/makePlot/high'
    elif 'Low-mass'  in xLabel: out = f'{outDir}/makePlot/low'
    else :                      out = f'{outDir}/makePlot/nominal'

    if not os.path.isdir(out):
            os.system(f'mkdir -p {out}')

    for pl in plot_file:
        fig.savefig(f"{out}/{outName}.{pl}")

#__________________________________________________________
def Bias(biasDir, nomDir, outDir, h_decays, plot_file=['png'], ecm=240, lumi=10.8):

    df = pd.read_csv(f'{biasDir}/bias_results.csv')
    mode, bias = df['mode'], df['bias'] * 100
    unc = np.loadtxt(f'{nomDir}/results.txt')[-1] * 10000

    leg = root.TLegend(.2, .925-(len(h_decays)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    if unc>bias.max(): xMin, xMax = -int(unc*1.2), int(unc*1.2)
    else:              xMin, xMax = -int(bias.max()*1.2), int(bias.max()*1.2)

    h_pulls = root.TH2F("pulls", "pulls", (xMax-xMin)*10, xMin, xMax, len(h_decays), 0, len(h_decays))
    g_pulls = root.TGraphErrors(len(h_decays))

    for i, h_decay in enumerate(h_decays):
        b = bias[np.where((mode==h_decay))[0]]
        g_pulls.SetPoint(i, b, float(i) + 0.5)
        h_pulls.GetYaxis().SetBinLabel(i+1, h_decays_labels[h_decay])

    canvas = root.TCanvas("c", "c", 800, 800)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.05)
    canvas.SetFillStyle(4000) # transparency?
    canvas.SetGrid(1, 0)
    canvas.SetTickx(1)

    xTitle = "Bias [%] (#times 100)"

    h_pulls.GetXaxis().SetTitleSize(0.04)
    h_pulls.GetXaxis().SetLabelSize(0.035)
    h_pulls.GetXaxis().SetTitle(xTitle)
    h_pulls.GetXaxis().SetTitleOffset(1)
    h_pulls.GetYaxis().SetLabelSize(0.055)
    h_pulls.GetYaxis().SetTickLength(0)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.SetNdivisions(506, 'XYZ')
    h_pulls.Draw("HIST 0")

    maxx = len(h_decays)

    lineup = root.TLine(unc, 0, unc, maxx)
    lineup.SetLineColor(root.kBlack)
    lineup.SetLineWidth(3)
    lineup.Draw("SAME")

    line = root.TLine(0, 0, 0, maxx)
    line.SetLineColor(root.kGray)
    line.SetLineWidth(2)
    line.Draw("SAME")

    linedw = root.TLine(-unc, 0, -unc, maxx)
    linedw.SetLineColor(root.kBlack)
    linedw.SetLineWidth(3)
    linedw.Draw("SAME")

    g_pulls.SetMarkerSize(1.2)
    g_pulls.SetMarkerStyle(20)
    g_pulls.SetLineWidth(2)
    g_pulls.Draw('P0 SAME')

    latex = root.TLatex()
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

    if not os.path.isdir(f'{outDir}/bias'):
            os.system(f'mkdir -p {outDir}/bias')

    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/bias/bias_test.{pl}")
