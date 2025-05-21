import sys
import os
import math
import copy
import array
import importlib

userConfig = importlib.import_module('userConfig')

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import tools.plotter as plotter # type: ignore

def getHist(hName, procs, rebin=-1):
    hist = None
    for proc in procs:
        fInName = f"{inputDir}/{proc}.root"
        if os.path.exists(fInName):
            fIn = ROOT.TFile(fInName)
        elif os.path.exists(fInName.replace("wzp6", "wz3p6")):
            fIn = ROOT.TFile(fInName.replace("wzp6", "wz3p6"))
        else:
            print(f"ERROR: input file {fInName} not found")
            quit()
        h = fIn.Get(hName)
        h.SetDirectory(0)
        if hist == None:
            hist = h
        else:
            hist.Add(h)
        fIn.Close()
    if rebin!=-1:
        hist.Rebin(rebin)
    return hist

def makeCutFlow(hName="cutFlow", cuts=[], labels=[], sig_scale=1.0, yMin=1e6, yMax=1e10):

    leg = ROOT.TLegend(.55, 0.99-(len(procs))*0.06, .99, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    hists_yields = []
    significances = []
    h_sig = getHist(hName, procs_cfg[procs[0]])

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
        h_bkg = getHist(hName, procs_cfg[bkg])

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
        for j,histProc in enumerate(hists_yields):
            nbkg = nbkg + histProc.GetBinContent(i+1)
            print(histProc.GetBinContent(i+1))
        if (nsig+nbkg) == 0:
            print(f"Cut {cut} zero yield sig+bkg")
            s = -1
        else:
            s = nsig / (nsig + nbkg)**0.5
        print(i, cut, s)
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

    if not os.path.isdir(f'{outDir}'):
            os.system(f'mkdir -p {outDir}')

    plotter.aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{outDir}/{hName}.{userConfig.plot_file}")

    out_orig = sys.stdout
    with open(f"{outDir}/{hName}.txt", 'w') as f:
        sys.stdout = f

        formatted_row = '{:<10} {:<25} ' + ' '.join(['{:<25}']*len(procs))
        print(formatted_row.format(*(["Cut", "Significance"]+procs)))
        print(formatted_row.format(*(["----------"]+["-----------------------"]*(len(procs)+1))))
        for i,cut in enumerate(cuts):
            row = ["Cut %d"%i, "%.3f"%significances[i]]
            for histProc in range(hists_yields):
                yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
                row.append("%.4e +/- %.2e" % (yield_, err))

            print(formatted_row.format(*row))
    sys.stdout = out_orig


def makeCutFlowHiggsDecays(hName="cutFlow", outName="", cuts=[], cut_labels=[], yMin=0, yMax=150, z_decays=[], h_decays=[], h_decays_labels=[], h_decays_colors=[]):

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
        h_sig = getHist(hName, sig)
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
    txt.DrawLatex(0.2, 0.2, f"Avg. eff.: {eff_avg:.2f}#pm{eff_avg_err:.2f} %")
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

    if not os.path.isdir(f'{outDir}/higgsDecays'):
            os.system(f'mkdir -p {outDir}/higgsDecays')

    canvas.SaveAs(f"{outDir}/higgsDecays/{outName}.{userConfig.plot_file}")

    out_orig = sys.stdout
    
    with open(f"{outDir}/higgsDecays/{outName}.txt", 'w') as f:
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
    if final_state == "mumu":
        xMin, xMax = 68, 74
        if len(z_decays) > 1:
            xMin, xMax = 0, 5
        xMin, xMax = 0, 100
    if final_state == "ee":
        xMin, xMax = 58, 66
        if len(z_decays) > 1:
            xMin, xMax = 0, 5
    if final_state == "qq":
        xMin, xMax = 55, 85
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
    # line.Draw("SAME")


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
    txt.DrawLatex(0.2, 0.2, f"Avg. eff.: {eff_avg:.2f}#pm{eff_avg_err:.2f} %")
    txt.DrawLatex(0.2, 0.15, f"Min/max: {eff_min:.2f}/{eff_max:.2f}")
    txt.Draw("SAME")

    if not os.path.isdir(f'{outDir}/higgsDecays'):
            os.system(f'mkdir -p {outDir}/higgsDecays')

    canvas.SaveAs(f"{outDir}/higgsDecays/{outName}_finalSelection.{userConfig.plot_file}")


def makePlotHiggsDecays(hName, outName="", xMin=0, xMax=100, yMin=1, yMax=1e5, xLabel="", yLabel="Events", logX=False, logY=True, rebin=-1, xLabels=[]):

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
    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        h_sig = getHist(hName, sig)
        if rebin!=-1:
            h_sig.Rebin(rebin)
        if h_sig.Integral() > 0:
            h_sig.Scale(1./h_sig.Integral())

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
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

    if not os.path.isdir(f'{outDir}/higgsDecays'):
            os.system(f'mkdir -p {outDir}/higgsDecays')

    canvas.SaveAs(f"{outDir}/higgsDecays/{outName}.{userConfig.plot_file}")
    canvas.Close()


def makePlotSignalRatios(hName, outName="", xMin=0, xMax=100, yMin=1, yMax=1e5, xLabel="xlabel", yLabel="Events", rebin=-1):

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
    h_tot = None
    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        h_sig = getHist(hName, sig)
        if rebin!=-1:
            h_sig.Rebin(rebin)
        # h_sig.Scale(1./h_sig.Integral())

        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)

        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)
        
        if h_tot == None:
            h_tot = h_sig.Clone("h_tot")
        else:
            h_tot.Add(h_sig)

    h_tot.Scale(1./h_tot.Integral())

    for h in hists:
        h.Scale(1./h.Integral())
        h.Divide(h_tot)

    cfg = {

        'logy'              : False,
        'logx'              : False,

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
    dummy = plotter.dummy()
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

    if not os.path.isdir(f'{outDir}/higgsDecays'):
            os.system(f'mkdir -p {outDir}/higgsDecays')

    canvas.SaveAs(f"{outDir}/higgsDecays/{outName}.{userConfig.plot_file}")
    canvas.Close()


def makePlot(hName, outName="", xMin=0, xMax=100, yMin=1, yMax=1e5, xLabel="xlabel", yLabel="Events", logX=False, logY=True, rebin=-1, legPos=[0.3, 0.75, 0.9, 0.9], sig_scale=1, xLabels=[]):

    if outName == "":
        outName = hName

    st = ROOT.THStack()
    st.SetName("stack")

    leg = ROOT.TLegend(.55, 0.99-(len(procs))*0.06, .99, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    h_sig = getHist(hName, procs_cfg[procs[0]])
    h_sig.SetLineColor(procs_colors[procs[0]])
    h_sig.SetLineWidth(3)
    h_sig.SetLineStyle(1)
    h_sig.Scale(sig_scale)
    if sig_scale != 1:
        leg.AddEntry(h_sig, f"{procs_labels[procs[0]]} (#times {int(sig_scale)})", "L")
    else:
        leg.AddEntry(h_sig, procs_labels[procs[0]], "L")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    for i,bkg in enumerate(procs[1:]):
        h_bkg = getHist(hName, procs_cfg[bkg])

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(procs_colors[bkg])
        h_bkg.SetLineColor(ROOT.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)

        leg.AddEntry(h_bkg, procs_labels[bkg], "F")
        st.Add(h_bkg)


    if yMax < 0:
        if logY:
            yMax = math.ceil(max([h_bkg_tot.GetMaximum(), h_sig.GetMaximum()])*10000)/10.
        else:
            yMax = 1.4*max([h_bkg_tot.GetMaximum(), h_sig.GetMaximum()])

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
    st.Draw("HIST SAME")

    h_bkg_tot.SetLineColor(ROOT.kBlack)
    h_bkg_tot.SetFillColor(0)
    h_bkg_tot.SetLineWidth(2)
    #hTot_err.Draw("E2 SAME")
    h_bkg_tot.Draw("HIST SAME")
    h_sig.Draw("HIST SAME")
    leg.Draw("SAME")
    
    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

    if not os.path.isdir(f'{outDir}'):
            os.system(f'mkdir -p {outDir}')

    canvas.SaveAs(f"{outDir}/{outName}.{userConfig.plot_file}")
    canvas.Close()


def significance(hName, xMin=-10000, xMax=10000, reverse=False):

    h_sig = getHist(hName, procs_cfg[procs[0]])
    sig_tot = h_sig.Integral()

    bkgs_procs = []
    for i,bkg in enumerate(procs[1:]):
        bkgs_procs.extend(procs_cfg[bkg])

    h_bkg = getHist(hName, bkgs_procs)
    x, y, l = [], [], []

    for i in range(1, h_sig.GetNbinsX()+1):
        if reverse:
            iStart = 1
            iEnd = i
        else:
            iStart = i
            iEnd = h_sig.GetNbinsX()+1
        center = h_sig.GetBinCenter(i)
        if center > xMax or center < xMin:
            continue
        sig = h_sig.Integral(iStart, iEnd)
        bkg = h_bkg.Integral(iStart, iEnd)
        if (sig+bkg) <= 0 or sig_tot <= 0:
            significance = 0
            sig_loss = 0
        else:
            significance = sig / (sig + bkg)**0.5
            sig_loss = sig / sig_tot
        print(f"{i} {center:.5f} {significance:.5f} {sig_loss:.5f}")
        x.append(center)
        y.append(significance)
        l.append(sig_loss)


    graph_sign = ROOT.TGraph(len(x), array.array('d', x), array.array('d', y))
    graph_l = ROOT.TGraph(len(x), array.array('d', x), array.array('d', l))
    max_y = max(y)
    max_index = y.index(max_y)
    max_x = x[max_index]
    max_l = l[max_index]

    canvas = ROOT.TCanvas("", "", 800, 800)
    graph_sign.SetMarkerStyle(20)
    graph_sign.SetMarkerColor(ROOT.kBlue)
    graph_sign.GetXaxis().SetRangeUser(xMin, xMax)
    graph_sign.Draw("AP")
    canvas.Update()

    # Add a marker for the maximum point
    max_marker = ROOT.TMarker(max_x, max_y, 20)
    max_marker.SetMarkerColor(ROOT.kRed)
    max_marker.SetMarkerSize(1.5)
    max_marker.Draw()

    rightmax = 1.0
    print(ROOT.gPad.GetUymin())
    print(ROOT.gPad.GetUymax())
    scale =  ROOT.gPad.GetUymax()/rightmax
    rightmin = ROOT.gPad.GetUymin()/ROOT.gPad.GetUymax()
    graph_l.Scale(scale)
    graph_l.SetLineColor(ROOT.kRed)
    graph_l.SetLineWidth(2)
    graph_l.Draw("SAME L")

    axis_r = ROOT.TGaxis(ROOT.gPad.GetUxmax(),ROOT.gPad.GetUymin(), ROOT.gPad.GetUxmax(), ROOT.gPad.GetUymax(),rightmin,rightmax,510,"+L")
    axis_r.SetLineColor(ROOT.kRed)
    axis_r.SetLabelColor(ROOT.kRed)
    axis_r.Draw()

    # Add a text box to indicate the maximum value
    text = ROOT.TLatex()
    text.SetTextSize(0.03)
    text.SetTextColor(ROOT.kBlack)
    text.DrawLatexNDC(0.1, 0.92, f"Max: x = {max_x}, y = {max_y:.5f}, signal loss = {max_l:.5f}")

    suffix = "_reverse" if reverse else ""
    if not os.path.isdir(f'{outDir}/significance'):
            os.system(f'mkdir -p {outDir}/significance')
    canvas.SaveAs(f"{outDir}/significance/{hName}{suffix}.{userConfig.plot_file}")
    canvas.Close()

if __name__ == "__main__":

    ecm = userConfig.ecm
    final_state = userConfig.final_state
    lumi = userConfig.intLumi

    z_decays = ['mumu', 'ee']
    h_decays = ["bb", "cc", "gg", "ss", "mumu", "tautau", "ZZ", "WW", "Za", "aa"]

    h_decays_labels = {"bb": "H#rightarrowb#bar{b}", "cc": "H#rightarrowc#bar{c}", "ss": "H#rightarrows#bar{s}", 
                       "gg": "H#rightarrowgg", "mumu": "H#rightarrow#mu^{#plus}#mu^{#minus}", "tautau": "H#rightarrow#tau^{#plus}#tau^{#minus}", 
                       "ZZ": "H#rightarrowZZ*", "WW": "H#rightarrowWW*", "Za": "H#rightarrowZ#gamma", 
                       "aa": "H#rightarrow#gamma#gamma", "inv": "H#rightarrowInv"}
    h_decays_colors = {"bb": ROOT.kBlack, "cc": ROOT.kBlue , "ss": ROOT.kRed, "gg": ROOT.kGreen+1, "mumu": ROOT.kOrange, 
                       "tautau": ROOT.kCyan, "ZZ": ROOT.kGray, "WW": ROOT.kGray+2, "Za": ROOT.kGreen+2, "aa": ROOT.kRed+2, 
                       "inv": ROOT.kBlue+2}

    procs_cfg = {
        "ZH"        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays],
        "ZmumuH"    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
        "ZeeH"      : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in ["ee"] for y in h_decays],
        "WW"        : [f'p8_ee_WW_ecm{ecm}'],
        "ZZ"        : [f'p8_ee_ZZ_ecm{ecm}'],
        "Zgamma"    : [f'wzp6_ee_tautau_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
                       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],
        "Rare"      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
                       f'wzp6_gaga_mumu_60_ecm{ecm}', f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
                       f'wzp6_gammae_eZ_Zee_ecm{ecm}', f'wzp6_gaga_ee_60_ecm{ecm}', 
                       f'wzp6_gaga_tautau_60_ecm{ecm}', f'wzp6_ee_nuenueZ_ecm{ecm}'],
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
        "ZH"        : ROOT.TColor.GetColor("#e42536"),
        "ZqqH"      : ROOT.TColor.GetColor("#e42536"),
        "ZmumuH"    : ROOT.TColor.GetColor("#e42536"),
        "ZnunuH"    : ROOT.TColor.GetColor("#e42536"),
        "ZeeH"      : ROOT.TColor.GetColor("#e42536"),
        "WW"        : ROOT.TColor.GetColor("#f89c20"),
        "ZZ"        : ROOT.TColor.GetColor("#5790fc"),
        "Zgamma"    : ROOT.TColor.GetColor("#964a8b"),
        "Zqqgamma"  : ROOT.TColor.GetColor("#964a8b"),
        "Rare"      : ROOT.TColor.GetColor("#9c9ca1")
    }

    if final_state == "ee" or final_state == "mumu":

        print(f'----->[Info] Making plots for {final_state} channel')

        inputDir = userConfig.loc.MODEL
        outDir = userConfig.loc.MODEL_PLOTS

        z_decays = [final_state] # only mumu or ee
        procs = [f"Z{final_state}H", "WW", "ZZ", "Zgamma", "Rare"] # first must be signal

        cuts = ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"] # , "cut6"]
        if final_state=='mumu':
            cut_labels = ["All events", "#geq 1 #mu^{#pm} + ISO", "#geq 2 #mu^{#pm} + OS", "86 < m_{#mu^{+}#mu^{#minus}} < 96", "20 < p_{#mu^{+}#mu^{#minus}} < 70", "120 < m_{rec} < 140"] # , "|cos#theta_{miss}| < 0.98"]
        else:
            cut_labels = ["All events", "#geq 1 e^{#pm} + ISO", "#geq 2 e^{#pm} + OS", "86 < m_{e^{+}e^{#minus}} < 96", "20 < p_{e^{+}e^{#minus}} < 70", "120 < m_{rec} < 140"] # , "|cos#theta_{miss}| < 0.98"]

        # makeCutFlow(f"{final_state}_cutFlow", cuts, cut_labels, sig_scale=1, yMin=1e4, yMax=1e10)
        makePlotHiggsDecays(f"{final_state}_mva_score", outName="mva_score", xMin=0, xMax=1, yMin=1e-4, yMax=1e1, xLabel="MVA score", yLabel="Events", logY=True)
        makeCutFlowHiggsDecays(f"{final_state}_cutFlow", outName="cutFlow", cuts=cuts, cut_labels=cut_labels, yMin=40, yMax=150, z_decays=z_decays, h_decays=h_decays, h_decays_labels=h_decays_labels, h_decays_colors=h_decays_colors)
        