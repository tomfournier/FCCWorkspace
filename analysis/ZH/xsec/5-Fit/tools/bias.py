import os, json, ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

#__________________________________________________________
def getMetaInfo(proc, info='crossSection', remove=True, 
                fcc='/cvmfs/fcc.cern.ch/FCCDicts',
                procFile="FCCee_procDict_winter2023_IDEA.json"):
    if not ('eos' in procFile):
        if os.getenv('FCCDICTSDIR') is None: fccdict = fcc
        else:
            fccdict = os.getenv('FCCDICTSDIR').split(':')[0]
        procFile = os.path.join(fccdict, '') + procFile 
    with open(procFile, 'r') as f:
        procDict=json.load(f)
    xsec = procDict[proc][info]
    if remove:
        if 'HZZ' in proc: 
            xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'))
            xsec = xsec - xsec_inv
        if 'p8_ee_WW_ecm' in proc:
            xsec_ee   = getMetaInfo(proc.replace('WW_ecm', 'WW_ee_ecm'))
            xsec_mumu = getMetaInfo(proc.replace('WW_ecm', 'WW_mumu_ecm'))
            xsec = xsec-xsec_ee-xsec_mumu
    return xsec

#__________________________________________________________
def removeNegativeBins(hist):
    totNeg, tot = 0., 0.
    if "TH1" in hist.ClassName():
        pass
    elif "TH2" in hist.ClassName():
        pass

    elif "TH3" in hist.ClassName():
        nbinsX = hist.GetNbinsX()
        nbinsY = hist.GetNbinsY()
        nbinsZ = hist.GetNbinsZ()
        for x in range(1, nbinsX + 1):
            for y in range(1, nbinsY + 1):
                for z in range(1, nbinsZ + 1):
                    content = hist.GetBinContent(x, y, z)
                    tot += content
                    if content < 0:
                        totNeg += content
                        hist.SetBinContent(x, y, z, 0)
                        hist.SetBinError(x, y, z, 0)
    if totNeg != 0:
        print(f"WARNING: TOTAL {tot}, NEGATIVE {totNeg}, FRACTION {totNeg/tot}")
    return hist

#__________________________________________________________
def getSingleHist(hName, proc, inputDir, suffix='', lazy=True):
    fIn = ROOT.TFile(f"{inputDir}/{proc}{suffix}.root")
    h = fIn.Get(hName)
    h.SetDirectory(0)
    fIn.Close()
    h = removeNegativeBins(h)
    if 'HZZ' in proc: 
        xsec     = getMetaInfo(proc, remove=False)
        xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'), 
                               remove=False)
        h.Scale((xsec-xsec_inv)/xsec)
    if 'p8_ee_WW_ecm' in proc:
        xsec      = getMetaInfo(proc, remove=False)
        xsec_ee   = getMetaInfo(proc.replace('WW_ecm', 'WW_ee_ecm'), 
                                remove=False)
        xsec_mumu = getMetaInfo(proc.replace('WW_ecm', 'WW_mumu_ecm'), 
                                remove=False)
        h.Scale((xsec-xsec_ee-xsec_mumu)/xsec)
    return h

#__________________________________________________________
def getHists(inputDir, hName, procName, procs_cfg, suffix='', proc_scales={}):
    hist = None
    procs = procs_cfg[procName]
    for proc in procs:
        if not os.path.exists(f"{inputDir}/{proc}{suffix}.root"): continue
        h = getSingleHist(hName, proc, inputDir, suffix=suffix)
        if hist == None:
            hist = h
        else:
            hist.Add(h)
    if procName in proc_scales:
        hist.Scale(proc_scales[procName])
        print(f"SCALE {procName} with factor {proc_scales[procName]}")
    return hist


def make_pseudodata(inputDir, procs, procs_cfg, hName, target, z_decays, h_decays, 
                    suffix='', proc_scales={}, ecm=240, variation=1.0):

    print(f'----->[Info] Making pseudo data for {target} channel')
    print(f'----->[Info] Perturbation of the cross-section: {(variation-1)*100:.2f} %')

    xsec_tot = 0 # total cross-section
    xsec_target = 0  # nominal cross-section of the target process
    xsec_rest = 0 # cross-section of the rest

    sigProcs = procs_cfg[procs[0]] # get all signal processes
    for h_decay in h_decays:
        xsec = 0.
        for z_decay in z_decays:
            proc = f'wzp6_ee_{z_decay}H_H{h_decay}_ecm{ecm}'
            if not proc in sigProcs:
                continue
            xsec += getMetaInfo(proc)
            # print(f'xsec for {proc}: {getMetaInfo(proc, inputDir, ecm):.3e}')
        xsec_tot += xsec
        if h_decay != target:
            xsec_rest += xsec
        else:
            xsec_target += xsec

    xsec_new = variation*xsec_tot
    xsec_delta = xsec_new - xsec_tot # difference in cross-section
    print(f'----->[Info] Cross-section of {target} channel: {xsec_target*1e6:.1f} fb')
    scale_target = (xsec_target + xsec_delta)/xsec_target
    scale_rest = (xsec_rest - xsec_delta)/xsec_rest

    # print(f"----->[Info] Total cross-section of the signal: {xsec_tot*1e6:.1f} fb")
    # print(f"----->[Info] New cross-section: {xsec_new*1e6:.1f} fb")
    # print(f"----->[Info] Difference between new and previous cross-section: {xsec_delta*1e6:.1f} fb")
    print(f"----->[Info] Scale of the {target} channel: {scale_target:.3f}")
    # print(f"----->[Info] Scale of the rest: {scale_rest:.3f}")

    hist_pseudo = None # start with all backgrounds
    for proc in procs[1:]:
        h = getHists(inputDir, hName, proc, procs_cfg, 
                     suffix=suffix, proc_scales=proc_scales)
        if hist_pseudo == None:
            hist_pseudo = h
        else:
            hist_pseudo.Add(h)

    xsec_tot_new = 0
    for h_decay in h_decays:
        xsec = 0.
        hist = None
        for z_decay in z_decays:
            proc = f'wzp6_ee_{z_decay}H_H{h_decay}_ecm{ecm}'
            if not proc in sigProcs: continue
            if not os.path.exists(f"{inputDir}/{proc}{suffix}.root"): continue
            xsec += getMetaInfo(proc)
            h = getSingleHist(hName, proc, inputDir, 
                              suffix=suffix)
            if hist == None:
                hist = h
            else:
                hist.Add(h)
        if h_decay == target:
            hist.Scale(scale_target)
            xsec_tot_new += xsec*scale_target
        else:
            # hist.Scale(scale_rest)
            xsec_tot_new += xsec*scale_rest
        hist_pseudo.Add(hist)
    
    print(f'----->[CROSS-CHECK] This quantity {xsec_tot*1e6:.2f} must be equal to this one {xsec_tot_new*1e6:.2f}')
    print(f'\tDifference: {np.abs(xsec_tot-xsec_tot_new)*1e6:.2f}')
    return hist_pseudo

def make_datacard(outDir, procs, target, bkg_unc, categories, 
                  freezeBackgrounds=False, floatBackgrounds=False, plot_dc=False):
    
    p = -1 if floatBackgrounds else 1
    procs_idx = [0, p*1, p*2, p*3, p*4]

    cats_str           = "".join([f"{cat:{' '}{'<'}{12}}"  for cat  in categories])
    procs_str          = "".join([f"{proc:{' '}{'<'}{12}}" for proc in procs] * len(categories))
    cats_procs_str     = "".join([f"{cat:{' '}{'<'}{12}}"  for cat  in categories for _ in range(len(procs))])
    cats_procs_idx_str = "".join([f"{str(proc_idx):{' '}{'<'}{12}}" for proc_idx in procs_idx] * len(categories))
    rates_cats         = "".join([f"{'-1':{' '}{'<'}{12}}"]*(len(categories)))
    rates_procs        = "".join([f"{'-1':{' '}{'<'}{12}}"]*(len(categories)*len(procs)))

    ## datacard header
    dc = ""
    dc += f"imax *\n"
    dc += f"jmax *\n"
    dc += "kmax *\n"
    dc += "#########################################################################################\n"
    dc += f"shapes *        * datacard_{target}.root $CHANNEL_$PROCESS\n" # $CHANNEL_$PROCESS_$SYSTEMATIC\n"
    dc += f"shapes data_obs * datacard_{target}.root $CHANNEL_data_{target}\n"
    dc += "#########################################################################################\n"
    dc += f"bin                        {cats_str}\n"
    dc += f"observation                {rates_cats}\n"
    dc += f"########################################################################################\n"
    dc += f"bin                        {cats_procs_str}\n"
    dc += f"process                    {procs_str}\n"
    dc += f"process                    {cats_procs_idx_str}\n"
    dc += f"rate                       {rates_procs}\n"
    dc += f"########################################################################################\n"

    if not freezeBackgrounds and not floatBackgrounds:
        if False:
            dc_tmp = f"{f'bkg_norm':{' '}{'<'}{15}} {'lnN':{' '}{'<'}{5}} {'-':{' '}{'<'}{12}}"
            for i in range(len(procs)-1):
                dc_tmp += f"{str(bkg_unc):{' '}{'<'}{12}}"
            dc += dc_tmp
        else:
            for i, proc in enumerate(procs):
                if i==0: continue # no signal
                dc_tmp = f"{f'norm_{proc}':{' '}{'<'}{15}} {'lnN':{' '}{'<'}{10}} "
                for cat in categories:
                    for proc1 in procs:
                        val = str(bkg_unc) if proc==proc1 else "-"
                        dc_tmp += f"{val:{' '}{'<'}{12}}"
                dc += f"{dc_tmp}\n"

    else:
        for proc in procs:
            dc_tmp = f"{f'norm_{proc}':{' '}{'<'}{15}} {'lnN':{' '}{'<'}{10}} "
            for cat in categories:
                for proc1 in procs:
                    val = str(1.000000005) if proc1==procs[0] else '-'
                    dc_tmp += f"{val:{' '}{'<'}{12}}"
        dc += dc_tmp

    print('----->[Info] Saving datacard')
    f = open(f"{outDir}/datacard_{target}.txt", 'w')
    f.write(dc)
    f.close()
    print(f'----->[Info] Saved datacard in {outDir}/datacard_{target}.txt')

    if plot_dc:
        print(f'\n{dc}\n')
