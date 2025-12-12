import os, sys, json, copy, ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

#__________________________________________________________
def getMetaInfo(proc, info='crossSection', remove=False,
                fcc='/cvmfs/fcc.cern.ch/FCCDicts',
                procFile="FCCee_procDict_winter2023_IDEA.json"):
    if os.getenv('FCCDICTSDIR') is None: fccdict = fcc
    else: fccdict = os.getenv('FCCDICTSDIR').split(':')[0]
    procFile = os.path.join(fccdict, '') + procFile 
    with open(procFile, 'r') as f:
        procDict=json.load(f)
    xsec = procDict[proc][info]
    if remove:
        # if 'HZZ' in proc:
        #     xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'))
        #     xsec     = getMetaInfo(proc) - xsec_inv
        if 'p8_ee_WW_ecm' in proc:
            xsec_ee   = getMetaInfo(proc.replace('WW', 'WW_ee'))
            xsec_mumu = getMetaInfo(proc.replace('WW', 'WW_mumu'))
            xsec      = getMetaInfo(proc) - xsec_ee - xsec_mumu
    return xsec

#__________________________________________________________
def getHist(hName, procs, inputDir, suffix='', rebin=1, lazy=True, proc_scales=1):
    hist = None
    for proc in procs:
        fInName = f"{inputDir}/{proc}{suffix}.root"
        if os.path.exists(fInName): fIn = ROOT.TFile(fInName)
        else:
            if lazy: continue
            else: sys.exit(f"ERROR: input file {fInName} not found")
        h = fIn.Get(hName)
        h.SetDirectory(0)
        if 'p8_ee_WW_ecm' in proc:
            xsec_old = getMetaInfo(proc)
            xsec_new = getMetaInfo(proc, remove=True)
            h.Scale( xsec_new/xsec_old )
        if hist == None: hist = h
        else: hist.Add(h)
        fIn.Close()
    hist.Rebin(rebin)
    hist.Scale(proc_scales)
    return hist

#__________________________________________________________
def make_pseudodata(inputDir, procs, procs_cfg, hName, target, cat, z_decays, h_decays, 
                    suffix='', proc_scales={}, ecm=240, variation=1.05, tot=True):

    print(f'----->[Info] Making pseudo data for {target} channel')
    print(f'----->[Info] Perturbation of the cross-section: {(variation-1)*100:.2f} %')

    xsec_tot, xsec_target, xsec_rest = 0, 0, 0

    if target!='inv':
        if tot: sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays]
        else:   sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in [cat]]    for y in h_decays]
    else:
        print('----->[Info] Replacing ZZ to ZZ_noInv to avoid overcounting')
        if tot: sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}'.replace('ZZ', 'ZZ_noInv') for x in z_decays] for y in h_decays]
        else:   sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}'.replace('ZZ', 'ZZ_noInv') for x in [cat]]    for y in h_decays]

    for h_decay, sig in zip(h_decays, sigs):
        xsec = sum([getMetaInfo(s, remove=True) for s in sig])
        xsec_tot += xsec
        if h_decay==target: 
            print(f'----->[Info] xsec for {target}: {xsec:.3e} pb-1')
            xsec_target += xsec
        else:               
            xsec_rest   += xsec

    print(f'xsec_tot: {xsec_tot:.3} pb-1')
    xsec_new = variation * xsec_tot
    print(f'xsec_new: {xsec_new:.3e} pb-1')
    xsec_delta = xsec_new - xsec_tot
    print(f'xsec_delta: {xsec_delta:.3e} pb-1')
    scale_target = (xsec_target + xsec_delta)/xsec_target
    print(f"----->[Info] Scale of the {target} channel: {scale_target:.3f}")

    hist_pseudo, h_bkg = None, None
    for proc in procs[1:]:
        if proc not in proc_scales: proc_scales[proc] = 1
        h = getHist(hName, procs_cfg[proc], inputDir, 
                    suffix=suffix, proc_scales=proc_scales[proc])
        if hist_pseudo==None: hist_pseudo = h.Clone('h_pseudo')
        else: hist_pseudo.Add(h)
        if h_bkg==None: h_bkg = h.Clone('h_bkg')
        else: h_bkg.Add(h)
        
    xsec_tot_new, hist_old, hist_new = 0, None, None
    for h_decay, sig in zip(h_decays, sigs):
        xsec = sum([getMetaInfo(s, remove=True) for s in sig])
        if 'ZH' not in proc_scales:proc_scales['ZH'] = 1
        h = getHist(hName, sig, inputDir, 
                    suffix=suffix, proc_scales=proc_scales['ZH'])
        # Old signal
        if hist_old==None: hist_old = h.Clone('h_old')
        else: hist_old.Add(h)

        if h_decay==target:
            h.Scale(scale_target)
            xsec_tot_new   += scale_target * xsec
        else: xsec_tot_new += xsec
        # New signal
        if hist_new==None: hist_new = h.Clone('h_new')
        else: hist_new.Add(h)
        hist_pseudo.Add(h)

    test = hist_pseudo.Clone('test')
    test.Add(h_bkg, -1)

    old, new, test = hist_old.Integral(), hist_new.Integral(), test.Integral()
    print(f'----->[Info] Added {(new/old-1)*100:.2f} % of signal to ZH production cross-section')
    print(f'----->[CROSS-CHECK] This quantity {xsec_tot_new/xsec_tot:.2f} should be equal to {variation}\n')
    return hist_pseudo


#__________________________________________________________
def make_pseudosignal(inputDir, hName, target, cat, z_decays, h_decays, v= False,
                      suffix='', proc_scales={}, ecm=240, variation=1.05, tot=True):

    if v:
        print(f'----->[Info] Making pseudo data for {target} channel')
        print(f'----->[Info] Perturbation of the cross-section: {(variation-1)*100:.2f} %')

    xsec_tot, xsec_target, xsec_rest = 0, 0, 0

    if tot: sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays]
    else:   sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in [cat]]    for y in h_decays]

    if target=='inv':
        print('----->[Info] Replacing ZZ to ZZ_noInv to avoid overcounting')
        for h in sigs: 
            for s in h: 
                s.replace('ZZ', 'ZZ_noInv')

    for h_decay, sig in zip(h_decays, sigs):
        xsec = sum([getMetaInfo(s, remove=True) for s in sig])
        xsec_tot += xsec
        if h_decay==target: 
            xsec_target += xsec
        else:               
            xsec_rest   += xsec

    xsec_new = variation * xsec_tot
    xsec_delta = xsec_new - xsec_tot
    scale_target = (xsec_target + xsec_delta)/xsec_target
    if v: print(f"----->[Info] Scale of the {target} channel: {scale_target:.3f}")

    hist_pseudo = None 
    xsec_tot_new, hist_old = 0, None
    for h_decay, sig in zip(h_decays, sigs):
        xsec = sum([getMetaInfo(s, remove=True) for s in sig])
        if 'ZH' not in proc_scales:proc_scales['ZH'] = 1
        h = getHist(hName, sig, inputDir, 
                    suffix=suffix, proc_scales=proc_scales['ZH'])
        # Old signal
        if hist_old==None: hist_old = copy.deepcopy(h)
        else: hist_old.Add(h)

        if h_decay==target:
            h.Scale(scale_target)
            xsec_tot_new   += scale_target * xsec
        else: xsec_tot_new += xsec
        if hist_pseudo==None: hist_pseudo = copy.deepcopy(h)
        else: hist_pseudo.Add(h)

    old, new = hist_old.Integral(), hist_pseudo.Integral()
    if v:
        print(f'----->[Info] Added {(new/old-1)*100:.2f} % of signal to ZH production cross-section')
        print(f'\n----->[CROSS-CHECK] This quantity {xsec_tot_new/xsec_tot:.2f} should be equal to {variation}\n')
    return hist_pseudo


#__________________________________________________________
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

    if plot_dc: print(f'\n{dc}\n')
