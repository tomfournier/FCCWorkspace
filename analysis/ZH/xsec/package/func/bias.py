from ..tools.process import getMetaInfo, getHist

def _signal_lists(cat: str,
                  z_decays: list[str],
                  h_decays: list[str],
                  target: str,
                  ecm: int = 240,
                  tot: bool = True
                  ) -> list[list[str]]:
    
    cats = z_decays if tot else [cat]
    template = f'wzp6_ee_{{0}}H_H{{1}}_ecm{ecm}'
    h_list = [y.replace('ZZ', 'ZZ_noInv') if target=='inv' else y for y in h_decays]

    return [[template.format(x, y) for x in cats] for y in h_list]

def _scaling(sigs: list[list[str]], 
             h_decays: list[str], 
             target: str, 
             variation: float, 
             verbose: bool = True
             ) -> tuple[float, 
                        float, 
                        float]:
    xsec_tot = xsec_target = 0.0

    for h_decay, sig in zip(h_decays, sigs):
        xsec = sum(getMetaInfo(s, remove=True) for s in sig)
        xsec_tot += xsec
        if h_decay==target:
            xsec_target = xsec

    xsec_delta = xsec_tot * (variation - 1.0)
    scale_target = 1.0 + xsec_delta / xsec_target
    xsec_tot_new = xsec_tot * variation

    if verbose:
        print(f'----->[Info] Making pseudo data for {target} channel')
        print(f'----->[Info] Perturbation: {(variation-1)*100:.2f} %, Scale: {scale_target:.3f}')
        print(f'----->[Info] Target xsec: {xsec_target:.3e} pb-1')

    return scale_target, xsec_tot, xsec_tot_new

#_______________________________________________________
def make_pseudodata(hName: str, 
                    inDir: str, 
                    procs: list[str], 
                    processes: dict[str, list[str]], 
                    cat: str, 
                    z_decays: list[str], 
                    h_decays: list[str], 
                    target: str,
                    ecm: int = 240, 
                    variation: float = 1.05, 
                    suffix: str = '', 
                    proc_scales: dict[str, float] = None, 
                    tot: bool = True):
    
    if proc_scales is None:
        proc_scales = {}

    sigs = _signal_lists(cat, z_decays, h_decays, target, ecm=ecm, tot=tot)

    scale_target, xsec_tot, xsec_tot_new = _scaling(
        sigs, h_decays, target, variation, verbose=True
    )

    hist_pseudo = hist_old = hist_new = h_bkg = None

    for proc in procs[1:]:
        scale = proc_scales.get(proc, 1.0)
        h = getHist(hName, processes[proc], inDir, 
                    suffix=suffix, proc_scale=scale)
        
        if hist_pseudo is None: 
            hist_pseudo = h.Clone('h_pseudo')
            h_bkg = h.Clone('h_bkg')
        else: 
            hist_pseudo.Add(h)
            h_bkg.Add(h)

    zh_scale = proc_scales.get('ZH', 1.0)
        
    for h_decay, sig in zip(h_decays, sigs):
        h = getHist(hName, sig, inDir, 
                    suffix=suffix, proc_scale=zh_scale)
        # Old signal
        if hist_old is None: 
            hist_old = h.Clone('h_old')
        else:
            hist_old.Add(h)

        if h_decay==target:
            h.Scale(scale_target)

        # New signal
        if hist_new is None: 
            hist_new = h.Clone('h_new')
        else: 
            hist_new.Add(h)
        
        hist_pseudo.Add(h)

    delta_pct = (hist_new.Integral() / hist_old.Integral() - 1.0) * 100
    scale_ratio = xsec_tot_new / xsec_tot

    print(f'----->[Info] Signal increased by {delta_pct:.2f} %')
    print(f'----->[CROSS-CHECK] Scale ratio {scale_ratio:.2f} vs target {variation}\n')
    return hist_pseudo


#________________________________________________________
def make_pseudosignal(hName: str, 
                      inDir: str, 
                      target: str, 
                      cat: str, 
                      z_decays: list[str], 
                      h_decays: list[str], 
                      ecm: int = 240, 
                      variation: float = 1.05, 
                      suffix: str = '', 
                      proc_scales: dict[str, float] = None,
                      v: bool = False, 
                      tot: bool = True):
    
    if proc_scales is None:
        proc_scales = {}

    sigs = _signal_lists(cat, z_decays, h_decays, target, ecm=ecm, tot=tot)
    scale_target, xsec_tot, xsec_tot_new = _scaling(
        sigs, h_decays, target, variation, verbose=v
    )


    hist_pseudo = hist_old = None
    zh_scale = proc_scales.get('ZH', 1.0)

    for h_decay, sig in zip(h_decays, sigs):
        h = getHist(hName, sig, inDir, 
                    suffix=suffix, 
                    proc_scale=zh_scale)
        
        # Old signal
        if hist_old is None: 
            hist_old = h.Clone('h_old')
        else: 
            hist_old.Add(h)

        if h_decay==target:
            h.Scale(scale_target)

        if hist_pseudo is None: 
            hist_pseudo = h.Clone('h_pseudo')
        else: 
            hist_pseudo.Add(h)

    if v:
        delta_pct = (hist_pseudo.Integral() / hist_old.Integral() - 1.0) * 100
        print(f'----->[Info] Signal increased by {delta_pct:.2f} %')
        print(f'----->[CROSS-CHECK] Scale ratio {xsec_tot_new/xsec_tot:.2f} vs target {variation}\n')
    return hist_pseudo


#__________________________________________
def make_datacard(outDir: str, 
                  procs: list[str], 
                  target: str, 
                  bkg_unc: float, 
                  categories: list[str], 
                  freezeBkgs: bool = False,
                  floatBkgs: bool = False, 
                  plot_dc: bool = False
                  ) -> None:
    
    nprocs = len(procs)
    ncats = len(categories)

    col_w = 12

    cats_str       = ''.join([f'{cat:<{col_w}}'  for cat  in categories])
    procs_str      = ''.join([f'{proc:<{col_w}}' for proc in procs]) * ncats
    cats_procs_str = ''.join([f'{cat:<{col_w}}'  for cat  in categories for _ in range(nprocs)])

    p = -1 if floatBkgs else 1
    procs_idx = [0] + [p*i for i in range(1, nprocs)]
    cats_procs_idx_str = ''.join([f'{idx:<{col_w}}' for idx in procs_idx]) * ncats

    rates_cats  = f'{-1:<{col_w}}' * ncats
    rates_procs = f'{-1:<{col_w}}' * (ncats * nprocs)

    ## datacard header
    sep = '#' * (22 + len(cats_procs_str))
    dc_lines = [
        'imax *',
        'jmax *',
        'kmax *',
        sep,
        f'shapes *        * datacard_{target}.root $CHANNEL_$PROCESS', # $CHANNEL_$PROCESS_$SYSTEMATIC'
        f'shapes data_obs * datacard_{target}.root $CHANNEL_data_{target}',
        sep,
        f'bin                        {cats_str}',
        f'observation                {rates_cats}',
        sep,
        f'bin                        {cats_procs_str}',
        f'process                    {procs_str}',
        f'process                    {cats_procs_idx_str}',
        f'rate                       {rates_procs}',
        sep,
    ]

    if not freezeBkgs and not floatBkgs:
        for proc in procs[1:]:
            vals = ''.join(
                f"{bkg_unc if p1==proc else '-':<{col_w}}"
                for _ in categories for p1 in procs
            )
            dc_lines.append(f"{'norm_'+proc:<15} {'lnN':<10} {vals}")
            
    else:
        vals = ''.join(
            f"{1.000000005 if p==procs[0] else '-':<{col_w}}"
            for _ in categories for p in procs
        )
        dc_lines.append(f"{'norm_'+procs[0]:<15} {'lnN':<10} {vals}")

    dc = '\n'.join(dc_lines) + '\n'

    fOut = f'{outDir}/datacard_{target}.txt'
    print(f'----->[Info] Saving datacard to {fOut}')
    with open(fOut, 'w') as f:
        f.write(dc)

    if plot_dc: 
        print(f'\n{dc}\n')
