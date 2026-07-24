from ..tools.process import getHist

def get_hist(
        hName: str,
        procs: str | list[str],
        suffix: str,
        cat_idx_min: int,
        cat_idx_max: int,
        lumiScale: float | int,
        yield_nom: float | int,
        outName: str = '',
        normYields: bool = True):
    if isinstance(procs, str): procs = [procs]

    hist = getHist(hName, procs)
    hist.Scale(lumiScale)

    hist = hist.ProjectionX(f'hist_zh_{suffix}', cat_idx_min, cat_idx_max)
    if outName:    hist.SetName(outName)
    if normYields: hist.Scale(yield_nom / hist.Integral())

    return hist


def make_params(flavor: str, ecm: int, cat: int, mH: float | int, config: dict[int, dict[str, dict[str, dict[str, list[float | int]]]]]):

    import ROOT

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
