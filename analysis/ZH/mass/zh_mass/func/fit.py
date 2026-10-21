import json, math, array, subprocess, ROOT

from ..userConfig import PathObj
from ..tools.process import getHist
from ..plots.fit import (
    plot_mass_scan,
    plot_mass_breakdown_curves,
    plot_mass_breakdown_impacts
)


class RooModel:
    """Own a RooFit model and all objects needed by its dependency graph."""

    def __init__(self, name: str):
        self.name = name
        self.objects: dict[str, object] = {}

    def add(self, name: str, obj):
        self.objects[name] = obj
        return obj

    def get(self, name: str):
        return self.objects[name]

    def extend(self, other: 'RooModel') -> 'RooModel':
        self.objects.update(other.objects)
        return self

    def import_to(self, workspace: ROOT.RooWorkspace, *names: str) -> None:
        objects = (self.objects[name] for name in names) if names else self.objects.values()
        for obj in objects:
            workspace.Import(obj)

    def fit_to(self, data, *options):
        return self.get('fit').fitTo(data, *options)


class RooModelBuilder:
    """Build a :class:`RooModel` from a declarative signal specification."""

    def __init__(self, spec: dict):
        self.spec: dict[str, callable | list[str | dict[str, str]] | str] = spec

    @staticmethod
    def get_src_val(source: dict, key: str, idx: int | None = None):
        value = source[key]
        if idx is None: return value
        return value[idx]

    @staticmethod
    def suffixed(base: str, suffix: str = ''):
        """Append a model-point suffix when one is configured."""
        return base if not suffix else f'{base}_{suffix}'

    @staticmethod
    def argl(*items, name: str | None = None):
        arg_list = ROOT.RooArgList()
        if name is not None: arg_list.setName(name)
        for item in items: arg_list.add(item)
        return arg_list

    @staticmethod
    def get_pdf(
            kind: str,
            name: str,
            title: str,
            mrec,
            params: dict,
            spec: dict,
            observables: set[str] | None = None):
        """Instantiate a ROOT PDF from a declarative component specification.

        ``kind`` identifies the ROOT class unless ``root_class`` overrides it.
        Entries in ``args`` resolve observable names to ``mrec`` and parameter
        names to objects in ``params``; other values are passed through literally.
        """

        # Names in this set refer to the observable passed to the builder.
        obs = observables or {'recoilmass', 'mrecoil', 'observable', 'x'}

        # Resolve the configured ROOT class, for example RooGaussian or RooCBShape.
        root_kind = spec.get('root_class', kind)
        comp_cls = getattr(ROOT, root_kind, None)
        if comp_cls is None:
            raise ValueError(f'ROOT has no component class {root_kind!r}')

        if 'args' in spec:
            # Convert parameter and observable names into the objects expected by ROOT.
            comp_args = [
                mrec if isinstance(arg, str) and arg in obs  else params[arg] if isinstance(arg, str) and arg in params
                         else arg for arg in spec['args']
            ]
        else:
            raise ValueError(f"Component {kind!r} needs an explicit ROOT constructor argument list in its spec under 'args'")

        if 'factory_args' in spec:
            comp_cls = comp_cls(*spec['factory_args'])

        # ROOT constructors consistently receive the object name and title first.
        return comp_cls(name, title, *comp_args)


    def build_params(self, source: dict, mH: float | int) -> dict[str, ROOT.RooAbsArg]:
        params: dict[str, ROOT.RooAbsArg] = {}
        for entry in self.spec['params']:
            if entry['kind'] == 'mH':
                params[entry['name']] = ROOT.RooRealVar(entry['name'], '', mH)
                continue

            if entry['kind'] == 'source':
                raw = self.get_src_val(source, entry['key'], entry.get('idx'))
                if isinstance(raw, (list, tuple)) and len(raw) == 3:
                    params[entry['name']] = ROOT.RooRealVar(entry['name'], '', *raw)
                elif 'lo' in entry and 'hi' in entry:
                    params[entry['name']] = ROOT.RooRealVar(entry['name'], '', raw, entry['lo'], entry['hi'])
                else:
                    params[entry['name']] = ROOT.RooRealVar(entry['name'], '', raw)
                params[entry['name']].setConstant(ROOT.kTRUE)
                continue

            if entry['kind'] == 'variable':
                params[entry['name']] = ROOT.RooRealVar(
                    entry['name'], entry.get('title', entry['name']), entry['value'], *entry['range']
                )
                continue

            if entry['kind'] == 'arg_list':
                params[entry['name']] = self.argl(
                    *(params[arg] for arg in entry['args']), name=entry.get('title')
                )
                continue

            if entry['kind'] == 'formula':
                expr = entry['expr'].format(mH=mH)
                params[entry['name']] = ROOT.RooFormulaVar(
                    entry['name'], '', expr, self.argl(*(params[arg] for arg in entry['args']))
                )
                continue

            raise ValueError(f"Unsupported parameter kind: {entry['kind']}")
        return params


    def build(
        self,
        observable: ROOT.RooRealVar,
        params: dict[str, ROOT.RooAbsArg],
        nevents: float | int,
        suffix: str = '',
        extended: bool = True,
    ) -> RooModel:
        model = RooModel(self.suffixed(self.spec.get('model_name', 'zh_model'), suffix))
        for name, param in params.items():
            model.add(name, param)

        comps = []
        for spec in self.spec['components']:
            comp_name = self.suffixed(spec['name'], suffix)
            pdf = self.get_pdf(spec['kind'], comp_name, spec['title'], observable, params, spec)
            model.add(comp_name, pdf)
            comps.append(pdf)

        if len(comps) == 1 and not self.spec.get('fractions'):
            pdf = next(iter(comps))
        else:
            pdf = ROOT.RooAddPdf(
                self.suffixed(self.spec.get('sum_name', 'sig'), suffix), '',
                self.argl(*comps), self.argl(*(params[n] for n in self.spec['fractions'])),
            )
        norm = ROOT.RooRealVar(self.suffixed(self.spec.get('yield_name', 'norm'), suffix), '', nevents, 0, 1e8)
        model.add(pdf.GetName(), pdf)
        model.add(norm.GetName(), norm)

        if extended:
            fit = ROOT.RooAddPdf(
                self.suffixed(self.spec.get('model_name', 'model'), suffix), '', self.argl(pdf), self.argl(norm)
            )
            model.add(fit.GetName(), fit)
            model.add('fit', fit)
            model.add('pdf', fit)
        else:
            model.add('pdf', pdf)
        return model



class SignalParametrizer:
    """Fit signal models at the mass points required by a model specification."""

    def __init__(self, model_spec: dict, parameter_source: dict):
        self.model_spec = model_spec
        self.parameter_source = parameter_source
        self.builder = RooModelBuilder(model_spec)

    def fit_grid(
        self,
        observable: ROOT.RooRealVar,
        masses: list[float | int],
        processes: list[str],
        histogram_loader,
        normalize: bool = True,
        only_coeffs: bool = False
    ) -> tuple[dict[str, list], list[RooModel], float]:
        if len(masses) != len(processes):
            raise ValueError('Signal masses and processes must have equal length')

        nominal_hist = histogram_loader(processes[0], 'norm', False)
        yield_nom = nominal_hist.Integral()
        var_dict = {key: [] for key in (*self.model_spec['param_names'], 'yield', 'mH')}
        models = []

        for mass, process in zip(masses, processes):
            mass_label = f'{mass:.3f}'.replace('.', 'p')
            hist = histogram_loader(process, mass_label, normalize, yield_nom)
            data = ROOT.RooDataHist(f'rdh_zh_{mass_label}', f'rdh_zh_{mass_label}', ROOT.RooArgList(observable), ROOT.RooFit.Import(hist))
            model = self.builder.build(observable, self.builder.build_params(self.parameter_source, mass), data.sum(False), mass_label)
            model.fit_to(data, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))
            model.add(data.GetName(), data)
            models.append(model)

            params = {name: model.get(name) for name in self.model_spec['param_names'] if name != 'yield'}
            for name in var_dict:
                if name == 'mH':
                    var_dict[name].append(mass)
                elif name == 'yield':
                    var_dict[name].append(model.get(f'{self.model_spec["yield_name"]}_{mass_label}').getVal())
                else:
                    var_dict[name].append(params[name].getVal())
        
        if only_coeffs:
            return var_dict
        return var_dict, models, yield_nom



def get_hist(
        hName: str,
        inputdir: str,
        procs: str | list[str],
        suffix: str,
        cat_idx: tuple[int, int],
        lumiScale: float | int,
        yield_nom: float | int,
        outName: str = '',
        normYields: bool = True):
    if isinstance(procs, str): procs = [procs]

    hist = getHist(hName, procs, inputdir)
    hist.Scale(lumiScale)

    hist = hist.ProjectionX(f'hist_zh_{suffix}', *cat_idx)
    if outName:    hist.SetName(outName)
    if normYields: hist.Scale(yield_nom / hist.Integral())

    return hist


def _with_suffix(base: str, suffix: str) -> str:
    return base if suffix == '' else f'{base}_{suffix}'


def _const_real(
        name: str,
        coeffs: dict[str, list[float | int]],
        lo: float | int | None = None,
        hi: float | int | None = None,
        idx: int = 1) -> ROOT.RooRealVar:
    value = coeffs.get(name, 1)[idx]
    if lo is None or hi is None:
        var = ROOT.RooRealVar(name, '', value)
    else:
        var = ROOT.RooRealVar(name, '', value, lo, hi)
    var.setConstant(ROOT.kTRUE)
    return var


def _arg_list(*items, name: str | None = None):
    arg_list = ROOT.RooArgList()
    if name is not None:
        arg_list.setName(name)
    for item in items:
        arg_list.add(item)
    return arg_list


def _formula_var(name: str, expr: str, *items):
    return ROOT.RooFormulaVar(name, '', expr, _arg_list(*items))


def make_p(mH, mH_label, inDir):

    with open(f'{inDir}/coeff.json') as fIn: coeffs = json.load(fIn)

    mean0     = _const_real('mean0',    coeffs, .5,  1.5)   # slope
    mean1     = _const_real('mean_cb1', coeffs, -1,  1)     # offset
    mean_gt1  = _const_real('mean_gt1', coeffs, -2,  2)
    sigma0    = _const_real('sigma_cb', coeffs, +0,  10)    # 0.4335
    sigma_gt0 = _const_real('sigma_gt', coeffs, +0,  10)
    alpha10   = _const_real('alpha_1',  coeffs, -5,  5)
    alpha20   = _const_real('alpha_2',  coeffs, -5,  5)
    n10       = _const_real('n_1',      coeffs, -50, 50)
    n20       = _const_real('n_2',      coeffs, -50, 50)
    cb10      = _const_real('cb_1',     coeffs, +0,  1)
    cb20      = _const_real('cb_2',     coeffs, +0,  1)

    mean = _formula_var(f'mean_{mH_label}', f'@1 + @0*{mH}', mean0, mean1)
    mean_gt = _formula_var(f'mean_gt_{mH_label}', '@0 + @1', mean, mean_gt1)

    params = {
        'mean':     mean,
        'sigma':    _formula_var(f'sigma_{mH_label}', '@0', sigma0),
        'sigma_gt': _formula_var(f'sigma_gt_{mH_label}', '@0', sigma_gt0),
        'alpha_1':  _formula_var(f'alpha_1_{mH_label}', '@0', alpha10),
        'alpha_2':  _formula_var(f'alpha_2_{mH_label}', '@0', alpha20),
        'n_1':      _formula_var(f'n_1_{mH_label}', '@0', n10),
        'n_2':      _formula_var(f'n_2_{mH_label}', '@0', n20),
        'cb_1':     _formula_var(f'cb_1_{mH_label}', '@0', cb10),
        'cb_2':     _formula_var(f'cb_2_{mH_label}', '@0', cb20),
        'mean0':    mean0,
        'mean1':    mean1,
        'mean_gt1': mean_gt1,
        'mean_gt':  mean_gt,
    }
    return params



def build_2cbg_pdf(
    recoilmass: ROOT.RooRealVar,
    Vars: dict[str, ROOT.RooRealVar],
    yield_zh: float | int,
    suffix: str,
    model_spec: dict | None = None,
     ):
    model_spec = model_spec or {
        'components': (
            {'kind': 'RooCBShape', 'name': 'CrystallBall_1', 'title': 'CrystallBall_1',
             'args': ('mrecoil', 'mean', 'sigma', 'alpha_1', 'n_1')},
            {'kind': 'RooCBShape', 'name': 'CrystallBall_2', 'title': 'CrystallBall_2',
             'args': ('mrecoil', 'mean', 'sigma', 'alpha_2', 'n_2')},
            {'kind': 'RooGaussian', 'name': 'gauss', 'title': 'gauss',
             'args': ('mrecoil', 'mean_gt', 'sigma_gt')},
        ),
        'fractions': ('cb_1', 'cb_2'),
        'yield_name': 'sig_norm',
        'sum_name': 'sig',
        'model_name': 'zh_model',
    }
    return RooModelBuilder(model_spec).build(
        recoilmass, Vars, yield_zh, suffix
    ).get('pdf')


def build_background_pdf(
    recoilmass: ROOT.RooRealVar,
    coeffs: dict[str, ROOT.RooRealVar],
    yield_bkg: float | int,
    suffix: str,
    order: int = 3,
    name: str = 'bkg',
    model_name: str = 'bkg_fit',
    yield_name: str = 'bkg_norm_tmp',
):
    coeff_values = [coeffs[key] for key in sorted(coeffs) if key.startswith('bern')][:order]
    coeff_list = _arg_list(*coeff_values)
    bkg = ROOT.RooBernsteinFast(order)(_with_suffix(name, suffix), _with_suffix(name, suffix), recoilmass, coeff_list)
    bkg_norm = ROOT.RooRealVar(_with_suffix(yield_name, suffix), _with_suffix(yield_name, suffix), yield_bkg, 0, 1e6)
    bkg_fit = ROOT.RooAddPdf(_with_suffix(model_name, suffix), '', ROOT.RooArgList(bkg), ROOT.RooArgList(bkg_norm))
    return bkg, bkg_norm, bkg_fit


def make_unc_import(workspace, spline_vals, val_names, syst, val_up=None, val_dw=None):

    for idx, (spline_val, val_name) in enumerate(zip(spline_vals, val_names)):
        nominal_val = spline_val.getVal()
        if val_up is None or val_dw is None:
            value = 0.0
        else:
            value = 0.0 if nominal_val == 0 else 0.5 * abs(val_up[idx] - val_dw[idx]) / nominal_val

        nominal = ROOT.RooRealVar(f'sig_{val_name}_{syst}', f'sig_{val_name}_{syst}', value)
        nominal.setConstant(ROOT.kTRUE)
        workspace.Import(nominal)


def findCrossing(
        xv: list[float | int],
        yv: list[float | int],
        left: bool = True,
        flip: float | int = 125,
        cross: float | int = 1.):

    closest, idx = 1e9, -1
    for i in range(0, len(xv)):
        if     left and xv[i] > flip: continue
        if not left and xv[i] < flip: continue
        dy = abs(yv[i] - cross)
        if dy < closest:
            closest = dy
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


def _graph_from_points(
        xv: list[float | int],
        yv: list[float | int]) -> ROOT.TGraph:
    return ROOT.TGraph(len(xv), array.array('d', xv), array.array('d', yv))


def _load_nll_curve(
        file_path: PathObj) -> tuple[float, float, list[float], list[float], ROOT.TGraph]:
    xv, yv = [], []
    with open(file_path, 'r') as fIn:
        for i, line in enumerate(fIn.readlines()):
            line = line.rstrip()
            if i == 0:
                best_mass = float(line.split(' ')[3])
                unc_mass = float(line.split(' ')[2])
            else:
                xv.append(float(line.split(' ')[0]))
                yv.append(float(line.split(' ')[1]))

    return best_mass, unc_mass, xv, yv, _graph_from_points(xv, yv)


def _load_fit_summary(
        file_path: PathObj,
        scale: float = 1.0) -> tuple[float, float]:
    with open(file_path, 'r') as fIn:
        first_line = fIn.readline().rstrip()
    best = float(first_line.split(' ')[3])
    unc = float(first_line.split(' ')[2]) * scale
    return best, unc


def _write_fit_curve(
        out_path: PathObj,
        unc_m: float,
        unc_p: float,
        unc: float,
        mass: float,
        xv: list[float],
        yv: list[float]) -> None:
    lines = [f'{unc_m} {unc_p} {unc} {mass}\n']
    lines.extend(f'{x} {y}\n' for x, y in zip(xv, yv))
    with open(out_path, 'w') as fOut:
        fOut.writelines(lines)


def _run_combine(
        cmd: list[str],
        runDir: str) -> None:
    subprocess.call(cmd, cwd=runDir)


def analyzeMass(
        runDir: PathObj,
        outDir: PathObj,
        xMin: float | int = -1,
        xMax: float | int = -1,
        yMin: float | int = 0,
        yMax: float | int = 2,
        label: str = 'label',
        top_right: str = '',
        suffix: str = ''):

    outDir.mkdir(exist_ok=True, parents=True)

    fIn = ROOT.TFile(runDir / 'higgsCombinemass.MultiDimFit.mH125.root', 'READ')
    t = fIn.Get('limit')

    xv, yv = [], []
    for i in range(0, t.GetEntries()):

        t.GetEntry(i)

        if t.quantileExpected < -1.5: continue
        if t.deltaNLL > 20: continue
        xv.append(t.MH)
        yv.append(t.deltaNLL*2.)

    xv, yv = zip(*sorted(zip(xv, yv)))
    g = _graph_from_points(list(xv), list(yv))

    # bestfit = minimum
    mass = 1e9
    for i in range(g.GetN()):
        if g.GetY()[i] == 0: mass = g.GetX()[i]

    # extract uncertainties at crossing = 1
    unc_m = findCrossing(xv, yv, True,  mass)
    unc_p = findCrossing(xv, yv, False, mass)
    unc = (abs(mass - unc_m) + abs(unc_p - mass)) / 2

    plot_mass_scan(
        outDir,
        g,
        label,
        unc * 1000.0,
        min(xv) if xMin < 0 else xMin,
        max(xv) if xMax < 0 else xMax,
        yMin,
        yMax,
        '#bf{FCC-ee} #scale[0.7]{#it{Internal}}',
        top_right,
        output_name=f'mass{suffix}',
        graph_color=ROOT.kRed,
        graph_width=2,
    )

    # Write values to text file
    _write_fit_curve(outDir / f'mass{suffix}.txt', unc_m, unc_p, unc, mass, list(xv), list(yv))
    _write_fit_curve(runDir / f'mass{suffix}.txt', unc_m, unc_p, unc, mass, list(xv), list(yv))


def doFit_mass(
        runDir: str,
        mhMin: float | int = 124.99,
        mhMax: float | int = 125.01,
        npoints: int = 50,
        combineOptions: list[str] = []):

    cmd = ['combine', '-M', 'MultiDimFit', 'ws.root', '-t', '-1', '-v', '2', '-n', 'mass',
           '--setParameterRanges', f'MH={mhMin},{mhMax}', '--algo', 'grid', '--points', f'{npoints}',
           '--expectSignal=1', '-m', '125', '--redefinesSignalPOIs', 'MH',
           '--X-rtd', 'TMCSO_AdaptivePseudoAsimov', '--X-rtd', 'ADDNLL_CBNLL=0'] + combineOptions
    _run_combine(cmd, runDir)


def doFitDiagnostics_mass(
        runDir: str,
        mhMin: float | int = 124.99,
        mhMax: float | int = 125.01,
        combineOptions: list[str] = []):

    cmd = ['combine', '-M', 'MultiDimFit', 'ws.root', '-t', '-1', '-v', '2', '-m', '125', '-n', 'mass',
           '--setParameterRanges', f'MH={mhMin},{mhMax}', '--expectSignal=1',
           '--algo', 'singles', '--redefineSignalPOIs', 'MH', '--floatParameters', 'MH',
           '--X-rtd', 'TMCSO_AdaptivePseudoAsimov', '--X-rtd', 'ADD_CBNLL=0'] + combineOptions

    _run_combine(cmd, runDir)

    # Get the uncertainty
    ### Will optimize it later (use code from xsec extraction)
    with ROOT.TFile(f'{runDir}/higgsCombinemass.MultiDimFit.mH125.root') as fIn:
        tt = fIn.Get('limit')
        vals = []
        for i in range(tt.GetEntries()):
            tt.GetEntry(i)
            vals.append(float(tt.MH))

        vals = sorted(vals)
        lo, best, hi = vals[0], vals[1], vals[2]

        err_down, err_up = best - lo, hi - best
        err_avg = (err_up + err_down) / 2
    return err_avg

def breakDown(
        outDir: PathObj,
        top_right: str = '',
        mass_systematics: list[str] | None = None):

    if mass_systematics is None:
        mass_systematics = ['BES', 'SQRTS', 'LEPSCALE_MU', 'LEPSCALE_EL']

    def getUnc(tag, type):
        scale = 1000 if type == 'mass' else 100 if type == 'xsec' else 1
        best, unc = _load_fit_summary(outDir / f'{type}{tag}.txt', scale)
        return best, unc

    _, unc_ref = getUnc('_stat', 'mass')
    params = [f'_{name}' for name in mass_systematics] + ['']
    labels = ['BES', '#sqrt{s} #pm 2 MeV', 'Muon scale (~10^{-5})', 'El. scale (~10^{-5})', 'Syst. combined']

    impacts_mev = []
    for p in params:
        _, unc = getUnc(p, 'mass')
        impacts_mev.append(math.sqrt(unc**2 - unc_ref**2))

    plot_mass_breakdown_impacts(
        outDir,
        impacts_mev,
        labels,
        top_right,
        '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    )

    params = ['_stat', '_BES', '_LEPSCALE', '_SQRTS', '']
    labels = ['Stat. only', 'Beam energy spread', 'Lepton scale', 'Center-of-mass energy', 'Stat. + syst. combined']

    tags = [f'{outDir}/mass{p}.txt' for p in params]
    curves = []
    style_spec = [
        (ROOT.kRed + 1,    4, 1),
        (ROOT.kBlue - 4,   2, 2),
        (ROOT.kOrange - 3, 2, 2),
        (ROOT.kGreen + 3,  2, 2),
        (ROOT.kBlack,      4, 1),
    ]
    for index, tag in enumerate(tags):
        _, unc, x_values, y_values, _ = _load_nll_curve(tag)
        curves.append((x_values, y_values, unc * 1000.0, *style_spec[index]))

    plot_mass_breakdown_curves(
        outDir,
        curves,
        labels,
        top_right,
        '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    )


def text2workspace(runDir):
    cmd = ['text2workspace.py', 'datacard.txt', '-o', 'ws.root', '-v', '10', '--X-allow-no-background']
    _run_combine(cmd, runDir)


def combineCards(
        runDir: PathObj,
        cards: list[str] = []):

    runDir.mkdir(exist_ok=True, parents=True)
    cmd = ['combineCards.py', '--force-shape'] + cards

    with open(runDir / 'datacard.txt', 'w') as out:
        subprocess.call(cmd, cwd=runDir, stdout=out)


def run_mass_pipeline(
        runDir: PathObj,
        outDir: PathObj,
        label: str,
        combineOptions: list[str],
        fitRange: tuple[float, float] = (124.95, 125.05),
        top_right: str = '',
        suffix: str = '') -> None:
    mh_err = doFitDiagnostics_mass(runDir, fitRange[0], fitRange[1], combineOptions)
    mhMin, mhMax = mHrange(mh_err)
    doFit_mass(runDir, mhMin, mhMax, 50, combineOptions)
    analyzeMass(runDir, outDir, mhMin, mhMax, label=label, top_right=top_right, suffix=suffix)



def mHrange(mh_err: float | int) -> tuple[float | int, float | int]:
    if 1.5 * mh_err > 0.05:  # bound to 50 MeV
        return 124.95, 125.05
    return 125 - 1.5 * mh_err, 125 + 1.5 * mh_err
