################################
### STANDARD LIBRARY IMPORTS ###
################################

from typing import Any
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np



#########################################
### LOGGING AND CUSTOM MODULE IMPORTS ###
#########################################

from logger import get_logger
LOGGER = get_logger(__name__)



##########################
### CUTFLOW PLOT CLASS ###
##########################

class CutFlowPlot:
    '''Plot cutflow histograms with a stack, total background, and signal.'''

    def __init__(
        self,
        flow: dict[str, dict[str, Any]] | None,
        outputdir: str,
        selection: str,
        plots: dict[str, dict[str, list[str]]],
        colors: dict[str, int],
        legend: dict[str, str],
        ecm: int = 240,
        lumi: float = 10.8,
        tot: bool = False,
        output_subdir: str = 'yield',
    ) -> None:
        if not plots or not plots.get('signals'):
            raise ValueError('CutFlowPlot requires at least one signal process')

        self.outputdir = outputdir
        self.output_subdir = output_subdir
        self.selection = selection
        self.plots  = plots
        self.colors = colors
        self.legend = legend
        self.ecm  = ecm
        self.lumi = lumi
        self.tot  = tot

        self.sigs  = list(plots['signals'])
        self.bkgs  = list(plots.get('backgrounds', {}))
        self.procs = [*self.sigs, *self.bkgs]
        self.flow  = flow or {}

        return None

    def style_hist(
        self,
        hist: Any,
        color: int,
        width: int = 1,
        style: int = 1,
        scale: float = 1.,
        fill_color: int | None = None,
    ) -> None:
        hist.SetLineColor(color)
        hist.SetLineWidth(width)
        hist.SetLineStyle(style)
        if fill_color is not None:
            hist.SetFillColor(fill_color)
        if scale != 1.:
            hist.Scale(scale)

        return None

    def get_histograms(self) -> dict[str, Any]:
        '''Build and cache process histograms from the cutflow values.'''
        if hasattr(self, '_source_hists'):
            return self._source_hists

        import ROOT

        self._source_hists = {}
        for proc in self.procs:
            values = self.flow.get(proc)
            if values is None: continue

            counts: dict = values.get('cut', {})
            errors: dict = values.get('err', {})
            hist = ROOT.TH1D(f'hists_{proc}', f'hists_{proc}', len(counts), 0, len(counts))
            hist.SetDirectory(0)
            for index, cut in enumerate(counts):
                hist.SetBinContent(index + 1, counts[cut])
                hist.SetBinError(index + 1, errors.get(cut, 0.))
            self._source_hists[proc] = hist

        return self._source_hists


    def prepare(self, sig_scale: float = 1., rescale: bool = False) -> None:
        '''Style flow histograms for stack or independent-curve mode.'''
        import copy, ROOT

        hists   = self.get_histograms()
        missing = [proc for proc in self.procs if proc not in hists]
        if missing:
            raise KeyError(f'Missing cutflow histograms: {missing}')
        self.hists = hists
        self.nsig  = len(self.sigs)
        self.yields = [copy.deepcopy(hists[proc]) for proc in self.procs]

        if rescale and self.bkgs:
            raise ValueError('Rescaled curves require signal-only histograms')
        for proc in self.sigs:
            scale = 1.
            if rescale:
                first_bin = hists[proc].GetBinContent(1)
                scale = 100. / first_bin if first_bin else 1.
            self.style_hist(hists[proc], self.colors[proc],
                            4 if self.bkgs else 2, 1,
                            sig_scale if self.bkgs else scale)

        self.signals = [hists[proc] for proc in self.sigs]
        self.sig_scale = 1. if rescale else sig_scale
        self.table_yields = self.signals if rescale else self.yields

        if not self.bkgs: return None
        self.stack = ROOT.THStack('stack', 'stack')
        self.bkgs_hists, self.bkg_tot = [], None

        for proc in self.bkgs:
            hist = hists[proc]
            if self.bkg_tot is None:
                self.bkg_tot = hist.Clone('h_cuflow_bkg_tot')
            else:
                self.bkg_tot.Add(hist)

            self.style_hist(hist, ROOT.kBlack, 1, 1,
                            1, self.colors[proc])
            self.stack.Add(hist)
            self.bkgs_hists.append(hist)

        if self.bkg_tot is None:
            raise ValueError('CutFlowPlot requires at least one background process')
        self.style_hist(self.bkg_tot, ROOT.kBlack, 2)

        return None

    def define_legend(
            self,
            sig_scale: float = 1.,
            num_entries: int = 1,
            columns: int | None = None,
            x1: float | None = None,
            y1: float | None = None,
            x2: float | None = None,
            y2: float | None = None,
            border_size: int = 0,
            fill_style: int = 0,
            text_size: float = 0.03,
            set_margin: float = 0.2,
            text_font: int = -1,
            return_leg: bool = False
    ) -> Any | None:
        '''Create and populate the cutflow legend.'''
        import ROOT

        columns = columns or 1 if self.bkgs else 4
        x1 = x1 or 0.55 if self.bkgs else 0.2
        y1 = y1 or 0.99 if self.bkgs else 0.925
        x2 = x2 or 0.99 if self.bkgs else 0.95
        y2 = y2 or 0.90 if self.bkgs else 0.925

        self.leg = ROOT.TLegend(x1, y1 - num_entries * 0.06 / columns, x2, y2)

        if text_font != -1:
            self.leg.SetTextFont(text_font)
        self.leg.SetBorderSize(border_size)
        self.leg.SetFillStyle(fill_style)
        self.leg.SetTextSize(text_size)
        self.leg.SetMargin(set_margin)
        self.leg.SetNColumns(columns)

        for proc in self.procs:
            label = self.legend[proc]
            if proc in self.sigs and self.bkgs and sig_scale != 1:
                label += f' (#times {int(sig_scale)})'
            self.leg.AddEntry(self.hists[proc], label, 'L' if proc in self.sigs else 'F')

        if return_leg:
            return self.leg
        return None


    def ordered_labels(self, labels: dict[str, str] | list[str]) -> list[str]:
        return ([labels[key] for key in sorted(labels)]
                if isinstance(labels, dict) else labels)

    def get_significance(self, yields: list[Any], sig_scale: float) -> 'np.ndarray':
        import numpy as np

        contents = np.asarray([[float(h.GetBinContent(i + 1)) for i in range(yields[0].GetNbinsX())] for h in yields])
        sig = contents[:self.nsig].sum(axis=0)
        sig = sig / sig_scale if sig_scale else sig
        bkg = contents[self.nsig:].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return list(np.where(sig + bkg == 0, -1., sig / np.sqrt(sig + bkg)))

    def build_config(
        self,
        xmin: float,
        xmax: float,
        ymin: float | None,
        ymax: float | None,
        logX: bool = False,
        logY: bool = True,
        xtitle: str | None = 'None',
        ytitle: str = 'Events',
    ) -> dict[str, Any]:
        '''Build the log-scale configuration used by a cutflow frame.'''
        from .plotter import make_cfg

        return make_cfg({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
                         'logx': logX, 'logy': logY, 'xtitle': xtitle, 'ytitle': ytitle},
                         self.ecm, self.lumi)


    def draw_frame(self, labels: list[str], canvas_size: tuple[int, int] = (800, 800)) -> Any:
        from . import plotter
        plotter.cfg = self.cfg
        canvas = plotter.canvas(*canvas_size, bottom=0.12)
        dummy = plotter.dummy(len(labels), labels,
                              28, 1.1, 1.5)
        self._frame_hist = dummy
        dummy.GetXaxis().LabelsOption('u')
        dummy.Draw('HIST')
        return canvas

    def draw(
        self,
        labels: dict[str, str] | list[str],
        ymin: float | None,
        ymax: float | None,
        outName: str,
        file_formats: list[str],
        suffix: str = '',
        quiet: bool = False,
        stats: tuple[float, float, float, float] | None = None,
    ) -> tuple[list[float], list[list[Any]]]:
        from .plotter import draw_latex, setup_latex
        labels = self.ordered_labels(labels)

        self.cfg = self.build_config(0, len(labels), ymin, ymax)
        if not self.bkgs:
            self.cfg.update(logy=False, ytitle='Selection Efficiency [%]')

        canvas = self.draw_frame(labels)
        if self.bkgs:
            self.stack.Draw('SAME HIST')
            self.bkg_tot.Draw('SAME HIST')
        for hist in self.signals:
            hist.Draw('SAME HIST')

        Z    = self.get_significance(self.yields, self.sig_scale) if self.bkgs else None
        rows = self.yield_rows(self.table_yields, Z)
        
        if stats is not None:
            avg, avg_err, Min, Max = stats
            text = setup_latex(0.04, 11)
            draw_latex(text, [(f'Avg eff: {avg:.2f} #pm {avg_err:.2f} %', 0.2, 0.25, 0.04),
                                (f'Min/Max: {Min:.2f}/{Max:.2f}', 0.2, 0.2, 0.04)])
            text.Draw('SAME')

        if hasattr(self, 'leg'): self.leg.Draw('SAME')
        self.save(canvas, outName, suffix, file_formats, quiet)
        canvas.Close()
        return rows


    def yield_rows(self, yields: list[Any], tab: list | None) -> list[list[str]]:
        nbins, rows = next(iter(yields)).GetNbinsX(), []
        for i in range(1, nbins + 1):
            row = [f'Cut {i-1}']
            if tab is not None: row.append(f'{tab[i-1]:.3f}')
            row.extend('%.2e +/- %.2e' % (h.GetBinContent(i), h.GetBinError(i)) for h in yields)
            rows.append(row)
        return rows

    def write_table(
        self,
        file_name: str,
        headers: list[str],
        rows: list[list[str]],
        first_col_width: int = 10,
        other_col_width: int = 25,
        file_type: str = 'txt',
    ) -> None:
        '''Write a formatted yield table in the plot output directory.'''
        out = self.output_dir()
        out.mkdir(exist_ok=True, parents=True)
        widths = [first_col_width] + [other_col_width] * (len(headers) - 1)
        fmt = '{:<%d} ' % widths[0] + ' '.join('{:<%d}' % width for width in widths[1:])
        with open(out / f'{file_name}.{file_type}', 'w') as table:
            table.write(fmt.format(*headers) + '\n')
            table.write(fmt.format(*('-' * width for width in widths)) + '\n')
            for row in rows:
                table.write(fmt.format(*(str(value) for value in row)) + '\n')

    def output_dir(self) -> Path:
        '''Return the standard output directory for this selection.'''
        base_sel = self.selection.replace('_high', '').replace('_low', '')
        direction = ('high' if '_high' in self.selection else 'low' if '_low' in self.selection else 'nominal')
        return Path(self.outputdir) / self.output_subdir / base_sel / direction

    def save(
        self,
        canvas: Any,
        outName: str,
        suffix: str,
        file_formats: list[str],
        quiet: bool,
    ) -> None:
        from .plotter import finalize_canvas, save_canvas

        out = self.output_dir()
        out.mkdir(exist_ok=True, parents=True)

        suffix_tot = ('_tot' if self.tot else '') + suffix

        finalize_canvas(canvas)
        save_canvas(canvas, out, outName, suffix_tot, file_formats, quiet)