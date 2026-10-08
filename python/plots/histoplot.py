'''Reusable ROOT plot objects.

This module contains the small amount of orchestration shared by the ROOT
histogram plots. The lower-level ROOT details remain in ``plots.root`` so
that specialized plot classes can override individual steps later.
'''

from typing import Any
from pathlib import Path

from logger import get_logger

LOGGER = get_logger(__name__)


class HistogramPlot:
    '''Plot one signal process and a collection of background processes.'''

    def __init__(
        self,
        variable: str,
        sel: str,
        inDir: str,
        outDir: str,
        plots: dict[str, dict[str, list[str]]],
        colors: dict[str, Any],
        legend: dict[str, str],
        ecm: int = 240,
        lumi: float = 10.8,
        tot: bool = False
    ) -> None:

        self.variable = variable
        self.inDir = inDir
        self.outDir = outDir
        self.sel = sel
        self.plots = plots
        self.colors = colors
        self.legend = legend
        self.ecm = ecm
        self.lumi = lumi
        self.tot = tot

        if not plots:
            raise ValueError(
                'HistogramPlot requires a non-empty plots mapping')
        self.sig_processes = plots.get('signals',     {})
        self.bkg_processes = plots.get('backgrounds', {})
        self.signals = list(self.sig_processes)
        self.backgrounds = list(self.bkg_processes)
        self.processes = [*self.signals, *self.backgrounds]

        return None

    def load_histograms(
        self,
        suffix: str,
        rebin: int = 1,
        lazy: bool = True,
        normalize: bool = False
    ) -> dict[str, Any]:
        '''Load one histogram for each configured process.'''

        from tools.process import getHist

        process_map = dict(self.bkg_processes)
        process_map.update(self.sig_processes)

        raw_hists = {proc: getHist(self.variable, proc_list,
                                   self.inDir, suffix, rebin, lazy)
                     for proc, proc_list in process_map.items()}

        if normalize:
            for k, h in raw_hists.items():
                integral = h.Integral()
                norm = 1.0 / integral if integral > 0 else 1
                raw_hists[k].Scale(norm)
        return raw_hists

    def define_legend(
            self,
            num_entries: int,
            columns: int = 1,
            x1: float = 0.55,
            y1: float = 0.99,
            x2: float = 0.99,
            y2: float = 0.90,
            border_size: int = 0,
            fill_style: int = 0,
            text_size: float = 0.03,
            set_margin: float = 0.2,
            text_font: int = -1
    ) -> Any:
        '''Create the legend used by the plot.'''

        import ROOT
        leg = ROOT.TLegend(x1, y1 - num_entries * 0.06 / columns, x2, y2)

        if text_font != -1:
            leg.SetTextFont(text_font)
        leg.SetBorderSize(border_size)
        leg.SetFillStyle(fill_style)
        leg.SetTextSize(text_size)
        leg.SetMargin(set_margin)
        leg.SetNColumns(columns)

        return leg

    def style_hist(
            self,
            hist,
            color: int,
            width: int = 1,
            style: int = 1,
            scale: float = 1.,
            fill_color: int | None = None
    ) -> None:

        hist.SetLineColor(color)
        hist.SetLineWidth(width)
        hist.SetLineStyle(style)
        if fill_color is not None:
            hist.SetFillColor(fill_color)
        if scale != 1.:
            hist.Scale(scale)

        return None

    def style_histograms(
        self,
        histograms: dict[str, Any],
        legend_obj: Any,
        sig_scale: float = 1.,
        bkg_scale: float = 1.
    ) -> tuple[Any, list[Any]]:
        '''Style histograms and return the background stack contents.'''
        import ROOT

        stack = ROOT.THStack('stack', 'stack')
        signals, backgrounds = [], []
        for process in self.processes:
            hist = histograms.get(process)
            if hist is None:
                continue

            is_signal = process in self.signals
            self.style_hist(hist,
                            self.colors[process] if is_signal else ROOT.kBlack,
                            3 if is_signal else 1, 1,
                            sig_scale if is_signal else bkg_scale,
                            self.colors[process] if not is_signal else None)
            label = self.legend[process]
            if is_signal and sig_scale != 1:
                label += f' (#times {int(sig_scale)})'
            if not is_signal and bkg_scale != 1:
                label += f' (#times {int(bkg_scale)})'
            legend_obj.AddEntry(hist, label, 'L' if is_signal else 'F')

            if is_signal:
                signals.append(hist)
            else:
                stack.Add(hist)
                backgrounds.append(hist)

        missing_signals = [signal for signal in self.signals
                           if histograms.get(signal) is None]
        if missing_signals:
            LOGGER.warning(
                f'Could not load signal histograms: {missing_signals}')
        return stack, signals, backgrounds

    def build_config(
        self,
        signal_hists: list[Any],
        backgrounds: list[Any],
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        logX: bool = False,
        logY: bool = False,
        xtitle: str | None = '',
        ytitle: str = 'Events',
        scale_min: float | None = None,
        scale_max: float | None = None,
        strict: bool = True,
        stack: bool = False,
    ) -> dict[str, Any]:
        '''Build the ROOT plot configuration for the loaded histograms.'''
        from .root.helper import make_cfg

        ref_hist = signal_hists[0] if signal_hists else backgrounds[0]
        all_hists = [*signal_hists, *backgrounds]
        xMin, xMax, yMin, yMax = self._get_ranges(all_hists, backgrounds,
                                                  xmin, xmax, ymin, ymax,
                                                  scale_min, scale_max,
                                                  logY, strict, stack)

        if xtitle in ('', None):
            xTitle = ref_hist.GetXaxis().GetTitle() if not xtitle else ''
        else:
            xTitle = xtitle

        bwidth = ref_hist.GetBinWidth(1)
        if 'MeV' in xTitle:
            unit = 'MeV'
        elif 'GeV' in xTitle:
            unit = 'GeV'
        elif 'TeV' in xTitle:
            unit = 'TeV'
        else:
            unit = ''

        if bwidth.is_integer():
            ytitle += f' / {bwidth} {unit}'
        else:
            ytitle += f' / {bwidth:.2f} {unit}'

        return make_cfg({
            'xmin': xMin,     'xmax': xMax,
            'ymin': yMin,     'ymax': yMax,
            'logx': logX,     'logy': logY,
            'xtitle': xTitle, 'ytitle': ytitle,
        }, self.ecm, self.lumi)

    def _get_ranges(
        self,
        histograms: list[Any],
        backgrounds: list[Any],
        xmin: float | int | None = None,
        xmax: float | int | None = None,
        ymin: float | int | None = None,
        ymax: float | int | None = None,
        min_scale: float | None = None,
        max_scale: float | None = None,
        logY: bool = False,
        strict: bool = True,
        stack: bool = False,
    ) -> tuple[float, float, float, float]:
        '''Get common axis limits for signals and backgrounds.'''
        from tools.process import get_xrange, get_yrange, get_stack

        if not histograms:
            raise ValueError('At least one histogram is required for ranges')

        total = get_stack(histograms)
        xMin, xMax = get_xrange(total, strict, xmin, xmax)

        scale_min = min_scale if min_scale is not None else (
            0.5 if logY else 1.0)
        scale_max = max_scale if max_scale is not None else (
            1e4 if logY else 1.5)

        y_ranges = [get_yrange(hist, logY, ymin, ymax,
                               scale_min, scale_max) for hist in histograms]
        yMin = min(axis_range[0] for axis_range in y_ranges)

        if stack:
            stacked_range = get_yrange(
                total, logY, ymin, ymax, scale_min, scale_max)
            yMax = stacked_range[1]
        else:
            y_max_hists = list(histograms[:len(histograms) - len(backgrounds)])
            if backgrounds:
                y_max_hists.append(get_stack(backgrounds))
            yMax = max(get_yrange(hist, logY, ymin, ymax, scale_min,
                       scale_max)[1] for hist in y_max_hists)

        return xMin, xMax, yMin, yMax

    def draw(
        self,
        histograms: dict[str, Any],
        stack: Any | None,
        backgrounds: list[Any],
        legend_obj: Any,
        stack_signals: bool = False,
        xlabels: list[str] = []
    ) -> tuple[Any, Any]:
        '''Draw the configured histograms on a standard ROOT canvas.'''

        from .root import plotter

        plotter.cfg = self.cfg
        canvas = plotter.canvas()
        dummy = plotter.dummy(1 if not xlabels else len(xlabels),
                              xlabels,
                              0.75 if len(xlabels) > 0 else 1,
                              1.3 if len(xlabels) > 0 else 1)

        dummy.Draw('HIST')
        if stack_signals:
            if stack is None:
                LOGGER.warning('No stack was provided while stack = True. '
                               'Just plotting the signal')
                for signal in self.signals:
                    histograms[signal].Draw('HIST SAME')
            else:
                for signal in self.signals:
                    stack.Add(histograms[signal])
                stack.Draw('HIST SAME')
        else:
            if backgrounds:
                stack.Draw('HIST SAME')
            for signal in self.signals:
                histograms[signal].Draw('HIST SAME')
        legend_obj.Draw('SAME')
        return canvas, dummy

    def save(
        self,
        canvas: Any,
        outName: str,
        suffix: str,
        file_formats: list[str],
        logY: bool,
        quiet: bool,
    ) -> None:
        '''Finalize and save the canvas using the standard output layout.'''

        from .root.plotter import finalize_canvas, save_canvas

        base_sel = self.sel.replace('_high', '').replace('_low', '')
        direction = ('high' if '_high' in self.sel
                     else 'low' if '_low' in self.sel
                     else 'nominal')
        category = 'tot' if self.tot else 'cat'
        out = Path(f'{self.outDir}/{base_sel}/{direction}/{category}')
        out.mkdir(exist_ok=True, parents=True)

        finalize_canvas(canvas)
        save_canvas(canvas, out, outName,
                    ('_log' if logY else '_lin') + suffix,
                    file_formats, quiet)


class CutFlowPlot(HistogramPlot):
    '''Plot cutflow histograms with a stack, total background, and signal.'''

    def __init__(
        self,
        flow: dict[str, dict[str, Any]],
        out_dir: str,
        category: str,
        selection: str,
        plots: dict[str, dict[str, list[str]]],
        colors: dict[str, int],
        labels: dict[str, str],
        ecm: int = 240,
        lumi: float = 10.8,
        tot: bool = False,
    ) -> None:
        if not plots or not plots.get('signals'):
            raise ValueError('CutFlowPlot requires at least one signal process')
        super().__init__('cutflow', selection,
                         '', out_dir, plots,
                         colors, labels, ecm, lumi, tot)
        self.flow = flow
        self.category  = category
        self.processes = [*self.signals, *self.backgrounds]


    def prepare(self, signal_scale: float = 1.) -> dict[str, Any]:
        '''Style flow histograms for stack or independent-curve mode.'''
        import copy
        import ROOT

        histograms = {process: self.flow[process]['hist'][0]
                      for process in self.processes}
        self._histograms = histograms
        yield_hists = [copy.deepcopy(histograms[process])
                       if process in self.signals else histograms[process]
                       for process in self.processes]

        for process in self.signals:
            self.style_hist(histograms[process], self.colors[process],
                            4 if self.backgrounds else 2, 1,
                            signal_scale if self.backgrounds else 1.)

        if not self.backgrounds:
            return {'mode': 'curves',
                    'histograms': [histograms[process] for process in self.signals],
                    'yield_hists': yield_hists, 'signal_scale': signal_scale}

        if len(self.signals) != 1:
            raise ValueError('Stack mode requires exactly one signal process')
        signal = histograms[self.signals[0]]
        stack = ROOT.THStack('stack', 'stack')
        backgrounds = []
        background_total = None
        for process in self.backgrounds:
            histogram = histograms[process]
            if background_total is None:
                background_total = histogram.Clone('h_bkg_tot')
            else:
                background_total.Add(histogram)

            self.style_hist(histogram, ROOT.kBlack, 1, 1,
                            1, self.colors[process])
            stack.Add(histogram)
            backgrounds.append(histogram)

        if background_total is None:
            raise ValueError('CutFlowPlot requires at least one background process')
        background_total.SetLineColor(ROOT.kBlack)
        background_total.SetLineWidth(2)

        return {
            'mode': 'stack',
            'signal':       signal,
            'signal_scale': signal_scale,
            'stack':        stack,
            'backgrounds':      backgrounds,
            'background_total': background_total,
            'yield_hists':      yield_hists,
        }

    def define_legend(self, signal_scale: float = 1.) -> Any:
        '''Create and populate the cutflow legend.'''
        columns = 1 if self.backgrounds else 4
        legend = super().define_legend(
            len(self.processes), columns,
            0.55 if self.backgrounds else 0.2,
            0.99 if self.backgrounds else 0.925,
            0.99 if self.backgrounds else 0.95,
            0.90 if self.backgrounds else 0.925
        )
        for process in self.processes:
            label = self.legend[process]
            if process in self.signals and self.backgrounds and signal_scale != 1:
                label += f' (#times {int(signal_scale)})'
            legend.AddEntry(self._histograms[process], label,
                            'L' if process in self.signals else 'F')
        return legend

    def build_cutflow_config(
        self,
        prepared: dict[str, Any],
        xmin: float,
        xmax: float,
        ymin: float | None,
        ymax: float | None,
    ) -> dict[str, Any]:
        '''Build the log-scale configuration used by a cutflow frame.'''
        from .root.helper import make_cfg
        from tools.process import get_range

        x_min, x_max, y_min, y_max = get_range(
            [prepared['signal']], prepared['backgrounds'],
            True, False, False,
            0.5, 1e4,
            xmin, xmax, ymin, ymax)

        return make_cfg({
            'xmin': x_min, 'xmax': x_max,
            'ymin': y_min, 'ymax': y_max,
            'logx': False, 'logy': True,
            'xtitle': 'None', 'ytitle': 'Events',
        }, self.ecm, self.lumi
        )


    def draw(
        self,
        prepared: dict[str, Any],
        legend: Any,
        labels: dict[str, str] | list[str],
        y_min: float | None,
        y_max: float | None,
        out_name: str,
        file_formats: list[str],
        suffix: str = '',
        quiet: bool = False,
        curve_stats: tuple[float, float, float, float] | None = None,
    ) -> tuple[list[float], list[list[Any]]]:
        '''Draw either a stacked cutflow or independent curves.'''
        import numpy as np
        from .root import plotter

        yield_hists = prepared['yield_hists']
        nbins = yield_hists[0].GetNbinsX()
        ordered_labels = ([labels[key] for key in sorted(labels)]
                          if isinstance(labels, dict) else labels)

        if prepared['mode'] == 'curves':
            from .root.helper import draw_latex, make_cfg, setup_latex

            self.cfg = make_cfg({
                'xmin': 0, 'xmax': nbins,
                'ymin': y_min, 'ymax': y_max,
                'logx': False, 'logy': False,
                'xtitle': 'None',
                'ytitle': 'Selection efficiency [%]',
            }, self.ecm, self.lumi)
            plotter.cfg = self.cfg
            canvas = plotter.canvas(800, 800)
            dummy = plotter.dummy(nbins, ordered_labels, 0.75, 1.3)
            dummy.Draw('HIST')
            if curve_stats is not None:
                average, average_error, spread_min, spread_max = curve_stats
                text = setup_latex(0.04, 11, text_color=1, text_font=42)
                draw_latex(text, [
                    (f'Avg eff: {average:.2f} #pm {average_error:.2f} %', 0.2, 0.2, 0.04),
                    (f'Min/max: {spread_min:.2f}/{spread_max:.2f}', 0.2, 0.15, 0.04),
                ])
                text.Draw('SAME')
            for histogram in prepared['histograms']:
                histogram.Draw('SAME HIST')
            legend.Draw('SAME')
            self.save(canvas, out_name, suffix, file_formats, False, quiet)
            canvas.Close()
            return [], []

        contents = np.vstack([
            np.fromiter((float(hist.GetBinContent(i + 1))
                         for i in range(nbins)), dtype=float)
            for hist in yield_hists
        ])
        signal = contents[0]
        signal_scale = prepared['signal_scale']
        if signal_scale != 1.:
            signal = signal / signal_scale
        background = contents[1:].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            significance = np.where(signal + background == 0,
                                    -1., signal / np.sqrt(signal + background))

        self.cfg = self.build_cutflow_config(prepared, 0, nbins, y_min, y_max)
        plotter.cfg = self.cfg
        canvas = plotter.canvas()
        dummy  = plotter.dummy(nbins, ordered_labels, 0.75, 1.3)
        dummy.Draw('HIST')
        prepared['stack'].Draw('SAME HIST')
        prepared['background_total'].Draw('SAME HIST')
        prepared['signal'].Draw('SAME HIST')
        legend.Draw('SAME')

        self.save(canvas, out_name, suffix, file_formats, True, quiet)
        canvas.Close()

        rows = []
        for index in range(nbins):
            row = [f'Cut {index}', f'{significance[index]:.3f}']
            row.extend('%.2e +/- %.2e' % (hist.GetBinContent(index + 1),
                            hist.GetBinError(index + 1)) for hist in yield_hists)
            rows.append(row)
        return significance.tolist(), rows

    @property
    def sel_base(self) -> str:
        return self.sel.replace('_high', '').replace('_low', '')

    @property
    def direction(self) -> str:
        return ('high' if '_high' in self.sel
                else 'low' if '_low' in self.sel
                else 'nominal')

    def output_dir(self) -> Path:
        '''Return the HistogramPlot-style output directory for this selection.'''
        category = 'tot' if self.tot else 'cat'
        return Path(self.outDir) / self.sel_base / self.direction / category
