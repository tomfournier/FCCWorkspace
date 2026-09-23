'''Reusable ROOT plot objects.

This module contains the small amount of orchestration shared by the ROOT
histogram plots. The lower-level ROOT details remain in ``plots.root`` so
that specialized plot classes can override individual steps later.
'''

from typing import Any
from pathlib import Path

from ..logger import get_logger

LOGGER = get_logger(__name__)


class HistogramPlot:
    '''Plot one signal process and a collection of background processes.'''

    def __init__(
        self,
        variable: str,
        inDir: str,
        outDir: str,
        sel: str,
        plots: dict[str, dict[str, list[str]]],
        colors: dict[str, Any],
        legend: dict[str, str],
        ecm: int = 240,
        lumi: float = 10.8,
    ) -> None:

        self.variable = variable
        self.inDir    = inDir
        self.outDir   = outDir
        self.sel      = sel
        self.plots    = plots
        self.colors   = colors
        self.legend   = legend
        self.ecm      = ecm
        self.lumi     = lumi

        if not plots:
            raise ValueError('HistogramPlot requires a non-empty plots mapping')
        self.sig_processes = plots.get('signal', {})
        self.bkg_processes = plots.get('backgrounds', {})
        self.signals     = list(self.sig_processes)
        self.backgrounds = list(self.bkg_processes)
        self.processes   = [*self.signals, *self.backgrounds]


    def load_histograms(
        self,
        suffix: str,
        rebin: int = 1,
        lazy: bool = True,
    ) -> dict[str, Any]:

        '''Load one histogram for each configured process.'''

        from .root.helper import load_hists

        process_map = dict(self.bkg_processes)
        process_map.update(self.sig_processes)
        return load_hists(
            process_map,
            self.variable,
            self.inDir,
            suffix, rebin, lazy,
        )


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
            hist.SetFilleColor(fill_color)
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
        backgrounds = []
        for process in self.processes:
            hist = histograms.get(process)
            if hist is None:
                continue

            is_signal = process in self.signals
            self.style_hist(
                hist,
                self.colors[process] if is_signal else ROOT.kBlack,
                3 if is_signal else 1,
                self.colors[process] if not is_signal else None,
                sig_scale if is_signal else bkg_scale,
            )
            label = self.legend[process]
            if is_signal and sig_scale != 1:
                label += f' (#times {int(sig_scale)})'
            if not is_signal and bkg_scale != 1:
                label += f' (#times {int(bkg_scale)})'
            legend_obj.AddEntry(hist, label, 'L' if is_signal else 'F')

            if not is_signal:
                stack.Add(hist)
                backgrounds.append(hist)

        missing_signals = [
            signal for signal in self.signals
            if histograms.get(signal) is None
        ]
        if missing_signals:
            LOGGER.warning(f'Could not load signal histograms: {missing_signals}')
        return stack, backgrounds


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
        xMin, xMax, yMin, yMax = self._get_ranges(
            all_hists, backgrounds,
            xmin, xmax, ymin, ymax,
            scale_min, scale_max,
            logY, strict, stack,
        )

        if xtitle in ('', None):
            xTitle = ref_hist.GetXaxis().GetTitle() if xtitle == '' else ''
        else:
            xTitle = xtitle

        bwidth = ref_hist.GetBinWidth(1)
        if   'MeV' in xTitle: unit = 'MeV'
        elif 'GeV' in xTitle: unit = 'GeV'
        elif 'TeV' in xTitle: unit = 'TeV'
        else: unit = ''

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
        from ..tools.process import get_xrange, get_yrange, get_stack

        if not histograms:
            raise ValueError('At least one histogram is required for ranges')

        total = get_stack(histograms)
        xMin, xMax = get_xrange(
            total, strict, xmin, xmax,
        )

        scale_min = min_scale if min_scale is not None else (0.5 if logY else 1.0)
        scale_max = max_scale if max_scale is not None else (1e4 if logY else 1.5)
        y_ranges = [
            get_yrange(
                hist, logY, ymin, ymax, scale_min, scale_max,
            )
            for hist in histograms
        ]
        yMin = min(axis_range[0] for axis_range in y_ranges)

        if stack:
            stacked_range = get_yrange(
                total, logY, ymin, ymax, scale_min, scale_max,
            )
            yMax = stacked_range[1]
        else:
            y_max_hists = list(histograms[:len(histograms) - len(backgrounds)])
            if backgrounds:
                y_max_hists.append(get_stack(backgrounds))
            yMax = max(
                get_yrange(
                    hist, logY, ymin, ymax, scale_min, scale_max,
                )[1]
                for hist in y_max_hists
            )

        return xMin, xMax, yMin, yMax


    def draw(
        self,
        histograms: dict[str, Any],
        stack: Any,
        backgrounds: list[Any],
        legend_obj: Any,
        stack_signals: bool = False,
    ) -> tuple[Any, Any]:

        '''Draw the configured histograms on a standard ROOT canvas.'''

        from .root import plotter

        plotter.cfg = self.cfg
        canvas, dummy = plotter.canvas(), plotter.dummy()
        dummy.Draw('HIST')
        if stack_signals:
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
        format: list[str],
        logY: bool,
        quiet: bool,
    ) -> None:

        '''Finalize and save the canvas using the standard output layout.'''

        from .root.plotter import finalize_canvas
        from .root.helper import save_plot

        base_sel = self.sel.replace('_high', '').replace('_low', '')
        direction = (
            'high' if '_high' in self.sel
            else 'low' if '_low' in self.sel
            else 'nominal'
        )
        category = 'tot' if 'ZH' in self.signals else 'cat'
        out = Path(f'{self.outDir}/{base_sel}/{direction}/{category}')
        out.mkdir(exist_ok=True, parents=True)

        finalize_canvas(canvas)
        save_plot(
            canvas, out, outName,
            ('_log' if logY else '_lin') + suffix,
            format, quiet,
        )


    def plot(
        self,
        suffix: str = '',
        outName: str = '',
        format: list[str] = ['png'],
        ecm: int | None = None,
        lumi: int | None = None,
        xmin: float | int | None = None,
        xmax: float | int | None = None,
        ymin: float | int | None = None,
        ymax: float | int | None = None,
        rebin: int = 1,
        sig_scale: float = 1.,
        bkg_scale: float = 1.,
        xtitle: str = '',
        ytitle: str = 'Events',
        scale_min: float | None = None,
        scale_max: float | None = None,
        strict: bool = True,
        logX: bool = False,
        logY: bool = True,
        stack: bool = False,
        lazy: bool = True,
        quiet: bool = False,
    ) -> None:

        '''Run the complete signal/background plotting workflow.'''

        import ROOT
        ROOT.gROOT.SetBatch(True)
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)

        if ecm  is not None: self.ecm  = ecm
        if lumi is not None: self.lumi = lumi
        if outName == '': outName = self.variable
        histograms = self.load_histograms(f'_{self.sel}_histo', rebin, lazy)
        legend_obj = self.define_legend(len(self.processes))
        stack_obj, backgrounds = self.style_histograms(
            histograms, legend_obj, sig_scale, bkg_scale
        )
        signal_hists = [histograms[signal] for signal in self.signals]
        self.cfg = self.build_config(
            signal_hists, backgrounds,
            xmin, xmax, ymin, ymax,
            logX, logY, xtitle, ytitle,
            scale_min, scale_max,
            strict, stack,
        )
        canvas, _ = self.draw(
            histograms, stack_obj,
            backgrounds, legend_obj,
            stack,
        )
        self.save(canvas, outName, suffix, format, logY, quiet)
        canvas.Close()
