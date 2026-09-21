'''Reusable ROOT plot objects.

This module contains the small amount of orchestration shared by the ROOT
histogram plots. The lower-level ROOT details remain in ``plots.root`` so
that specialized plot classes can override individual steps later.
'''

from typing import Any, Optional, Union


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
            suffix=suffix,
            rebin=rebin,
            lazy=lazy,
        )


    def define_legend(self, num_entries: int) -> Any:
        '''Create the legend used by the plot.'''
        from .root.helper import mk_legend

        return mk_legend(num_entries)


    def style_histograms(
        self,
        histograms: dict[str, Any],
        legend_obj: Any,
        sig_scale: float = 1.,
    ) -> tuple[Any, list[Any]]:
        '''Style histograms and return the background stack contents.'''
        import ROOT
        from .root.helper import style_hist

        stack = ROOT.THStack('stack', 'stack')
        backgrounds = []
        for process in self.processes:
            hist = histograms.get(process)
            if hist is None:
                continue

            is_signal = process in self.signals
            style_hist(
                hist,
                color=self.colors[process] if is_signal else ROOT.kBlack,
                width=3 if is_signal else 1,
                fill_color=self.colors[process] if not is_signal else None,
                scale=sig_scale if is_signal else 1.,
            )
            label = self.legend[process]
            if is_signal and sig_scale != 1:
                label += f' (#times {int(sig_scale)})'
            legend_obj.AddEntry(hist, label, 'L' if is_signal else 'F')

            if not is_signal:
                stack.Add(hist)
                backgrounds.append(hist)

        missing_signals = [
            signal for signal in self.signals
            if histograms.get(signal) is None
        ]
        if missing_signals:
            raise RuntimeError(
                f'Could not load signal histograms: {missing_signals}')
        return stack, backgrounds


    def build_config(
        self,
        signal_hists: list[Any],
        backgrounds: list[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        '''Build the ROOT plot configuration for the loaded histograms.'''
        from .root.helper import build_cfg

        config_keys = (
            'logX', 'logY', 'xmin', 'xmax', 'ymin', 'ymax',
            'strict', 'stack',
        )
        reference_hist = signal_hists[0] if signal_hists else backgrounds[0]
        other_hists = (
            [*signal_hists[1:], *backgrounds]
            if signal_hists
            else backgrounds[1:]
        )
        return build_cfg(
            reference_hist,
            ecm=self.ecm,
            lumi=self.lumi,
            hists=other_hists,
            **{key: kwargs[key] for key in config_keys if key in kwargs},
        )


    def draw(
        self,
        histograms: dict[str, Any],
        stack: Any,
        backgrounds: list[Any],
        legend_obj: Any,
    ) -> tuple[Any, Any]:
        '''Draw the configured histograms on a standard ROOT canvas.'''
        from .root import plotter

        plotter.cfg = self.cfg
        canvas, dummy = plotter.canvas(), plotter.dummy()
        dummy.Draw('HIST')
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
        from ..tools.utils import mkdir

        base_sel = self.sel.replace('_high', '').replace('_low', '')
        direction = (
            'high' if '_high' in self.sel
            else 'low' if '_low' in self.sel
            else 'nominal'
        )
        category = 'tot' if 'ZH' in self.signals else 'cat'
        out = f'{self.outDir}/makePlot/{base_sel}/{direction}/{category}'
        mkdir(out)
        finalize_canvas(canvas)
        save_plot(
            canvas,
            out,
            outName,
            ('_log' if logY else '_lin') + suffix,
            format,
            quiet,
        )


    def plot(
        self,
        suffix: str = '',
        outName: str = '',
        format: list[str] = ['png'],
        ecm: Optional[int] = None,
        lumi: Optional[float] = None,
        xmin: Optional[Union[float, int]] = None,
        xmax: Optional[Union[float, int]] = None,
        ymin: Optional[Union[float, int]] = None,
        ymax: Optional[Union[float, int]] = None,
        rebin: int = 1,
        sig_scale: float = 1.,
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

        if ecm is not None:
            self.ecm = ecm
        if lumi is not None:
            self.lumi = lumi
        if outName == '':
            outName = self.variable
        histograms = self.load_histograms(
            suffix=f'_{self.sel}_histo',
            rebin=rebin,
            lazy=lazy,
        )
        legend_obj = self.define_legend(len(self.processes))
        stack_obj, backgrounds = self.style_histograms(
            histograms, legend_obj, sig_scale=sig_scale
        )
        signal_hists = [histograms[signal] for signal in self.signals]
        self.cfg = self.build_config(
            signal_hists,
            backgrounds,
            logX=logX,
            logY=logY,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            strict=strict,
            stack=stack,
        )
        canvas, dummy = self.draw(
            histograms,
            stack_obj,
            backgrounds,
            legend_obj,
        )
        self.save(canvas, outName, suffix, format, logY, quiet)
        canvas.Close()
