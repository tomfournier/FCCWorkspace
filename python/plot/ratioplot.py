'''Reusable RooFit ratio and pull plot orchestration.'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import ROOT

from . import plotter


class RatioPlot:
    '''Build a two-panel RooFit plot with a data/model pull panel.

    The class deliberately leaves the RooFit drawing options open: callers can
    provide any options accepted by ``plotOn`` and ``paramOn``. The default
    ratio observable is the RooPlot pull, but a caller can provide another
    plotable through ``draw_ratio`` when a different lower panel is needed.
    '''

    def __init__(
        self,
        cfg: dict[str, Any],
        outputdir: str | Path | None = None,
        outName: str = 'ratio',
        suffix: str = '',
        file_formats: list[str] | None = None,
        ratio_lines: Iterable[float] = (0,),
        ratio_colors: Iterable[Any] = (ROOT.kBlack,),
        canvas_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.outputdir = Path(outputdir) if outputdir is not None else None
        self.outName = outName
        self.suffix  = suffix
        self.file_formats = file_formats or ['png']
        self.ratio_lines  = list(ratio_lines)
        self.ratio_colors = list(ratio_colors)
        self.canvas_kwargs = canvas_kwargs or {}
        self.canvas = None
        self.frame  = None
        self.pull   = None
        self._pads  = None


    @staticmethod
    def get_opts(options: Iterable[Any] | None) -> tuple[Any, ...]:
        return tuple(options or ())


    def mk_frame(self, observable, title: str | None = None):
        self.frame = observable.frame()
        if title is not None:
            self.frame.SetTitle(title)
        return self.frame


    def draw_main(
        self,
        data,
        pdfs,
        observable,
        nBins: int = 250,
        data_opts: Iterable[Any] | None = None,
        pdf_opts: Iterable[Any] | None = None,
        param_pdf=None,
        param_opts: Iterable[Any] | None = None,
        title: str | None = None,
    ):
        '''Draw the upper RooPlot and return it with its extracted pull.'''
        frame = self.mk_frame(observable, title)

        data_args = list(self.get_opts(data_opts))
        data_args.append(ROOT.RooFit.Binning(nBins))
        data.plotOn(frame, *data_args)

        pdf_list = [pdfs] if not isinstance(pdfs, (list, tuple)) else list(pdfs)
        for pdf in pdf_list:
            pdf.plotOn(frame, *self.get_opts(pdf_opts))

        if param_pdf is not None:
            param_pdf.paramOn(frame, *self.get_opts(param_opts))

        self.pull = frame.pullHist()
        return frame, self.pull


    def draw_ratio(self, pull=None, frame=None) -> None:
        '''Draw the lower pull panel on the active ratio pad.'''
        if self._pads is None:
            raise RuntimeError('draw() must be called before draw_ratio()')
        padB = self._pads[1]
        padB.cd()
        dummy = self._dummyB
        dummy.Draw('HIST')
        for line in self._lines:
            line.Draw('SAME')
        ratio_frame = self._pull_frame if frame is None else frame
        ratio_frame.addPlotable(self.pull if pull is None else pull, 'P')
        ratio_frame.Draw('SAME')

        return None


    def draw(
        self,
        data,
        pdfs,
        observable,
        nBins: int = 250,
        data_opts: Iterable[Any] | None = None,
        pdf_opts: Iterable[Any] | None = None,
        param_pdf=None,
        param_opts: Iterable[Any] | None = None,
        title: str | None = None,
        annotate: Callable[[Any], None] | None = None,
    ):
        '''Create the canvas, draw both panels, and return the canvas.'''
        self.draw_main(data, pdfs, observable, nBins, data_opts, pdf_opts, param_pdf, param_opts, title)
        self.canvas, padT, padB = plotter.canvasRatio(config=self.cfg, **self.canvas_kwargs)
        self._pads = (padT, padB)
        self._dummyT, self._dummyB, self._lines = plotter.dummyRatio(1, self.ratio_lines, self.ratio_colors, config=self.cfg)

        self.canvas.cd()
        padT.Draw()
        padT.cd()
        self._dummyT.Draw('HIST')
        self.frame.Draw('SAME')

        if annotate is not None: annotate(self.frame)
        plotter.finalize_canvas(self.canvas, config=self.cfg)

        self._pull_frame = observable.frame()
        self.draw_ratio()

        return self.canvas


    def save(
        self,
        outputdir: str | Path | None = None,
        outName: str | None = None,
        suffix: str | None = None,
        file_formats: list[str] | None = None,
    ) -> None:
        destination = Path(outputdir) if outputdir is not None else self.outputdir
        if destination is None:
            raise ValueError('An output directory is required to save a ratio plot')
        plotter.save_canvas(self.canvas, destination, outName or self.outName,
                            self.suffix if suffix is None else suffix, file_formats or self.file_formats)

        return None

    def close(self) -> None:
        if self.canvas is not None:
            self.canvas.Close()
            self.canvas = None

        return None
