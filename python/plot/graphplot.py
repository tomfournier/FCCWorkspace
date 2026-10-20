'''Reusable ROOT graph plot orchestration.'''

from array import array
from numbers import Real
from pathlib import Path
from typing import Any, Sequence


class GraphPlot:
    '''Draw one or more ROOT graphs with shared canvas and legend handling.'''

    def __init__(
        self,
        outputdir: str | Path,
        graph: Any | None = None,
        cfg: dict[str, Any] | None = None,
        ecm: int = 240,
        lumi: float = 10.8,
        file_formats: list[str] | None = None,
    ) -> None:
        
        self.outputdir = Path(outputdir)

        self.graph = graph
        self.cfg = cfg or {}
        self.ecm = ecm
        self.lumi = lumi
        self.file_formats = file_formats or ['png']
        self.graphs: list[tuple[Any, str, Any, str]] = []
        self.ref_lines: list[Any] = []

        if graph is not None:
            self.add(graph)

    def build_config(
        self,
        graph: Any | None = None,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        xtitle: str = '',
        ytitle: str = 'Events',
        topLeft: str | None = None,
        topRight: str | None = None,
    ) -> dict[str, Any]:
        '''Build plotter configuration from a ROOT graph and optional limits.'''

        from .plotter import make_cfg

        graph = graph or self.graph or (self.graphs[0][0] if self.graphs else None)
        if graph is None:
            raise ValueError('A graph is required to build the plot configuration')

        hist = graph.GetHistogram()
        xaxis, yaxis = hist.GetXaxis(), hist.GetYaxis()
        config = {'xmin': xaxis.GetXmin() if xmin is None else xmin,
                  'xmax': xaxis.GetXmax() if xmax is None else xmax,
                  'ymin': yaxis.GetXmin() if ymin is None else ymin,
                  'ymax': yaxis.GetXmax() if ymax is None else ymax,
                  'xtitle': xtitle, 'ytitle': ytitle}
        if topLeft is not None:
            config['topLeft'] = topLeft
        if topRight is not None:
            config['topRight'] = topRight
        self.cfg = make_cfg(config, self.ecm, self.lumi)
        return self.cfg

    @staticmethod
    def from_xy(
        x_vals: Sequence[float],
        y_vals: Sequence[float],
        graph_type: str = 'TGraph',
    ) -> Any:
        import ROOT

        if len(x_vals) != len(y_vals):
            raise ValueError('x_values and y_values must have equal lengths')
        graph_class = getattr(ROOT, graph_type)
        return graph_class(len(x_vals),
                           array('d', x_vals),
                           array('d', y_vals))

    def add(
        self,
        graph: Any,
        label: str = '',
        color: Any = None,
        width: int = 2,
        marker: int | None = None,
        marker_size: float = 1,
        draw_option: str = 'L',
    ) -> Any:
        if color is not None:
            graph.SetLineColor(color)
            graph.SetMarkerColor(color)
        graph.SetLineWidth(width)
        if marker is not None:
            graph.SetMarkerStyle(marker)
            graph.SetMarkerSize(marker_size)
        self.graphs.append((graph, label, color, draw_option))
        return graph

    @staticmethod
    def style_graph(
        graph: Any,
        color: Any,
        width: int = 4,
        marker: int = 20,
        size: float = 1,
    ) -> Any:
        graph.SetMarkerStyle(marker)
        graph.SetMarkerColor(color)
        graph.SetMarkerSize(size)
        graph.SetLineColor(color)
        graph.SetLineWidth(width)
        return graph

    def _legend(
            self,
            header: str = '',
            x1: float = 0.2,
            y1: float = 0.825,
            x2: float = 0.9,
            y2: float = 0.9,
            border_size: int = 0,
            fill_style: int = 0,
            text_size: float = 0.035,
            set_margin: float = 0.15,
            return_leg: bool = False
    ) -> Any:
        import ROOT

        leg = ROOT.TLegend(x1, y1, x2, y2)
        leg.SetBorderSize(border_size)
        leg.SetFillStyle(fill_style)
        leg.SetTextSize(text_size)
        leg.SetMargin(set_margin)
        if header:
            leg.SetHeader(header)
    
        self.leg = leg
        if return_leg:
            return leg
        return None

    def draw(
        self,
        outName: str,
        suffix: str = '',
        ref_y: list[float] | float | None = None,
        legend_header: str = '',
        legend: bool = True,
        ref_color: list[Any] | Any = None,
    ) -> Any:
        import ROOT
        from . import plotter

        if not self.graphs:
            raise ValueError('GraphPlot requires at least one graph')
        if not self.cfg:
            self.build_config()

        plotter.cfg = self.cfg
        canvas, dummy = plotter.canvas(), plotter.dummy()
        dummy.GetXaxis().SetNdivisions(507)
        dummy.Draw('HIST')

        for index, (graph, _, _, draw_option) in enumerate(self.graphs):
            graph.Draw(f'{draw_option} SAME' if index else f'{draw_option} SAME')

        if ref_y is None:
            ref_values = []
        elif isinstance(ref_y, Real):
            ref_values = [ref_y]
        else:
            ref_values = list(ref_y)

        if ref_color is None or isinstance(ref_color, Real):
            color = ROOT.kBlack if ref_color is None else ref_color
            ref_colors = [color] * len(ref_values)
        else:
            ref_colors = list(ref_color)
            if len(ref_values) != len(ref_colors):
                raise ValueError(
                    'ref_y and ref_color should be the same size '
                    f'({len(ref_values)} vs {len(ref_colors)})'
                )

        for y, color in zip(ref_values, ref_colors):
            ref_line = ROOT.TLine(self.cfg['xmin'], y, self.cfg['xmax'], y)
            ref_line.SetLineColor(color)
            ref_line.SetLineWidth(2)
            ref_line.Draw('SAME')
            self.ref_lines.append(ref_line)

        if legend:
            root_legend = self._legend(legend_header, return_leg=True)
            for graph, label, _, draw_option in self.graphs:
                if label:
                    root_legend.AddEntry(graph, label, draw_option)
            root_legend.Draw()

        plotter.finalize_canvas(canvas)
        plotter.save_canvas(canvas, self.outputdir, outName, suffix, self.file_formats)
        return canvas
