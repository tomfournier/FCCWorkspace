'''Reusable ROOT pull and reference-band plots.'''

from typing import Any, Sequence
from pathlib import Path


class PullPlot:
	'''Draw values with uncertainties against labelled rows.'''

	def __init__(
		self,
		outputdir: str,
		output_subdir: str,
		selection: str,
		values: Sequence[float],
		errors: Sequence[float],
		labels: Sequence[str],
		ref: float | None = None,
		ref_err: float = 0.,
		ecm: int = 240,
		lumi: float = 10.8,
		xtitle: str = '',
		name: str = 'pulls',
		reference_row: bool = True,
		reference_marker: bool = True,
	) -> None:
		if not values:
			raise ValueError('PullPlot requires at least one value')
		if len(values) != len(errors) or len(values) != len(labels):
			raise ValueError('values, errors, and labels must have equal lengths')

		self.values = list(values)
		self.errors = list(errors)
		self.labels = list(labels)
		self.reference       = ref
		self.reference_error = ref_err
		self.reference_row   = reference_row
		self.reference_marker = reference_marker
		self.ecm  = ecm
		self.lumi = lumi
		self.xtitle = xtitle
		self.name   = name
		self.selection = selection
		self.outputdir = outputdir
		self.output_subdir = output_subdir

	@property
	def row_count(self) -> int:
		return len(self.labels) + (1 if self.reference is not None and self.reference_row else 0)

	def _x_range(self) -> tuple[int, int]:
		values = self.values + ([self.reference] if self.reference is not None else [])
		return int(min(values)) - 5, int(max(values)) + 3

	def _histogram(self, xmin: float, xmax: float) -> Any:
		import ROOT

		histogram = ROOT.TH2F(self.name, self.name, max(1, int((xmax - xmin) * 10)),
		                      xmin, xmax, self.row_count, 0, self.row_count)
		offset = 0
		if self.reference is not None and self.reference_row:
			histogram.GetYaxis().SetBinLabel(1, 'Average')
			offset = 1
		for index, label in enumerate(self.labels):
			histogram.GetYaxis().SetBinLabel(index + offset + 1, label)
		return histogram

	def _config(
		self,
		xmin: float,
		xmax: float,
		top_left: str,
		top_right: str,
	) -> dict[str, Any]:
		from .plotter import make_cfg

		return make_cfg({
			'xmin': xmin,
			'xmax': xmax,
			'ymin': 0,
			'ymax': self.row_count,
			'xtitle': self.xtitle,
			'topLeft':  top_left,
			'topRight': top_right,
		}, self.ecm, self.lumi)


	def output_dir(self) -> Path:
		'''Return the configured output directory for this selection.'''
		if self.outputdir is None:
			raise ValueError('outputdir is required for automatic paths')
		if not self.output_subdir:
			return Path(self.outputdir)
		if self.selection is None:
			raise ValueError('selection is required for automatic paths')

		base_selection = self.selection.replace('_high', '').replace('_low', '')
		direction = ('high' if '_high' in self.selection else 'low' if '_low' in self.selection else 'nominal')
		return Path(f'{self.outputdir}/{self.output_subdir}/{base_selection}/{direction}')


	def draw(
		self,
		outName: str,
		suffix: str = '',
		file_formats: list[str] = ['png'],
		quiet: bool = False,
		xmin: float | None = None,
		xmax: float | None = None,
		top_left: str = '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
		top_right: str | None = None,
		statistics: Sequence[str] = (),
		point_colors: Sequence[Any] | None = None,
	) -> None:
		import ROOT
		from . import plotter

		default_min, default_max = self._x_range()
		xmin = default_min if xmin is None else xmin
		xmax = default_max if xmax is None else xmax
		top_right = top_right or f'#sqrt{{s}} = {self.ecm} GeV, {self.lumi} ab^{{#minus1}}'

		plotter.cfg = self._config(xmin, xmax, top_left, top_right)
		hist = self._histogram(xmin, xmax)
		canvas = plotter.canvas(800, 800)
		plotter.canvas_margins(canvas, top=0.08, bottom=0.1, left=0.15, right=0.05)
		canvas.SetFillStyle(4000)
		canvas.SetGrid(1, 0)
		canvas.SetTickx(1)

		hist.GetXaxis().SetTitle(self.xtitle)
		hist.GetXaxis().SetTitleSize(0.04)
		hist.GetXaxis().SetLabelSize(0.035)
		hist.GetXaxis().SetTitleOffset(1)
		hist.GetYaxis().SetLabelSize(0.055)
		hist.GetYaxis().SetTickLength(0)
		hist.GetYaxis().LabelsOption('v')
		hist.SetNdivisions(506, 'XYZ')
		hist.Draw('HIST 0')

		line = None
		band = None
		if self.reference is not None:
			line = ROOT.TLine(self.reference, 0, self.reference, self.row_count)
			line.SetLineColor(ROOT.kGray)
			line.SetLineWidth(2)
			line.Draw('SAME')

			band = ROOT.TGraph()
			band.SetPoint(0, self.reference-self.reference_error, 0)
			band.SetPoint(1, self.reference+self.reference_error, 0)
			band.SetPoint(2, self.reference+self.reference_error, self.row_count)
			band.SetPoint(3, self.reference-self.reference_error, self.row_count)
			band.SetPoint(4, self.reference-self.reference_error, 0)
			band.SetFillColor(16)
			band.SetFillColorAlpha(16, 0.35)
			band.Draw('SAME F')

		offset = 1 if self.reference is not None and self.reference_row else 0
		if point_colors is not None and len(point_colors) != len(self.values):
			raise ValueError('point_colors must match the number of values')

		graphs = []
		if self.reference is not None and self.reference_marker:
			reference_graph = ROOT.TGraphErrors(1)
			reference_graph.SetPoint(0, self.reference, 0.5)
			reference_graph.SetPointError(0, self.reference_error, 0.)
			graphs.append((reference_graph, None))

		if point_colors is None:
			groups = [(None, list(range(len(self.values))))]
		else:
			groups = []
			for index, color in enumerate(point_colors):
				for group_color, group_indices in groups:
					if group_color == color:
						group_indices.append(index)
						break
				else:
					groups.append((color, [index]))

		for color, indices in groups:
			graph = ROOT.TGraphErrors(len(indices))
			for point, index in enumerate(indices):
				graph.SetPoint(point, self.values[index], index + offset + 0.5)
				graph.SetPointError(point, self.errors[index], 0.)
			graphs.append((graph, color))

		for graph, color in graphs:
			if color is not None:
				graph.SetMarkerColor(color)
			graph.SetMarkerSize(1.2)
			graph.SetMarkerStyle(20)
			graph.SetLineWidth(2)
			graph.Draw('P0 SAME')

		latex = plotter.setup_latex(0.045, 30, 1, 42)
		latex.DrawLatex(0.95, 0.925, top_right)
		latex = plotter.setup_latex(0.045, 13, 1, 42)
		latex.DrawLatex(0.15, 0.96, top_left)
		if statistics:
			text = plotter.setup_latex(0.04, 11, 1, 42)
			for index, value in enumerate(statistics):
				text.DrawLatex(0.2, 0.2 - index * 0.05, value)
			text.Draw('SAME')

		plotter.save_canvas(canvas, self.output_dir(), outName, suffix, file_formats, quiet)
		canvas.Close()

		return None


	def save_text(
		self,
		outName: str,
		label: str = 'Value',
		value_format: str = '.2f',
		error_format: str = '.2f',
		separator: str = '+/-',
		col_width: int = 15,
	) -> None:
		'''Write the plotted values and uncertainties as a fixed-width table.'''

		out = self.output_dir()
		out.mkdir(exist_ok=True, parents=True)

		header = f"{'Labels':<{col_width}}" + ''.join(f'{label:<{col_width}}' for label in self.labels)
		separator_line = '-' * col_width * (len(self.labels) + 1)
		values = [f'{value:{value_format}}{separator}{error:{error_format}}'
		          for value, error in zip(self.values, self.errors)]
		row = f'{label:<{col_width}}' + ''.join(f'{value:<{col_width}}' for value in values)

		with open(out / f'{outName}.txt', 'w') as text_file:
			text_file.write('\n'.join((header, separator_line, row)) + '\n')

		return None
