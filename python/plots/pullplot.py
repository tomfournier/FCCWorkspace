'''Reusable ROOT pull and reference-band plots.'''

from typing import Any, Sequence


class PullPlot:
	'''Draw values with uncertainties against labelled rows.'''

	def __init__(
		self,
		values: Sequence[float],
		errors: Sequence[float],
		labels: Sequence[str],
		reference: float | None = None,
		reference_error: float = 0.,
		ecm: int = 240,
		lumi: float = 10.8,
		x_title: str = '',
		name: str = 'pulls',
		out_dir: str | None = None,
		selection: str | None = None,
		output_subdir: str = 'yield',
	) -> None:
		if not values:
			raise ValueError('PullPlot requires at least one value')
		if len(values) != len(errors) or len(values) != len(labels):
			raise ValueError('values, errors, and labels must have equal lengths')

		self.values = list(values)
		self.errors = list(errors)
		self.labels = list(labels)
		self.reference       = reference
		self.reference_error = reference_error
		self.ecm  = ecm
		self.lumi = lumi
		self.x_title = x_title
		self.name    = name
		self.out_dir = out_dir
		self.selection = selection
		self.output_subdir = output_subdir

	@property
	def row_count(self) -> int:
		return len(self.labels) + (1 if self.reference is not None else 0)

	def _x_range(self) -> tuple[int, int]:
		values = self.values + ([self.reference] if self.reference is not None else [])
		return int(min(values)) - 5, int(max(values)) + 3

	def _histogram(self, x_min: float, x_max: float) -> Any:
		import ROOT

		histogram = ROOT.TH2F(self.name, self.name, max(1, int((x_max - x_min) * 10)),
		                      x_min, x_max, self.row_count, 0, self.row_count)
		offset = 0
		if self.reference is not None:
			histogram.GetYaxis().SetBinLabel(1, 'Average')
			offset = 1
		for index, label in enumerate(self.labels):
			histogram.GetYaxis().SetBinLabel(index + offset + 1, label)
		return histogram

	def _config(
		self,
		x_min: float,
		x_max: float,
		top_left: str,
		top_right: str,
	) -> dict[str, Any]:
		from .root.helper import make_cfg

		return make_cfg({
			'xmin': x_min,
			'xmax': x_max,
			'ymin': 0,
			'ymax': self.row_count,
			'xtitle': self.x_title,
			'topLeft':  top_left,
			'topRight': top_right,
		}, self.ecm, self.lumi)


	def output_dir(self) -> str:
		'''Return the configured output directory for this selection.'''
		if self.out_dir is None or self.selection is None:
			raise ValueError('out_dir and selection are required for automatic paths')

		base_selection = self.selection.replace('_high', '').replace('_low', '')
		direction = ('high' if '_high' in self.selection
		             else 'low' if '_low' in self.selection
					 else 'nominal')
		return f'{self.out_dir}/{self.output_subdir}/{base_selection}/{direction}'


	def draw(
		self,
		out_dir: str | None,
		out_name: str,
		suffix: str = '',
		file_formats: list[str] = ['png'],
		quiet: bool = False,
		x_min: float | None = None,
		x_max: float | None = None,
		top_left: str = '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
		top_right: str | None = None,
		statistics: Sequence[str] = (),
	) -> None:
		import ROOT
		from .root import plotter
		from .root.plotter import save_canvas
		from .root.helper import setup_latex

		default_min, default_max = self._x_range()
		x_min = default_min if x_min is None else x_min
		x_max = default_max if x_max is None else x_max
		top_right = top_right or f'#sqrt{{s}} = {self.ecm} GeV, {self.lumi} ab^{{#minus1}}'

		plotter.cfg = self._config(x_min, x_max, top_left, top_right)
		histogram = self._histogram(x_min, x_max)
		canvas = plotter.canvas(800, 800)
		plotter.canvas_margins(canvas, top=0.08, bottom=0.1, left=0.15, right=0.05)
		canvas.SetFillStyle(4000)
		canvas.SetGrid(1, 0)
		canvas.SetTickx(1)

		histogram.GetXaxis().SetTitle(self.x_title)
		histogram.GetXaxis().SetTitleSize(0.04)
		histogram.GetXaxis().SetLabelSize(0.035)
		histogram.GetXaxis().SetTitleOffset(1)
		histogram.GetYaxis().SetLabelSize(0.055)
		histogram.GetYaxis().SetTickLength(0)
		histogram.GetYaxis().LabelsOption('v')
		histogram.SetNdivisions(506, 'XYZ')
		histogram.Draw('HIST 0')

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

		offset = 1 if self.reference is not None else 0
		graph = ROOT.TGraphErrors(len(self.values) + offset)
		if self.reference is not None:
			graph.SetPoint(0, self.reference, 0.5)
			graph.SetPointError(0, self.reference_error, 0.)
		for index, (value, error) in enumerate(zip(self.values, self.errors)):
			graph.SetPoint(index + offset, value, index + offset + 0.5)
			graph.SetPointError(index + offset, error, 0.)
		graph.SetMarkerSize(1.2)
		graph.SetMarkerStyle(20)
		graph.SetLineWidth(2)
		graph.Draw('P0 SAME')

		latex = setup_latex(0.045, 30, 1, 42)
		latex.DrawLatex(0.95, 0.925, top_right)
		latex = setup_latex(0.045, 13, 1, 42)
		latex.DrawLatex(0.15, 0.96, top_left)
		if statistics:
			text = setup_latex(0.04, 11, 1, 42)
			for index, value in enumerate(statistics):
				text.DrawLatex(0.2, 0.2 - index * 0.05, value)
			text.Draw('SAME')

		save_canvas(canvas, out_dir or self.output_dir(), out_name,
		            suffix, file_formats, quiet)
		canvas.Close()

		return None


	def save_text(
		self,
		out_dir: str,
		out_name: str,
		row_label: str = 'Value',
		value_format: str = '.2f',
		error_format: str = '.2f',
		separator: str = '+/-',
		column_width: int = 15,
	) -> None:
		'''Write the plotted values and uncertainties as a fixed-width table.'''
		from pathlib import Path

		Path(out_dir).mkdir(parents=True, exist_ok=True)
		header = f"{'Labels':<{column_width}}" + ''.join(
			f'{label:<{column_width}}' for label in self.labels)
		separator_line = '-' * column_width * (len(self.labels) + 1)
		values = [f'{value:{value_format}}{separator}{error:{error_format}}'
		          for value, error in zip(self.values, self.errors)]
		row = f'{row_label:<{column_width}}' + ''.join(
			f'{value:<{column_width}}' for value in values)

		with open(Path(out_dir) / f'{out_name}.txt', 'w') as text_file:
			text_file.write('\n'.join((header, separator_line, row)) + '\n')

		return None
