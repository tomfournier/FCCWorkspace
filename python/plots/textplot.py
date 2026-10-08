'''Reusable ROOT canvases for text and table-style plots.'''

from pathlib import Path
from typing import Any, Sequence


class TextPlot:
	'''Render metadata, legends, and tabular rows on a ROOT canvas.'''

	def __init__(
		self,
		out_dir: str,
		selection: str,
		output_subdir: str = 'yield',
		ecm: int = 240,
		lumi: float = 10.8,
		column_positions: Sequence[float] = (0.18, 0.5, 0.75),
		canvas_size: tuple[int, int] = (1000, 1000),
	) -> None:
		if len(column_positions) == 0:
			raise ValueError('TextPlot requires at least one column position')

		self.out_dir = out_dir
		self.selection = selection
		self.output_subdir = output_subdir
		self.ecm = ecm
		self.lumi = lumi
		self.column_positions = tuple(column_positions)
		self.canvas_size = canvas_size

	def output_dir(self) -> Path:
		'''Return the nominal/high/low directory for the configured selection.'''
		base_selection = self.selection.replace('_high', '').replace('_low', '')
		direction = (
			'high' if '_high' in self.selection
			else 'low' if '_low' in self.selection
			else 'nominal'
		)
		return Path(self.out_dir) / self.output_subdir / base_selection / direction

	def load_yields(
		self,
		h_name: str,
		in_dir: str,
		plots: dict[str, dict[str, list[str]]],
		labels: dict[str, str],
		colors: dict[str, Any],
		signal_scale: float = 1.,
		background_scale: float = 1.,
		lazy: bool = True,
	) -> tuple[list[tuple[str, float, float]], Any, float, float]:
		'''Load, style, and summarize process histograms for a text table.'''
		import ROOT
		from tools.process import getHist

		suffix = f'_{self.selection}_histo'
		processes = [*plots.get('signals', {}), *plots.get('backgrounds', {})]
		legend = ROOT.TLegend(0.6, 0.86 - len(processes) * 0.06, 0.9, 0.88)
		legend.SetBorderSize(0)
		legend.SetFillStyle(0)
		legend.SetTextFont(42)
		legend.SetTextSize(0.03)
		legend.SetMargin(0.2)

		rows = []
		signal_total = 0.
		background_total = 0.
		for process in processes:
			group = 'signals' if process in plots.get('signals', {}) else 'backgrounds'
			scale = signal_scale if group == 'signals' else background_scale
			process_hist = getHist(h_name, plots[group][process], in_dir,
								   suffix, lazy=lazy, use_cache=False)
			if process_hist is None:
				continue

			integral = process_hist.Integral() * scale
			entries = process_hist.GetEntries()
			process_hist.SetLineColor(colors[process] if group == 'signals' else ROOT.kBlack)
			process_hist.SetLineWidth(4 if group == 'signals' else 1)
			process_hist.SetLineStyle(1)
			if group == 'backgrounds':
				process_hist.SetFillColor(colors[process])
			if scale != 1.:
				process_hist.Scale(scale)
			legend.AddEntry(process_hist, labels[process], 'L' if group == 'signals' else 'F')

			rows.append((labels[process], integral, entries))
			if group == 'signals':
				signal_total += integral
			else:
				background_total += integral

		return rows, legend, signal_total, background_total

	def draw(
		self,
		rows: Sequence[Sequence[Any]],
		metadata: Sequence[tuple[str, float, float, float]],
		out_name: str,
		headers: Sequence[str] = ('Process', 'Yields', 'Raw MC'),
		legend: Any | None = None,
		file_formats: list[str] = ['png'],
		quiet: bool = False,
		suffix: str = '',
	) -> None:
		'''Draw and save a text table with optional metadata and legend.'''
		import ROOT
		from .root import plotter
		from .root.helper import draw_latex, setup_latex

		if len(headers) != len(self.column_positions):
			raise ValueError('headers and column_positions must have equal lengths')
		if any(len(row) != len(headers) for row in rows):
			raise ValueError('every row must have one value per header')

		canvas = plotter.canvas(
			*self.canvas_size, top=0., bottom=0., left=0.14, right=0.08
		)
		dummy = ROOT.TH1F(f'{out_name}_dummy', '', 1, 0, 1)
		dummy.SetStats(0)
		plotter.configure_axis(dummy.GetXaxis(), '', 0, 1,
							   label_offset=999, label_size=0)
		plotter.configure_axis(dummy.GetYaxis(), '', 0, 1,
							   label_offset=999, label_size=0)
		dummy.Draw('AH')
		if legend is not None:
			legend.Draw()

		text_data = list(metadata)
		text_data.extend(
			(f'#bf{{#it{{{header}}}}}', self.column_positions[index], 0.45, 0.035)
			for index, header in enumerate(headers)
		)
		for row_index, row in enumerate(rows):
			y = 0.4 - row_index * 0.05
			text_data.extend(
				(f'#bf{{#it{{{value}}}}}', self.column_positions[index], y, 0.035)
				for index, value in enumerate(row)
			)
		latex = setup_latex(0.035, 12)
		draw_latex(latex, text_data)

		output = self.output_dir()
		output.mkdir(parents=True, exist_ok=True)
		from .root.plotter import save_canvas
		save_canvas(canvas, output, out_name, suffix, file_formats, quiet)
		canvas.Close()
