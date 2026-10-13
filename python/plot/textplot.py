'''Reusable ROOT canvases for text and table-style plots.'''

################################
### STANDARD LIBRARY IMPORTS ###
################################

from pathlib import Path
from typing import Any, Sequence



#########################################
### LOGGING AND CUSTOM MODULE IMPORTS ###
#########################################

from logger import get_logger
LOGGER = get_logger(__name__)



#######################
### TEXT PLOT CLASS ###
#######################

class TextPlot:
	'''Render metadata, legends, and tabular rows on a ROOT canvas.'''

	def __init__(
		self,
		outputdir: str,
		output_subdir: str,
		selection: str,
		ecm: int = 240,
		lumi: float = 10.8,
		column_positions: Sequence[float] = (0.065, 0.35, 0.7),
		canvas_size: tuple[int, int] = (1000, 1000),
	) -> None:
		if len(column_positions) == 0:
			raise ValueError('TextPlot requires at least one column position')

		self.outputdir     = outputdir
		self.output_subdir = output_subdir
		self.selection = selection
		self.ecm  = ecm
		self.lumi = lumi
		self.col_pos = tuple(column_positions)
		self.canvas_size = canvas_size

	def output_dir(self) -> Path:
		'''Return the nominal/high/low directory for the configured selection.'''
		base_selection = self.selection.replace('_high', '').replace('_low', '')
		direction = ('high' if '_high' in self.selection else 'low' if '_low' in self.selection
					 else 'nominal')
		return Path(self.outputdir) / self.output_subdir / base_selection / direction


	def define_legend(
			self,
			num_entries: int,
			columns: int = 1,
			x1: float = 0.7,
			y1: float = 0.9,
			x2: float = 0.97,
			y2: float = 0.92,
			border_size: int = 0,
			fill_style: int = 0,
			text_size: float = 0.03,
			set_margin: float = 0.2,
			text_font: int = -1,
			return_leg: bool = False
	) -> Any | None:
		
		import ROOT
		self.leg = ROOT.TLegend(x1, y1 - num_entries * 0.06 / columns, x2, y2)

		if text_font != -1:
			self.leg.SetTextFont(text_font)
		self.leg.SetBorderSize(border_size)
		self.leg.SetFillStyle(fill_style)
		self.leg.SetTextSize(text_size)
		self.leg.SetMargin(set_margin)
		self.leg.SetNColumns(columns)

		if return_leg:
			return self.leg
		return None


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

		rows = []
		sig_tot, bkg_tot = 0, 0

		procs = [*plots.get('signals', {}), *plots.get('backgrounds', {})]
		for proc in procs:
			group = 'signals' if proc in plots.get('signals', {}) else 'backgrounds'
			scale = signal_scale if group == 'signals' else background_scale
			proc_hist = getHist(h_name, plots[group][proc], in_dir,
			                    suffix, lazy=lazy, use_cache=False)
			if proc_hist is None: continue

			integral = proc_hist.Integral() * scale
			entries  = proc_hist.GetEntries()
			proc_hist.SetLineColor(colors[proc] if group == 'signals' else ROOT.kBlack)
			proc_hist.SetLineWidth(4 if group == 'signals' else 1)
			proc_hist.SetLineStyle(1)
			if group == 'backgrounds':
				proc_hist.SetFillColor(colors[proc])
			if scale != 1.:
				proc_hist.Scale(scale)
			self.leg.AddEntry(proc_hist, labels[proc], 'L' if group == 'signals' else 'F')

			rows.append((labels[proc], integral, entries))
			if group == 'signals': sig_tot += integral
			else:                  bkg_tot += integral

		return rows, sig_tot, bkg_tot

	def draw(
		self,
		rows: Sequence[Sequence[Any]],
		metadata: Sequence[tuple[str, float, float, float]],
		out_name: str,
		headers: Sequence[str] = ('Process', 'Yields', 'Raw MC'),
		file_formats: list[str] = ['png'],
		quiet: bool = False,
		suffix: str = '',
	) -> None:
		'''Draw and save a text table with optional metadata and legend.'''
		import ROOT
		from . import plotter

		if len(headers) != len(self.col_pos):
			raise ValueError('headers and column_positions must have equal lengths')
		if any(len(row) != len(headers) for row in rows):
			raise ValueError('every row must have one value per header')

		canvas = plotter.canvas(*self.canvas_size, top=0.08, bottom=0.05, left=0.05, right=0.05)
		dummy = ROOT.TH1F(f'{out_name}_dummy', '', 1, 0, 1)
		dummy.SetStats(0)

		plotter.configure_axis(dummy.GetXaxis(), '', 0, 1, label_offset=999, label_size=0)
		plotter.configure_axis(dummy.GetYaxis(), '', 0, 1, label_offset=999, label_size=0)

		# Remove ticks from canvas
		dummy.GetXaxis().SetTickLength(0)
		dummy.GetYaxis().SetTickLength(0)

		dummy.Draw('AH')
		if hasattr(self, 'leg'): self.leg.Draw()
		else: LOGGER.warning('No legend was defined, not drawing it')

		text_data = list(metadata)
		text_data.extend((f'#bf{{#it{{{header}}}}}', self.col_pos[index], 0.5, 0.045)
		                 for index, header in enumerate(headers))
		for row_index, row in enumerate(rows):
			y = 0.445 - row_index * 0.05
			text_data.extend((f'#it{{{value}}}', self.col_pos[index], y, 0.035)
			                 for index, value in enumerate(row))
		latex = plotter.setup_latex(0.035, 12)
		plotter.draw_latex(latex, text_data)

		output = self.output_dir()
		output.mkdir(parents=True, exist_ok=True)
		plotter.save_canvas(canvas, output, out_name, suffix, file_formats, quiet)
		canvas.Close()
