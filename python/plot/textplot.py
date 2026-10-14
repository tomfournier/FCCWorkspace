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
		inputdir: str,
		outputdir: str,
		output_subdir: str,
		selection: str,
		ecm: int = 240,
		lumi: float = 10.8,
		tot: bool = False,
		column_positions: Sequence[float] = (0.065, 0.35, 0.7),
		canvas_size: tuple[int, int] = (1000, 1000),
	) -> None:
		if len(column_positions) == 0:
			raise ValueError('TextPlot requires at least one column position')

		self.inputdir  = inputdir
		self.outputdir = outputdir
		self.output_subdir = output_subdir
		self.selection = selection
		self.ecm  = ecm
		self.lumi = lumi
		self.col_pos = tuple(column_positions)
		self.canvas_size = canvas_size
		self.tot = tot


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
		hName: str,
		plots: dict[str, dict[str, list[str]]],
		colors: dict[str, Any],
		legend: dict[str, str],
		sig_scale: float = 1.,
		bkg_scale: float = 1.,
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
			scale = sig_scale if group == 'signals' else bkg_scale
			proc_hist = getHist(hName, plots[group][proc], self.inputdir,
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
			self.leg.AddEntry(proc_hist, legend[proc], 'L' if group == 'signals' else 'F')

			rows.append((legend[proc], integral, entries))
			if group == 'signals': sig_tot += integral
			else:                  bkg_tot += integral

		return rows, sig_tot, bkg_tot

	def draw(
		self,
		rows: Sequence[Sequence[Any]],
		metadata: Sequence[tuple[str, float, float, float]],
		headers: Sequence[str] = ('Process', 'Yields', 'Raw MC'),
	) -> Any:
		'''Draw and save a text table with optional metadata and legend.'''

		import ROOT
		from . import plotter

		if len(headers) != len(self.col_pos):
			raise ValueError('headers and column_positions must have equal lengths')
		if any(len(row) != len(headers) for row in rows):
			raise ValueError('every row must have one value per header')

		plotter.cfg = plotter.make_cfg({
			'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1,
			'topLeft': '', 'topRight': ''}, self.ecm, self.lumi)
		canvas = plotter.canvas(*self.canvas_size, top=0.08, bottom=0.05,
							left=0.05, right=0.05)
		dummy = plotter.dummy()
		dummy.GetYaxis().SetLabelOffset(999)
		self.dummy = dummy

		dummy.GetXaxis().SetTickLength(0)
		dummy.GetYaxis().SetTickLength(0)

		dummy.Draw('AXIS')
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

		return canvas


	def output_dir(self) -> Path:
		'''Return the nominal/high/low directory for the configured selection.'''
		base_selection = self.selection.replace('_high', '').replace('_low', '')
		direction = ('high' if '_high' in self.selection else 'low' if '_low' in self.selection
						else 'nominal')
		return Path(self.outputdir) / self.output_subdir / base_selection / direction


	def save(
		self,
		canvas,
		outName: str,
		suffix: str,
		file_formats: list[str] = ['png'],
		quiet: bool = False
	) -> None: 
		'''Finalize and save the canvas using the standard output layout.'''

		from .plotter import finalize_canvas, save_canvas
		out = self.output_dir()
		out.mkdir(exist_ok=True, parents=True)

		suffix = ('_tot' if self.tot else '') + suffix

		finalize_canvas(canvas)
		save_canvas(canvas, out, outName, suffix, file_formats, quiet)

		return None
