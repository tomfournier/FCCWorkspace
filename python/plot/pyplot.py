'''Reusable Matplotlib plot objects.

The domain-specific code prepares arrays and plot specifications while
``MatplotlibPlot`` owns figure creation, drawing, labels, and output.
'''

from __future__ import annotations

import matplotlib.pyplot as plt

from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Sequence


class MatplotlibPlot:
	'''Small orchestration wrapper for common Matplotlib plot workflows.'''

	_ARITY = {'plot': 2, 'line': 2, 'step': 2, 'scatter': 2, 'error': 4, 'errorbar': 4,
	          'bar': 2, 'hist': 1, 'fill_between': 3, 'contour': 3, 'axvline': 1, 'axhline': 1}
	_KINDS = set(_ARITY)

	def __init__(
		self,
		outputdir: str | PathLike[str] | None = None,
		output_subdir: str | None = None,
		selection: str | None = None,
		plotname: str = 'plot',
		suffix: str = '',
		file_formats: list[str] | tuple[str, ...] = ('png',),
		figsize: tuple[float, float] | None = None,
		dpi: int | None = None,
		axes: plt.Axes | None = None,
		style: bool = True,
	) -> None:

		if style:
			from .pyplotter import set_plt_style
			set_plt_style()

		self.outputdir = outputdir
		self.plotname = plotname
		self.suffix = suffix
		self.file_formats = list(file_formats)
		self.selection = selection
		self.output_subdir = output_subdir
		self.figure = axes.figure if axes is not None else plt.subplots(figsize=figsize, dpi=dpi)[0]
		self.axes = axes if axes is not None else self.figure.axes[0]

	@staticmethod
	def _parse_selection_dir(
		selection: str,
		outputdir: str | PathLike[str],
		subdir: str,
	) -> str:
		'''Build an output path grouped by selection and variation direction.'''
		base_selection = selection.replace('_high', '').replace('_low', '')
		direction = ('high' if '_high' in selection else
					 'low' if '_low' in selection else 'nominal')
		return f'{outputdir}/{subdir}/{base_selection}/{direction}'

	def output_dir(self) -> Path:
		'''Return the configured output directory for this plot.'''
		if self.selection is not None and self.output_subdir is not None:
			return Path(self._parse_selection_dir(
				self.selection, self.outputdir or '', self.output_subdir))
		if self.outputdir is None:
			raise ValueError('An output directory is required before saving')
		return Path(self.outputdir)


	def _normalise_spec(
		self,
		spec: Sequence[Any],
		options: dict[str, Any] | None = None,
	) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
		'''Convert a plot specification into kind, data, and Matplotlib kwargs.'''
		values = list(spec)
		if not values:
			raise ValueError('Plot specification cannot be empty')

		# Accept ([x, y, kind], kwargs) as well as [x, y, kind, kwargs].
		if len(values) == 2 and isinstance(values[1], dict):
			values, embedded = list(values[0]), values[1]
			options = {**embedded, **(options or {})}
		elif values and isinstance(values[-1], dict):
			values, embedded = values[:-1], values[-1]
			options = {**embedded, **(options or {})}
		else:
			options = dict(options or {})

		if not values or not isinstance(values[-1], str):
			raise ValueError('Plot specification must end with a plot kind')
		kind = values.pop().lower()
		if kind not in self._KINDS:
			raise ValueError(f'Unsupported plot kind: {kind}')

		expected = self._ARITY[kind]
		if len(values) != expected:
			raise ValueError(f'{kind} expects {expected} data values, got {len(values)}')
		return kind, tuple(values), options


	def draw(
		self,
		plots: Iterable[Sequence[Any]] = (),
	) -> list[Any]:
		'''Draw plot specifications and return the created Matplotlib artists.'''
		artists = []
		for item in plots:
			if isinstance(item, dict):
				kind = item.get('kind', item.get('type'))
				if kind is None:
					raise ValueError("Plot mapping requires 'kind'")
				data = item.get('data')
				if data is None:
					data = item.get('values')
				if data is None:
					raise ValueError("Plot mapping requires 'data'")
				kind, values, options = self._normalise_spec(
					[*data, kind], {k: v for k, v in item.items()
					 if k not in {'kind', 'type', 'data', 'values'}},
				)
			else:
				kind, values, options = self._normalise_spec(item)

			if kind in {'plot', 'line'}:
				artists.extend(self.axes.plot(*values, **options))
			elif kind == 'step':
				artists.extend(self.axes.step(*values, **options))
			elif kind == 'scatter':
				artists.append(self.axes.scatter(*values, **options))
			elif kind == 'bar':
				artists.append(self.axes.bar(*values, **options))
			elif kind == 'hist':
				artists.append(self.axes.hist(*values, **options))
			elif kind in {'error', 'errorbar'}:
				artists.append(self.axes.errorbar(*values, **options))
			elif kind == 'fill_between':
				artists.append(self.axes.fill_between(*values, **options))
			elif kind == 'contour':
				artists.append(self.axes.contour(*values, **options))
			elif kind == 'axvline':
				artists.append(self.axes.axvline(*values, **options))
			elif kind == 'axhline':
				artists.append(self.axes.axhline(*values, **options))
			else:
				raise ValueError(f'{kind = } is not supported, choose between [{self._KINDS}]')
		return artists


	def set_labels(
		self,
		xlabel: str = '',
		ylabel: str = '',
		left: str = '',
		right: str = '',
		locx: str = 'right',
		locy: str = 'top',
	) -> None:
		'''Set axis labels and the standard FCC-ee side labels.'''
		from .pyplotter import set_labels
		set_labels(self.axes, xlabel, ylabel, left, right, locx, locy)

	def set_title(self, title: str, **kwargs: Any) -> None:
		'''Set the centered Matplotlib title.'''
		self.axes.set_title(title, **kwargs)

	def set_axis_color(self, axis: str, color: Any) -> None:
		'''Set tick and axis-label colors for one axis.'''
		if axis not in {'x', 'y'}:
			raise ValueError("axis must be either 'x' or 'y'")
		self.axes.tick_params(axis=axis, labelcolor=color)
		getattr(self.axes, f'{axis}axis').label.set_color(color)

		return None


	def configure(
		self,
		xlim: tuple[float, float] | None = None,
		ylim: tuple[float, float] | None = None,
		xscale: str | None = None,
		yscale: str | None = None,
		grid: bool | None = None,
		legend: bool = False,
		legend_kwargs: dict[str, Any] | None = None,
	) -> None:
		'''Apply common axis, grid, and legend configuration.'''
		if xlim is not None:
			self.axes.set_xlim(*xlim)
		if ylim is not None:
			self.axes.set_ylim(*ylim)
		if xscale is not None:
			self.axes.set_xscale(xscale)
		if yscale is not None:
			self.axes.set_yscale(yscale)
		if grid is not None:
			self.axes.grid(grid)
		if legend:
			self.axes.legend(**(legend_kwargs or {}))

		return None


	def save(
		self,
		outputdir: str | PathLike[str] | None = None,
		plotname: str | None = None,
		suffix: str | None = None,
		file_formats: list[str] | tuple[str, ...] | None = None,
		close: bool = True,
		tight_layout: bool = True,
	) -> None:
		'''Lay out, save, and optionally close the figure.'''
		from .pyplotter import savefigs

		destination = Path(outputdir) if outputdir is not None else self.output_dir()
		destination.mkdir(parents=True, exist_ok=True)
		if tight_layout: self.figure.tight_layout()
		savefigs(self.figure, str(destination),
		         plotname if plotname is not None else self.plotname,
				 self.suffix if suffix is None else suffix,
				 list(file_formats) if file_formats is not None else self.file_formats)
		if close: self.close()

		return None

	def close(self) -> None:
		'''Close the owned Matplotlib figure.'''
		plt.close(self.figure)

		return None


# Short alias for callers that prefer the module name as the class name.
PyPlot = MatplotlibPlot
