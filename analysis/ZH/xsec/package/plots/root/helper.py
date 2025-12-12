import os, ROOT

from functools import lru_cache
from typing import Union

from ...config import warning
from ...tools.utils import mkdir, get_range
from ...tools.process import getHist

#____________________________________________________
def make_cfg(cfg: dict[str, 
                       Union[str, float, int, bool]], 
             ecm: int = 240, 
             lumi: float = 10.8,
             ratio_plot: bool = False
             ) -> dict[str, Union[str, float, int, None]]:

    # x-y range
    if ('xmin' not in cfg) or ('xmax' not in cfg) \
        or ('ymin' not in cfg) or ('ymax' not in cfg):
        msg = 'Histogram limits not set. Aborting code'
        warning(msg)

    # x-y scale
    cfg.setdefault('logx', False)
    cfg.setdefault('logy', False)
    
    # title
    cfg.setdefault('xtitle', '')
    cfg.setdefault('ytitle', 'Events')
    cfg.setdefault('topLeft', '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}')
    cfg.setdefault('topRight', f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{-1}}')

    
    # ratio variables
    if (('ymin' not in cfg) or ('ymax' not in cfg)) and ratio_plot:
        msg = 'Ratio limits of the histogram not set. Aborting code'
        warning(msg)
    cfg.setdefault('ytitleR', 'Ratio')
    cfg.setdefault('ratiofraction', 0.3)
    
    return cfg

#_____________________________________________
def build_cfg(hist: ROOT.TH1, 
              logX: bool = False, 
              logY: bool = False,
              xmin: Union[float, None] = None,
              xmax: Union[float, None] = None,
              ymin: Union[float, None] = None, 
              ymax: Union[float, None] = None,
              xtitle: str = '',
              ytitle: str = 'Events',
              ecm: int = 240, 
              lumi: float = 10.8,
              strict: bool = True, 
              stack: bool = False,
              hists: Union[list, None] = None,
              range_func: callable = get_range,
              cutflow: bool = False,
              decay: bool = False
              ) -> dict:
    scale_min, scale_max = 5e-1 if logY else 1, 1e4 if logY else 1.5
    if not cutflow:
        if not decay:
            xMin, xMax, yMin, yMax = range_func([hist], 
                                                hists, 
                                                logY=logY, 
                                                stack=stack, 
                                                strict=strict,
                                                xmin=xmin, xmax=xmax, 
                                                ymin=ymin, ymax=ymax,
                                                scale_min=scale_min, 
                                                scale_max=scale_max)
        else:
            xMin, xMax, yMin, yMax = range_func(hists, 
                                                logY=logY,  
                                                strict=strict,
                                                xmin=xmin, xmax=xmax, 
                                                ymin=ymin, ymax=ymax,
                                                scale_min=scale_min, 
                                                scale_max=scale_max)
    else:
        if (xmin is None) or (xmax is None) or \
            (ymin is None) or (ymax is None):
            warning('Range was not set, aborting...')
        xMin, xMax, yMin, yMax = xmin, xmax, ymin, ymax
    if xtitle=='':
        xTitle = hist.GetXaxis().GetTitle()
    elif xtitle=='None':
        xTitle = ''
    else: 
        xTitle = xtitle
    return make_cfg({
        'xmin': xMin, 'xmax': xMax, 
        'ymin': yMin, 'ymax': yMax,
        'logx': logX, 'logy': logY,
        'xtitle': xTitle, 
        'ytitle': ytitle,
    }, ecm=ecm, lumi=lumi)

#_____________________________________________________
def canvas_margins(c: ROOT.TCanvas, 
                   top:    Union[float, None] = 0.055, 
                   bottom: Union[float, None] = 0.11,
                   left:   Union[float, None] = 0.15, 
                   right:  Union[float, None] = 0.05
                   ) -> None:
    if top is not None:
        c.SetTopMargin(top)
    if bottom is not None:
        c.SetBottomMargin(bottom)
    if left is not None:
        c.SetLeftMargin(left)
    if right is not None:
        c.SetRightMargin(right)

#____________________________________
def pad_margins(pad: ROOT.TPad, 
                top:    float = 0.0, 
                bottom: float = 0.0,
                left:   float = 0.15,
                right:  float = 0.05
                ) -> None:
    pad.SetTopMargin(top)
    pad.SetBottomMargin(bottom)
    pad.SetLeftMargin(left)
    pad.SetRightMargin(right)

#______________________________________
def mk_legend(num_entries: int, 
              columns: int = 1, 
              x1: float = 0.55, 
              y1: float = 0.99, 
              x2: float = 0.99, 
              y2: float = 0.90,
              border_size: int = 0,
              fill_style:  int = 0,
              text_size:  float = 0.03,
              set_margin: float = 0.2,
              text_font: int = -1
              ) -> ROOT.TLegend:
    leg = ROOT.TLegend(x1, y1 - (num_entries) \
                       * 0.06 * (1/columns), 
                       x2, y2)
    if text_font!=-1:
        leg.SetTextFont(text_font)
    leg.SetBorderSize(border_size)
    leg.SetFillStyle(fill_style)
    leg.SetTextSize(text_size)
    leg.SetMargin(set_margin)
    leg.SetNColumns(columns)
    return leg

@lru_cache(maxsize=128)
def _get_hist_cached(hName: str,
                     procs: tuple,
                     inDir: str,
                     suffix: str,
                     rebin: int,
                     lazy: bool) -> ROOT.TH1:
    return getHist(hName, 
                   list(procs), 
                   inDir, 
                   suffix=suffix, 
                   rebin=rebin, 
                   lazy=lazy)

#_________________________________________
def load_hists(processes: dict[str, 
                               list[str]], 
               variable: str, 
               inDir: str, 
               suffix: str, 
               rebin: int = 1, 
               lazy: bool = True
               ) -> dict[str, ROOT.TH1]:
    return {proc: _get_hist_cached(variable, 
                                   tuple(proc_list), 
                                   inDir, 
                                   suffix=suffix, 
                                   rebin=rebin, 
                                   lazy=lazy)
            for proc, proc_list in processes.items()}

#_______________________________________________________
def axis_limits(cfg: dict[str, 
                          Union[str, float, int, bool]], 
                axis: str, 
                ratio: str = ''
                ) -> tuple[float, float]:
    is_log = cfg[f'log{axis}']
    min = float(cfg[f'{axis}min{ratio}'])
    max = float(cfg[f'{axis}max{ratio}'])
    
    if is_log:
        return 0.999 * min, 1.001 * max
    return min, max

#____________________________________________
def configure_axis(axis, 
                   title: str, 
                   axis_min:     float, 
                   axis_max:     float,
                   title_size:   int = 40, 
                   label_size:   int = 35, 
                   title_offset: float = 1.2, 
                   label_offset: float = 1.2, 
                   title_font:   int = 43,
                   label_font:   int = 43
                   ) -> None:
    if title:
        axis.SetTitle(title)
    axis.SetRangeUser(axis_min, axis_max)
    axis.SetTitleSize(title_size)
    axis.SetLabelSize(label_size)
    axis.SetTitleFont(title_font)
    axis.SetLabelFont(label_font)
    axis.SetTitleOffset(title_offset * axis.GetTitleOffset())
    axis.SetLabelOffset(label_offset * axis.GetLabelOffset())

#__________________________________________________
def style_hist(hist: ROOT.TH1, 
               color: int, 
               width: int = 1, 
               style: int = 1, 
               scale: float = 1., 
               fill_color: Union[int, None] = None
               ) -> None:
    hist.SetLineColor(color)
    hist.SetLineWidth(width)
    hist.SetLineStyle(style)
    if fill_color is not None:
        hist.SetFillColor(fill_color)
    if scale != 1.:
        hist.Scale(scale)

#_______________________________________________________
def setup_latex(text_size: float, 
                text_align: int, 
                text_color: Union[int, ROOT.TColor] = 1,
                text_font: int = 42) -> ROOT.TLatex:
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(text_size)
    latex.SetTextColor(text_color)
    latex.SetTextFont(text_font)
    latex.SetTextAlign(text_align)
    return latex

#________________________________
def y_offset(text: str, 
             high: float = 0.955, 
             low:  float = 0.945
             ) -> float:
    has_underscore = '_' in text
    has_caret = '^' in text
    return low if (has_underscore or has_caret) else high

#__________________________________________________
def draw_latex(latex: ROOT.TLatex, 
               text_data: list[tuple[str, float, 
                                     float, float]]
                                     ) -> None:
    for text, x, y, size in text_data:
        latex.SetTextSize(size)
        latex.DrawLatex(x, y, text)

#_________________________________________
def savecanvas(c: ROOT.TCanvas, 
               outDir: str, 
               plotname: str,
               suffix: str = '', 
               format: list[str] = ['png']
               ) -> None:
    fpath = os.path.join(outDir, plotname+suffix)
    for f in format:
        c.SaveAs(f'{fpath}.{f}')

#__________________________________
def save_plot(canvas: ROOT.TCanvas, 
              outDir: str, 
              outName: str, 
              suffix: str,
              format: list[str], 
              ) -> None:
    mkdir(outDir)
    savecanvas(canvas, outDir, 
               outName, 
               suffix=suffix,
               format=format)
