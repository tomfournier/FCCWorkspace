import ROOT

from .helper import canvas_margins, configure_axis, pad_margins
from .helper import axis_limits, y_offset, setup_latex, savecanvas

from ...tools.utils import mkdir

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

cfg = None



#________________________________
def canvas(width:  int = 1000, 
           height: int = 1000, 
           top:    float = 0.055,
           bottom: float = 0.11,
           left:   float = 0.15,
           right:  float = 0.05,
           batch: bool = False,
           yields: bool = False
           ) -> ROOT.TCanvas:

    c = ROOT.TCanvas('c', 'c', width, height)
    canvas_margins(c, 
                   top=top, 
                   bottom=bottom, 
                   left=left, 
                   right=right)

    if not yields:
        if cfg['logx']: c.SetLogx()
        if cfg['logy']: c.SetLogy()
    c.SetFillStyle(4000)
    if batch: c.SetTicks(1, 1)

    c.Modify()
    c.Update()

    return c

#_______________________________________
def canvasRatio(width:  int = 1000, 
                height: int = 1000, 
                left: float = 0.15, 
                eps:  float = 0.025
                ) -> tuple[ROOT.TCanvas,
                           ROOT.TPad, 
                           ROOT.TPad]:

    c = ROOT.TCanvas('c', 'c', width, height)
    canvas_margins(c, 
                   top=0., 
                   bottom=0., 
                   left=0., 
                   right=0.)

    pad1 = ROOT.TPad('p1','p1', 0, cfg['ratiofraction'], 1, 1)
    pad2 = ROOT.TPad('p2','p2', 0, 0.0, 1, cfg['ratiofraction']-0.7*eps)

    pad_margins(pad1, top=0.055/(1.-cfg['ratiofraction']), bottom=eps,
                left=left)
    pad_margins(pad2, bottom=0.37, left=left)

    if cfg['logy']: pad1.SetLogy()
    if cfg['logx']:
        pad1.SetLogx()
        pad2.SetLogx()

    c.Modify()
    c.Update()

    return c, pad1, pad2

#_________________
def aux() -> None:

    y_off = y_offset(cfg['topRight'])
    
    latex = setup_latex(text_size=0.04, text_align=10)
    latex.DrawLatexNDC(0.15, 0.95, cfg['topLeft'])

    latex = setup_latex(text_size=0.04, text_align=30)
    latex.DrawLatex(0.95, y_off, cfg['topRight'])

#______________________
def auxRatio() -> None:

    has_sqrt = '#sqrt' in cfg['topRight']
    has_special = '^' in cfg['topRight'] or '_' in cfg['topRight']
    y_off = 0.935 if (has_sqrt and has_special) \
        else y_offset(cfg['topRight'], 0.945, 0.935)
    
    latex = setup_latex(text_size=0.06, text_align=13)
    latex.DrawLatex(0.15, 0.975, cfg['topLeft'])

    latex = setup_latex(text_size=0.055, text_align=31)
    latex.DrawLatex(0.95, y_off, cfg['topRight'])

#________________________
def dummy(nbins: int = 1
          ) -> ROOT.TH1D:

    xmin, xmax = axis_limits(cfg, 'x')
    ymin, ymax = axis_limits(cfg, 'y')

    # dummy
    dummy = ROOT.TH1D('h', 'h', 
                      nbins, 
                      xmin, xmax)

    configure_axis(dummy.GetXaxis(), 
                   cfg['xtitle'], 
                   xmin, xmax,
                   title_offset=1.2, 
                   label_offset=1.2)
    configure_axis(dummy.GetYaxis(), 
                   cfg['ytitle'], 
                   ymin, ymax,
                   title_offset=1.7, 
                   label_offset=1.4)

    dummy.SetMinimum(ymin)
    dummy.SetMaximum(ymax)
    return dummy

#_________________________________________________________
def dummyRatio(nbins: int = 1, 
               rlines: list[float] = [1], 
               colors: list[ROOT.TColor] = [ROOT.kBlack]):

    xmin, xmax   = axis_limits(cfg, 'x')
    ymin, ymax   = axis_limits(cfg, 'y')
    yminR, ymaxR = axis_limits(cfg, 'y', ratio='R')

    # dummy
    dummyT = ROOT.TH1D('h1', 'h', 
                       nbins, 
                       xmin, xmax)
    dummyB = ROOT.TH1D('h2', 'h', 
                       nbins, 
                       xmin, xmax)

    # x-axis
    configure_axis(dummyT.GetXaxis(), 
                   '', 
                   xmin, xmax,
                   title_size=0,
                   label_size=0, 
                   title_font=0,
                   label_font=0)
    configure_axis(dummyB.GetXaxis(), 
                   cfg['xtitle'], 
                   xmin, xmax,
                   title_size=32, 
                   label_size=28, 
                   title_offset=1.0,
                   label_offset=3.0)
    
    # y-axis
    configure_axis(dummyT.GetYaxis(), 
                   cfg['ytitle'], 
                   ymin, ymax,
                   title_size=32, 
                   label_size=28, 
                   title_offset=1.7, 
                   label_offset=1.4)
    configure_axis(dummyB.GetYaxis(), 
                   cfg['ytitleR'], 
                   yminR, ymaxR, 
                   title_size=32, 
                   label_size=28, 
                   title_offset=1.7, 
                   label_offset=1.4)

    dummyT.SetMaximum(ymax)
    dummyT.SetMinimum(ymin)
    dummyB.SetMinimum(yminR)
    dummyB.SetMaximum(ymaxR)
    dummyB.GetYaxis().SetNdivisions(505)
    
    lines = []
    for rline, color in zip(rlines, colors):
        line = ROOT.TLine(xmin, rline, xmax, rline)
        line.SetLineColor(color), line.SetLineWidth(2)
        lines.append(line)

    return dummyT, dummyB, lines

#_________________________________________________________
def setup_cutflow_hist(n_cuts: int, 
                       labels_map: dict[str, str], 
                       cat: str
                       ) -> tuple[ROOT.TCanvas, ROOT.TH1]:
    c = canvas()
    c.SetGrid()
    c.SetTicks()
    d = dummy(n_cuts)

    d.GetXaxis().SetLabelSize(0.75 * d.GetXaxis().GetLabelSize())
    d.GetXaxis().SetLabelOffset(1.3 * d.GetXaxis().GetLabelOffset())
    for i, cut in enumerate(labels_map):
        d.GetXaxis().SetBinLabel(i+1, labels_map[cut].replace('#ell', cat))
    d.GetXaxis().LabelsOption('u')

    return c, d

#________________________________________
def finalize_canvas(canvas: ROOT.TCanvas,
                    grid: bool = True
                    ) -> None:
    canvas.SetGrid() if grid else None
    canvas.Modify()
    canvas.Update()
    aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

#_____________________________________________
def save_canvas(canvas: ROOT.TCanvas, 
                outDir: str, 
                outName: str, 
                suffix: str = '', 
                plot_file: list[str] = ['png']
                ) -> None:
    mkdir(outDir)

    aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    savecanvas(canvas, outDir, outName, suffix, plot_file)
