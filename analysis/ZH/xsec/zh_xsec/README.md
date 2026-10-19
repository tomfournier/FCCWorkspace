# `zh_xsec/`

Shared Python utilities for the FCC-ee ZH cross-section analysis. The package
supports the MVA, measurement, combine, fit, and self-coupling workflows.

## Package layout

| Module | Role |
| --- | --- |
| `parsing.py` | Reusable command-line argument builders and script parsers |
| `userConfig.py` | Output path templates and shared analysis parameters |
| `func/` | Analysis algorithms, including BDT and fit-related functions |
| `plots/` | Plotting and visualization helpers |
| `sel/` | Pre-selection and final-selection functions |
| `tools/` | ROOT I/O, histogram, and other data-processing utilities |

The package is an internal analysis library. Analysis scripts in the numbered
workflow directories are the main entry points.

## Common usage

Set up logging and obtain the parser for the script being run:

```python
from parsing import set_log
from zh_xsec.parsing import create_parser
from logger import get_logger

parser = create_parser('2-BDT', 'evaluation')
args = parse_args(parser)
set_log(args)
logger = get_logger(__name__)
logger.info('Starting analysis')
```

Use the configuration modules for processes and paths:

```python
from pathlib import Path

from samples import get_process_dict
from zh_xsec.userConfig import loc

processes = get_process_dict(procs=['ZH', 'WW'], ecm=240)
events = loc.EVENTS.get(cat='mumu', ecm=240, type=Path)
```

`loc` provides templates for events, MVA inputs, BDT models, histograms,
plots, and statistical-fit outputs. Expand templates with `cat`, `ecm`, and,
where applicable, `sel`. Use `type=Path` for filesystem operations; the
default result is a string-like `LocPath`.

## Conventions

- Categories are generally `ee`, `mumu`, or `qq`.
- Supported center-of-mass energies are 240 and 365 GeV.
- Process and decay names follow the FCC sample naming conventions.
- Package modules should use `get_logger(__name__)`. Scripts should configure
  logging once with `set_log(args)` or `setup_logging()`.

Prefer importing existing constants, process builders, path templates, and
parser factories instead of duplicating them in analysis scripts.

## Dependencies

The FCC analysis setup provides the runtime environment. Depending on the
workflow, the package uses Python, ROOT, `uproot`, `numpy`, `pandas`, `xgboost`,
`matplotlib`, and `scipy`.

```bash
source setup/FCCAnalyses.sh
```

See the repository setup documentation for environment installation and
workflow-specific commands.
