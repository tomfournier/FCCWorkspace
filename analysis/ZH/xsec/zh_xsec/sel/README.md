# `sel/` Selection Utilities

Selection building blocks for the FCC-ee ZH cross-section analysis. The code operates on ROOT `RDataFrame` objects and is shared by the event-processing scripts in the analysis workflow.

## Layout

| Directory | Purpose |
| --- | --- |
| `presel/` | Build analysis variables and apply the channel preselection |
| `final/` | Define final-selection cuts and histogram configurations |

The preselection stage contains separate implementations for leptonic and hadronic channels, plus shared helpers for collection aliases and cutflow histograms. The final stage provides channel-specific selection expressions and the histogram definitions consumed by later plotting and statistical steps.

## Workflow

1. Start with an input `RDataFrame` and apply the appropriate preselection.
2. Derive the kinematic and event-level quantities needed by the analysis.
3. Apply the final selection for the chosen channel and center-of-mass energy.
4. Write the selected events, cutflow information, and histograms for the downstream MVA, measurement, or fit workflow.

Typical entry points are:

```python
from sel.presel.leptonic import presel_ll, training_ll
from sel.presel.hadronic import presel_qq, training_qq
```

Use the `training_*` functions when preparing variables for training without the full cutflow output. Use the `presel_*` functions for event processing with cutflow bookkeeping. Final-selection modules expose channel-specific cut and histogram configuration used after these variables are available.

## Conventions

- Supported analysis categories are `ee`, `mumu`, and `qq` where applicable.
- Supported center-of-mass energies are 240 and 365 GeV.
- Selection functions return an updated `RDataFrame`; preselection functions that track cutflows also return the associated ROOT objects.
- Physics expressions are evaluated by ROOT/FCCAnalyses, so the required FCC analysis environment must be initialized before importing or running the selection code.

Keep channel logic in the corresponding `presel/` or `final/` module and use the shared helpers for aliases and cutflow handling.
