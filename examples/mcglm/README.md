# statcraft.mcglm examples

Runnable examples for multivariate covariance GLMs (`statcraft.mcglm`).
Each script loads a bundled dataset, specifies the response formulas, the
`z` matrix list and the link/variance functions, fits the model and
prints the summary.

Run any example from the repository root:

```bash
python examples/mcglm/soya.py
```

| Example | Description |
| --- | --- |
| `soya.py` | Three associated responses (Gaussian, Tweedie, binomial proportion) |
| `ahs.py` | Five health-care utilisation counts with Poisson-Tweedie variances |
| `soil.py` | Four soil chemistry responses over spatial coordinates |
| `sleepstudy.py` | Random intercept and slope via `mc_mixed` |
| `hunting.py` | Bivariate counts with a random hunter effect and an offset |
| `newborn.py` | Longitudinal responses with moving-average dependence via `mc_ma` |

All scripts resolve `statcraft/data/` relative to their own location, so
they work from any working directory as long as the repository layout is
intact.

## Diagnostic plots

`fit.plot()` draws one Pearson-residual panel per response plus the observed
and model-implied between-response correlation heatmaps. Individual figures
are available through `fit.plot_pearson()`, `fit.plot_correlation()` and
`fit.plot_diagnostics()`; `fit.plot(kind=...)` remains supported. Matplotlib
is optional:

```bash
pip install "statcraft[viz]"
python examples/mcglm/diagnostics.py
```

The `soya.py` fit is reused in `diagnostics.py`, which writes the per-response
residual panels and the two correlation heatmaps.
