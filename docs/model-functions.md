# Model function signatures

NLS, NLME and SAEM share one recommended signature for models written as
plain Python functions:

```python
def model(x, p):
    Time, Dose = x          # unpack exogenous values in the declared order
    ka, CL, V = p           # unpack parameters in the declared order
    k = CL / V
    return Dose * ka / (V * (ka - k)) * (np.exp(-k * Time) - np.exp(-ka * Time))
```

`x` and `p` are small read-only views. They can be:

- **unpacked directly**: `Time, Dose = x`, `ka, CL, V = p`
- **indexed by position**: `x[0]`, `p[-1]`
- **indexed by name**: `x["Time"]`, `p["CL"]`, or `x.Time`, `p.CL`
- **converted to arrays**: `np.asarray(x)`, `np.asarray(p)`
- **sliced NumPy-style**: `x[:, 0]` selects the first exogenous column

With a single predictor, `x` is used directly as the predictor values:
`np.asarray(x)` is 1-D and arithmetic applies to it. Unpacking with
`time, = x` still works for backwards compatibility. The same applies to a
single parameter: `a = p` behaves as the parameter value and `a, = p` still
works.

`len(x)` and `len(p)` are the numbers of exogenous values and parameters.

The names `x` and `p` are conventional, not required: any two argument names
are accepted for the vector form when more than one parameter is declared. With
a single parameter, a second argument named `p`, `params`, `parameters` or
`theta` also selects the vector form, so this works as expected:

```python
def growth(x, params):
    time = x
    a = params
    return a * (1.0 - np.exp(-a * time))
```

Other second-argument names keep the older named form `def f(time, a)`, where
the trailing arguments are parameters declared in `params`/`start`.

## What goes into `x` and `p`

| Engine | `x` comes from | `p` comes from |
|---|---|---|
| [NLS](nls.md) | `predictors=` (required for `f(x, p)`) | `start=` / `n_params=` |
| [NLME / GNLS](nlme.md) | `predictors=` (inferred from the first argument when it names a data column) | `params=`, or key order of `start=` when omitted |
| [SAEM](saem.md) | `predictors=[...]` when given, otherwise all data columns (time first) plus covariates | `params=`, or key order of `start=` when omitted |

Supply `predictors` to choose both the contents and the unpacking order. In
SAEM, declare the time column in `predictors` too if the model uses it, for
example `predictors=["Time", "Dose"]`. A dose record is always available as
`x.doses` without changing the unpacking order.

## Shared constructor keywords

NLS, NLME, GNLS and SAEM use the same names for equivalent data roles:

| Role | Keyword |
|---|---|
| Response column | `response=` |
| Predictor columns | `predictors=` |
| Parameter names/formulas (NLME/GNLS/SAEM) | `params=`, optional when `start=` is a mapping |

`LME` uses `fixed=` for its full `y ~ x` formula, and MCGLM uses `endog=`/`exog=`
for design arrays; those are different arguments, not alternative spellings. GLM
does not accept the shared callable form: it takes a formula or pre-built design
matrices through [`statcraft.glm`](glm.md). See the
[MCGLM manual](mcglm.md) for its matrix and formula interfaces.

When `params=` is omitted, `start=` must be a mapping: its key order defines
the order of `p` and each parameter gets the fixed formula `"1"`. Provide
`params=` explicitly whenever a parameter depends on a covariate.

## Examples

```python
from statcraft.nls import NLS
from statcraft.datasets import data_loader

muscle = data_loader("Muscle.csv").load()

def mm(x, p):
    Vm, K = p
    return Vm * x / (K + x)

fit = NLS(mm, muscle, response="length", predictors="conc",
          start={"Vm": 200.0, "K": 0.1}).fit()
```

```python
import numpy as np
import statcraft.nlme as nl

def growth(x, p):
    Asym, xmid, scal = p
    return Asym / (1 + np.exp((xmid - x) / scal))

fit = nl.nlme(growth, orange, response="circumference", predictors="age",
              random="Asym ~ 1 | Tree",
              start={"Asym": 190, "xmid": 730, "scal": 350}).fit()
```

```python
import numpy as np
import statcraft as sf

def model1cpt(x, p):
    Time, Dose = x
    ka, CL, V = p
    k = CL / V
    return Dose * ka / (V * (ka - k)) * (np.exp(-k * Time) - np.exp(-ka * Time))

fit = sf.SAEM(model1cpt, theo, id="Id", time="Time", response="Concentration",
              params=["ka", "CL", "V"], predictors=["Time", "Dose"],
              transform="log", error="constant",
              start={"ka": 1.5, "CL": 2.8, "V": 32.0},
              start_sd={"ka": 0.5, "CL": 0.5, "V": 0.5}).fit()
```

## Other supported model forms

```python
# NLS and NLME: exogenous arguments first, parameters after them
def old_style(conc, Vm, K):
    return Vm * conc / (K + conc)

# SAEM: explicit arguments, resolved by name
def explicit(ka, CL, V, Time, Dose):
    k = CL / V
    return Dose * ka / (V * (ka - k)) * (np.exp(-k * Time) - np.exp(-ka * Time))

# SAEM: SciPy-like sequences
def scipy_style(params, predictors):
    ka, CL, V = params
    Time, Dose = predictors
    return explicit(ka, CL, V, Time, Dose)
```

The shared `def model(x, p)` signature is recommended: it does not grow when
the number of parameters or exogenous columns grows.

!!! note "SAEM compilation"
    SAEM calls Python functions by keyword, so argument order is irrelevant.
    Unpacking, direct use of a single predictor, name access and indexing are
    recorded in the compiled C++ expression. `np.asarray(x)` cannot be traced
    and makes SAEM fall back to the Python callback; use `x` directly, unpack
    or index the views when you want compilation.

## See also

- Full constructor and fitting contracts: [NLS](nls.md), [NLME](nlme.md) and
  [SAEM](saem.md).
- [Data loading](data-loading.md) — the datasets used in the examples above.
- [Troubleshooting](troubleshooting.md) — callable and backend errors.
