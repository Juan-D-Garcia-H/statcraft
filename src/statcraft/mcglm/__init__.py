# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""mcglm_cpp: Multivariate Covariance Generalized Linear Models.

A C++20 engine (Eigen, OpenBLAS/LAPACKE) with nanobind bindings. The high-level entry points are
:class:`MCGLM` (specify and fit a model) and :class:`MCGLMResults` (estimates, inference, summary); the
building blocks of the covariance structure are the ``mc_*`` constructors of Z matrices.

The engine reproduces the estimation algorithms of the R package ``mcglm`` (Bonat, 2018, J. Stat. Softw. 84(4)).
"""

from .. import _mcglmfit  # noqa: F401 - ensures the compiled extension is loaded
from .._mcglmfit import (
    CovLink,
    CrossBlock,
    FitResult,
    InvalidInput,
    Link,
    Method,
    NumericalError,
    Options,
    Params,
    Problem,
    ResourceError,
    SparseZ,
    Structure,
    Variance,
    build_info,
    get_num_threads,
    link_eval,
    set_num_threads,
)
from .mcglm import MCGLM, MCGLMResults, mcglm
from .zmats import (
    mc_car,
    mc_diag,
    mc_dist,
    mc_dglm,
    mc_exp,
    mc_group,
    mc_id,
    mc_ma,
    mc_matern32,
    mc_mixed,
    mc_ns,
    mc_rw,
    mc_twin,
    mc_twin_bio,
    mc_twin_full,
    mc_wendland,
)

__all__ = [
    "MCGLM",
    "MCGLMResults",
    "mcglm",
    "Problem",
    "Options",
    "Params",
    "FitResult",
    "Link",
    "Variance",
    "CovLink",
    "Method",
    "CrossBlock",
    "Structure",
    "SparseZ",
    "NumericalError",
    "InvalidInput",
    "ResourceError",
    "set_num_threads",
    "get_num_threads",
    "build_info",
    "link_eval",
    "mc_id",
    "mc_diag",
    "mc_group",
    "mc_mixed",
    "mc_ma",
    "mc_rw",
    "mc_car",
    "mc_dist",
    "mc_exp",
    "mc_matern32",
    "mc_wendland",
    "mc_ns",
    "mc_dglm",
    "mc_twin",
    "mc_twin_bio",
    "mc_twin_full",
]
__version__ = "0.1.0"
