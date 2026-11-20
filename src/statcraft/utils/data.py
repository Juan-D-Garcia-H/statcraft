# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Legacy location for dataset loading helpers.

The canonical implementation now lives in :mod:`statcraft.datasets`, together
with the dataset catalogue and loader. These aliases remain so existing
``from statcraft.utils import data_loader`` code keeps working.
"""

import numpy as np
import pandas as pd

from ..datasets import DatasetLoader, data_loader

__all__ = ["DatasetLoader", "data_loader", "add_constant"]


def add_constant(data, prepend=True, has_constant="skip"):
    """Replica exacta de statsmodels.tools.tools.add_constant"""
    if has_constant not in ["skip", "raise", "add"]:
        raise ValueError("has_constant must be 'skip', 'raise' or 'add'")

    if isinstance(data, pd.DataFrame):
        obj_type = "dataframe"
        arr = data.values
    elif isinstance(data, pd.Series):
        obj_type = "series"
        arr = data.to_frame().values
    elif isinstance(data, np.ndarray):
        obj_type = "ndarray"
        arr = np.atleast_2d(data).T if data.ndim == 1 else data.copy()
    else:
        raise TypeError("data must be a NumPy array, pandas DataFrame or Series")

    already_has_constant = False
    if arr.shape[0] > 1:
        variance = np.var(arr, axis=0)
        if np.any(np.isclose(variance, 0.0)):
            already_has_constant = True

    if already_has_constant and has_constant == "skip":
        return data
    elif already_has_constant and has_constant == "raise":
        raise ValueError("The design matrix already contains a constant column.")

    ones = np.ones((arr.shape[0], 1))

    if obj_type == "dataframe":
        res = data.copy()
        loc = 0 if prepend else len(res.columns)
        res.insert(loc, "const", 1.0)
        return res
    elif obj_type == "series":
        res = data.to_frame(name=data.name if data.name else 0)
        loc = 0 if prepend else 1
        res.insert(loc, "const", 1.0)
        return res
    else:
        X_arr = np.atleast_2d(data).T if data.ndim == 1 else data
        return np.hstack([ones, X_arr]) if prepend else np.hstack([X_arr, ones])
