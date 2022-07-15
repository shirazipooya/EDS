# -----------------------------------------------------------------------------
# Calculate Fungicide Effective Residual
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_rc_res(
    day: int,
    fungicide: pd.DataFrame,
    residual: pd.DataFrame,
    spraying_time_interval: int = 7
) -> float:
    """Calculate Fungicide Effective Residual.

    Args:
        day (int): Days Since Planting.
        fungicide (pd.DataFrame): Necessary Columns:
            `spray_number`
            `spray_moment`
            `spray_eff`
        residual (pd.DataFrame): Crop-specific Lookup Table.
        spraying_time_interval (int): Day Interval for Spraying.

    Returns:
        float: Fungicide Effective Residual.
    """

    fungicide_efficacy_residual = 1.0

    fungicide["V4"] = fungicide["spray_moment"] + spraying_time_interval

    # flag = (day > fungicide["spray_moment"]) & (day <= fungicide["V4"])
    flag = fungicide["spray_moment"] < day <= fungicide["V4"]

    if flag.any():
        eff_res = fungicide["spray_eff"][flag]
        fungicide_efficacy_residual = residual * eff_res

    return fungicide_efficacy_residual
