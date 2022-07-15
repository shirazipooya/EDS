# -----------------------------------------------------------------------------
# Calculate Flow Residual
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_spray(
    day: int,
    daily_precipitation: float,
    fungicide: pd.DataFrame,
    spraying_time_interval: int = 7
) -> float:
    """Calculate Flow Residual

    Args:
        day (int): Days Since Planting.
        daily_precipitation (float): Daily Precipitation in mm
        fungicide (pd.DataFrame): Necessary Columns:
            `spray_number`
            `spray_moment`
            `spray_eff`
        spraying_time_interval (int): Day Interval for Spraying.

    Returns:
        float: Flow Residual.
    """

    flow_residual = 0.0

    fungicide["V4"] = fungicide["spray_moment"] + spraying_time_interval

    flag = (day > fungicide["spray_moment"]) & (day <= fungicide["V4"])

    if flag.any():
        flow_residual = daily_precipitation

    return flow_residual
