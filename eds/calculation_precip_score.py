
# -----------------------------------------------------------------------------
# Calculates a Score (1-4) of Antecedent Precipitation Conditions
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def calculation_precip_score(
    day: int,
    precip_occur: pd.Series
) -> int:
    """Calculates a Score (1-4) of Antecedent Precipitation Conditions.

    Args:
        day (int): Days After Planting. Used as index for Precipitation Boolean Series.
        precip_occur (pd.Series): Boolean Series of Whether or not Significant (currently >= 2mm) Precipitation Was Recorded for a Day.

    Returns:
        int: Antecedent Precipitation Conditions Score.
    """

    precip_occur = precip_occur.iloc[(day - 1):(day + 3)].copy()
    four_day_sum = precip_occur.sum()
    three_day_sums = precip_occur.rolling(3).sum()
    two_day_sums = precip_occur.rolling(2).sum()

    if four_day_sum == 0:
        antecedent_precip_conditions_score = 1
    elif four_day_sum == 1:
        antecedent_precip_conditions_score = 2
    elif four_day_sum == 2:
        antecedent_precip_conditions_score = 3
    elif (two_day_sums == 2).any():
        antecedent_precip_conditions_score = 3
    elif (three_day_sums == 3).any():
        antecedent_precip_conditions_score = 3
    else:
        antecedent_precip_conditions_score = 4

    return antecedent_precip_conditions_score
