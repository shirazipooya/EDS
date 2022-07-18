
"""
Helper Functions for the Calculation Disease Severity.
"""


from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_antecedent_precipitation_conditions_score(
    days_after_planting: int,
    precipitation_occur: pd.Series
) -> int:
    """Calculates a Score (1-4) of Antecedent Precipitation Conditions.

    Args:
        days_after_planting (int): Days After Planting. Used as Index for Precipitation Boolean Series.
        precip_occur (pd.Series): Boolean Series of Whether or not Significant (Currently >= 2mm) Precipitation Was Recorded for a Day.

    Returns:
        int: Antecedent Precipitation Conditions Score.
    """

    precipitation_occur = precipitation_occur.iloc[(days_after_planting - 1):(days_after_planting + 3)].copy()
    four_day_sum = precipitation_occur.sum()
    three_day_sums = precipitation_occur.rolling(3).sum()
    two_day_sums = precipitation_occur.rolling(2).sum()

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


def calculate_fungicide_effective_residual(
    days_after_planting: int,
    fungicide: pd.DataFrame,
    residual: pd.DataFrame,
    spray_interval: int = 7
) -> float:
    """Calculate Fungicide Effective Residual.

    Args:
        days_after_planting (int): Days After Planting.
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
    fungicide["tmp"] = fungicide["spray_moment"] + spray_interval
    flag = fungicide["spray_moment"] < days_after_planting <= fungicide["tmp"]

    if flag.any():
        efficacy_residual = fungicide["spray_eff"][flag]
        fungicide_efficacy_residual = residual * efficacy_residual

    return fungicide_efficacy_residual


def calculate_flow_residual(
    days_after_planting: int,
    daily_precipitation: float,
    fungicide: pd.DataFrame,
    spray_interval: int = 7
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
    fungicide["tmp"] = fungicide["spray_moment"] + spray_interval
    flag = fungicide["spray_moment"] < days_after_planting <= fungicide["tmp"]

    if flag.any():
        flow_residual = daily_precipitation

    return flow_residual


def spray_application_parameters(
    spray_number: List[int] = [1, 2, 3],
    spray_moment: List[int] = [30, 45, 60],
    spray_efficiency: List[float] = [0.5, 0.5, 0.5],
) -> pd.DataFrame:
    """Spray Application Parameters

    Args:
        spray_number (List[int], optional): Number of Spray. Defaults to [1, 2, 3].
        spray_moment (List[int], optional): Spray Moment. Defaults to [30, 45, 60].
        spray_efficiency (List[float], optional): Spray Efficiency. Defaults to [0.5, 0.5, 0.5].

    Returns:
        pd.DataFrame: Spray Application Parameters
    """

    spray = pd.DataFrame(
        {
            "spray_number": spray_number,
            "spray_moment": spray_moment,
            "spray_efficiency": spray_efficiency
        }
    )

    return spray


def genetic_mechanistic_parameters(
    p_opt: Dict[str, float] = {
        "Susceptible": 7,
        "Moderate": 10,
        "Resistant": 14
    },
    rc_opt_par: Dict[str, float] = {
        "Susceptible": 0.35,
        "Moderate": 0.25,
        "Resistant": 0.15
    },
    rrlex_par: Dict[str, float] = {
        "Susceptible": 0.1,
        "Moderate": 0.01,
        "Resistant": 0.0001
    },
) -> Dict:
    """Genetic Mechanistic Parameters

    Args:
        p_opt (_type_, optional): _description_. Defaults to { "Susceptible": 7, "Moderate": 10, "Resistant": 14 }.
        rc_opt_par (_type_, optional): _description_. Defaults to { "Susceptible": 0.35, "Moderate": 0.25, "Resistant": 0.15 }.
        rrlex_par (_type_, optional): _description_. Defaults to { "Susceptible": 0.1, "Moderate": 0.01, "Resistant": 0.0001 }.

    Returns:
        Dict: Genetic Mechanistic Parameters.
    """

    parameters = {

        "Susceptible": {
            "p_opt": p_opt["Susceptible"],
            "rc_opt_par": rc_opt_par["Susceptible"],
            "rrlex_par": rrlex_par["Susceptible"]
        },

        "Moderate": {
            "p_opt": p_opt["Moderate"],
            "rc_opt_par": rc_opt_par["Moderate"],
            "rrlex_par": rrlex_par["Moderate"]
        },

        "Resistant": {
            "p_opt": p_opt["Resistant"],
            "rc_opt_par": rc_opt_par["Resistant"],
            "rrlex_par": rrlex_par["Resistant"]
        }

    }

    return parameters


class CropParameters():

    def __init__(
        self,
        crop_name: List[str] = None,
        crop_parameters_path: List[str] = None,
    ) -> None:

        self.crop_name = crop_name
        self.crop_parameters_path = crop_parameters_path

    def crop_parameters_constant(
        self
    ) -> Dict:

        parameters = {
            "Corn": {

                'ip_t_cof': pd.DataFrame(
                    {
                        0: [10.00, 13.00, 15.50, 17.00, 20.00, 26.00, 30.00, 35.00],
                        1: [0.00, 0.14, 0.27, 0.82, 1.00, 0.92, 0.41, 0.00]
                    }
                ),

                'p_t_cof': pd.DataFrame(
                    {
                        0: [15.00, 20.00, 25.00],
                        1: [0.60, 0.81, 1.00]
                    }
                ),

                'rc_t_input': pd.DataFrame(
                    {
                        0: [15.00, 20.00, 22.50, 24.00, 26.00, 30.00],
                        1: [0.22, 1.00, 0.44, 0.43, 0.41, 0.22]
                    }
                ),

                'dvs_8_input': pd.DataFrame(
                    {
                        0: [110.00, 200.00, 350.00, 475.00, 610.00, 740.00, 1135.00, 1660.00, 1925.00, 2320.00, 2700.00],
                        1: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00, 9.00, 10.00, 11.00]
                    }
                ),

                'rc_a_input': pd.DataFrame(
                    {
                        0: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00],
                        1: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
                    }
                ),

                'fungicide': pd.DataFrame(
                    {
                        0: [1, 2, 3],
                        1: [45, 62, 79],
                        2: [0.4, 0.4, 0.4]
                    }
                ),

                'fungicide_residual': pd.DataFrame(
                    {
                        0: [0.00, 5.00, 10.00, 15.00, 20.00],
                        1: [1.00, 0.80, 0.50, 0.25, 0.00]
                    }
                )
            },

            "Soy": {

                'ip_t_cof': pd.DataFrame(
                    {
                        0: [5.00, 11.00, 17.00, 23.00, 29.00, 35.00],
                        1: [0.00, 0.33, 0.60, 1.00, 1.00, 0.00]
                    }
                ),

                'p_t_cof': pd.DataFrame(
                    {
                        0: [0.00, 4.00, 8.00, 12.00, 16.00, 20.00, 24.00, 28.00, 32.00, 36.00, 40.00],
                        1: [0.00, 0.31, 0.39, 0.53, 0.82, 1.00, 1.00, 0.75, 0.60, 0.30, 0.00]
                    }
                ),

                'rc_t_input': pd.DataFrame(
                    {
                        0: [10.00, 12.50, 15.00, 17.50, 20.00, 23.00, 25.00, 27.50, 30.00],
                        1: [0.00, 0.57, 0.88, 1.00, 1.00, 0.86, 0.61, 0.41, 0.00]
                    }
                ),

                'dvs_8_input': pd.DataFrame(
                    {
                        0: [137.00, 366.00, 595.00, 824.00, 1053.00, 1282.00, 1511.00, 1740.00],
                        1: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
                    }
                ),

                'rc_a_input': pd.DataFrame(
                    {
                        0: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00],
                        1: [1.0000, 1.0000, 1.0000, 0.1000, 0.0001, 0.0001, 0.0001, 0.0001]
                    }
                ),

                'fungicide': pd.DataFrame(
                    {
                        0: [1, 2, 3],
                        1: [45, 62, 79],
                        2: [0.4, 0.4, 0.4]
                    }
                ),

                'fungicide_residual': pd.DataFrame(
                    {
                        0: [0.00, 5.00, 10.00, 15.00, 20.00],
                        1: [1.00, 0.80, 0.50, 0.25, 0.00]
                    }
                )
            }
        }

        return parameters

    def crop_parameters_from_file(
        self
    ) -> Dict:

        parameters = ["ip_t_cof", "p_t_cof", "rc_t_input", "dvs_8_input", "rc_a_input", "fungicide", "fungicide_residual"]

        crop_parameters = dict()

        if (self.crop_parameters_path is not None) and (self.crop_name is not None):
            if len(self.crop_name) == len(self.crop_parameters_path):
                for i in range(len(self.crop_name)):
                    crop_parameters[self.crop_name[i]] = dict()
                    for p in parameters:
                        crop_parameters[self.crop_name[i]][p] = pd.read_excel(
                            self.crop_parameters_path[i],
                            engine="openpyxl",
                            sheet_name=p,
                            header=None
                        )
        return crop_parameters
