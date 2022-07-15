# -----------------------------------------------------------------------------
# Calculates Disease Severity From Weather Data And Crop Specific Tuning Parameters
# -----------------------------------------------------------------------------

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .calculation_precip_score import *
from .estimate_disease_severity_day_n import *
from .estimate_disease_severity_day_one import *
from .get_rc_res import *
from .get_spray import *


def estimate_disease_severity(
    weather_df: pd.DataFrame,
    ip_t_cof: pd.DataFrame,
    p_t_cof: pd.DataFrame,
    rc_t_input: pd.DataFrame,
    dvs_8_input: pd.DataFrame,
    rc_a_input: pd.DataFrame,
    p_opt: int,
    inocp: int,
    rrlex_par: float,
    rc_opt_par: float,
    ip_opt: int,
    GDU_treshhold: int,
    is_fungicide: bool = False,
    fungicide: pd.DataFrame = pd.DataFrame(),
    fungicide_residual: pd.DataFrame = pd.DataFrame(),
    days_after_planting: int = 140,
) -> Tuple[pd.DataFrame, int]:
    """Calculates Disease Severity From Weather Data And Crop-specific Tuning Parameters.

    Args:
        weather_df (pd.DataFrame): Daily Weather Dataset For A Single Field. Necessary Columns:
            `Temperature`: Degrees C
            `precip_occur`: Boolean
            `precip`: mm
        ip_t_cof (pd.DataFrame): Crop-specific Lookup Table, Indexed On Temperature (Degrees C).
        p_t_cof (pd.DataFrame): Crop-specific Lookup Table, Indexed On Temperature (Degrees C).
        rc_t_input (pd.DataFrame): Crop-specific Lookup Table, Indexed On Temperature (Degrees C).
        dvs_8_input (pd.DataFrame): Crop Specific Lookup Table, Indexed On Cumulative GDUs.
        rc_a_input (pd.DataFrame): Crop-specific Lookup Table, Indexed On DVS 8.
        p_opt (int): _description_
        inocp (int): _description_
        rrlex_par (float): _description_
        rc_opt_par (float): _description_
        ip_opt (int): _description_
        GDU_treshhold (int): _description_
        is_fungicide (bool, optional): Whether Or Not Fungicide Was Applied. Defaults to False.
        fungicide (pd.DataFrame, optional): _description_. Defaults to pd.DataFrame(). Necessary Columns:
            `spray_number`
            `spray_moment`
            `spray_eff`
        fungicide_residual (pd.DataFrame, optional): Crop-specific Lookup Table. Defaults to pd.DataFrame().
        days_after_planting (int, optional): _description_. Defaults to 140.

    Returns:
        Tuple[pd.DataFrame, int]: Dataframe and Final Day After Planting Of Model.
    """

    # Input Variables

    weather_df = weather_df.set_index("Day", drop=True)

    total_days = len(weather_df)

    ri_series = pd.Series(np.zeros((total_days)))

    rt_count = np.array([0] * total_days)

    results_list = []

    output_columns = [
        "Temp",
        "RRDD",
        "ip",
        "p",
        "RcT",
        "Rc_W",
        "GDU",
        "GDUsum",
        "DVS8",
        "RcA",
        "Rc",
        "I",
        "RI",
        "COFR",
        "RSEN",
        "L",
        "HSEN",
        "TOTSITES",
        "RLEX",
        "R",
        "RG",
        "CumuLeak",
        "DIS",
        "Sev",
        "RAUPC",
        "RDI",
        "RDL",
        "LeakL",
        "LeakI",
        "REM",
        "Agg",
        "RRG",
        "SITEmax",
        "RRSEN",
        "inocp",
        "RRLEX",
        "RcOpt",
        "p_opt",
        "ip_opt",
        "H",
        "LAT",
        "AUDPC",
        "RT",
        "RTinc",
    ]

    for day in range(1, total_days - 2):

        if day == 1:

            results_day, ri_series = estimate_disease_severity_day_one(
                weather_df=weather_df,
                ip_t_cof=ip_t_cof,
                p_t_cof=p_t_cof,
                rc_t_input=rc_t_input,
                dvs_8_input=dvs_8_input,
                rc_a_input=rc_a_input,
                p_opt=p_opt,
                inocp=inocp,
                rrlex_par=rrlex_par,
                rc_opt_par=rc_opt_par,
                ip_opt=ip_opt,
                GDU_treshhold=GDU_treshhold,
                ri_series=ri_series,
                is_fungicide=is_fungicide,
                fungicide=fungicide,
            )

            results_list = [results_day]

        else:

            results_day = results_day.copy()

            results_day["I"] += (
                results_day["RT"] + results_day["RLEX"] - results_day["REM"] - results_day["RDI"]
            )

            results_day["L"] += ri_series.loc[day - 2] - results_day["RT"] - results_day["RDL"]

            results_day["AUDPC"] += results_day["RAUPC"]

            results_day["GDUsum"] += results_day["RTinc"]

            if day > days_after_planting:

                for col in set(output_columns) - {"I", "L", "AUDPC", "GDUsum"}:

                    results_day[col] = 0

                results_list.append(results_day)

                rt_count = rt_count[: len(results_list) + 1]

                break

            results_list, ri_series = estimate_disease_severity_day_n(
                weather_df=weather_df,
                ip_t_cof=ip_t_cof,
                p_t_cof=p_t_cof,
                rc_t_input=rc_t_input,
                dvs_8_input=dvs_8_input,
                rc_a_input=rc_a_input,
                p_opt=p_opt,
                inocp=inocp,
                ip_opt=ip_opt,
                GDU_treshhold=GDU_treshhold,
                simulation_day=day,
                results_day=results_day,
                ri_series=ri_series,
                results_list=results_list,
                rt_count=rt_count,
                is_fungicide=is_fungicide,
                fungicide=fungicide,
                fungicide_residual=fungicide_residual,
            )

    results = pd.DataFrame.from_dict(results_list)

    results["RI"] = ri_series

    return results, day
