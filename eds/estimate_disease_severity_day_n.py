# -----------------------------------------------------------------------------
# Calculates Disease Severity From Weather Data And Crop Specific Tuning Parameters
# -----------------------------------------------------------------------------

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# from . import calculation_precip_score as cps
from .calculation_precip_score import *
from . import get_spray as gs


def estimate_disease_severity_day_n(
    weather_df: pd.DataFrame,
    ip_t_cof: pd.DataFrame,
    p_t_cof: pd.DataFrame,
    rc_t_input: pd.DataFrame,
    dvs_8_input: pd.DataFrame,
    rc_a_input: pd.DataFrame,
    p_opt: int,
    inocp: int,
    ip_opt: int,
    GDU_treshhold: int,
    simulation_day: int,
    results_day: Dict,
    ri_series: pd.Series,
    results_list: List,
    rt_count: np.array,
    is_fungicide: bool = False,
    fungicide: pd.DataFrame = pd.DataFrame(),
    fungicide_residual: pd.DataFrame = pd.DataFrame(),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Calculates Disease Severity From Weather Data And Crop-specific Tuning Parameters For Day N.

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
        ip_opt (int): _description_
        GDU_treshhold (int): _description_
        simulation_day (int): _description_
        results_day (Dict): _description_
        ri_series (pd.Series): _description_
        results_list (List): _description_
        rt_count (np.array): _description_
        is_fungicide (bool, optional): Whether Or Not Fungicide Was Applied. Defaults to False.
        fungicide (pd.DataFrame, optional): _description_. Defaults to pd.DataFrame(). Necessary Columns:
            `spray_number`
            `spray_moment`
            `spray_eff`
        fungicide_residual (pd.DataFrame, optional): _description_. Defaults to pd.DataFrame().

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Dataframe and Series.
    """
    results_day["H"] += (
        results_day["RG"]
        - ri_series.iloc[simulation_day - 2]
        - results_day["RSEN"]
        - results_day["RLEX"]
    )

    results_day["HSEN"] += results_day["RSEN"]

    results_day["LeakI"] += results_day["RDI"]

    results_day["LeakL"] += results_day["RDL"]

    results_day["R"] += results_day["REM"]

    if is_fungicide:
        results_day["ResSpray"] += results_day["FlowRes"]

    results_day["Temp"] = weather_df["Temperature"].iloc[simulation_day - 1]

    results_day["ip"] = ip_opt * np.interp(
        results_day["Temp"], ip_t_cof[0], ip_t_cof[1]
    )

    results_day["p"] = p_opt / np.interp(
        results_day["Temp"], p_t_cof[0].values, p_t_cof[1].values
    )

    results_day["CumuLeak"] = results_day["LeakL"] + results_day["LeakI"]

    results_day["DIS"] = (
        results_day["R"]
        + results_day["I"]
        + results_day["CumuLeak"]
        + results_day["L"]
    )

    results_day["TOTSITES"] = (
        results_day["HSEN"] + results_day["H"] + results_day["DIS"]
    )

    results_day["Sev"] = results_day["DIS"] / results_day["TOTSITES"]

    results_day["DVS8"] = np.interp(
        results_day["GDUsum"], dvs_8_input[0], dvs_8_input[1]
    )

    results_day["RAUPC"] = results_day["Sev"] if results_day["DVS8"] < 7 else 0

    results_day["GDU"] = results_day["Temp"] - GDU_treshhold

    results_day["RTinc"] = results_day["GDU"]

    results_day["RG"] = (
        results_day["RRG"]
        * results_day["H"]
        * (1 - (results_day["TOTSITES"] / results_day["SITEmax"]))
    )

    results_day["Rc_W"] = calculation_precip_score(
        day=simulation_day,
        precip_occur=weather_df["precip_occur"]
    )

    results_day["RcT"] = np.interp(
        results_day["Temp"], rc_t_input[0], rc_t_input[1]
    )

    results_day["RcA"] = np.interp(
        results_day["DVS8"], rc_a_input[0], rc_a_input[1]
    )

    if is_fungicide:
        try:
            results_day["FungEffcCur"] = fungicide[
                (fungicide["spray_moment"] <= simulation_day) & (simulation_day <= fungicide["V4"])
            ]["spray_eff"].iloc[0]
        except IndexError:
            results_day["FungEffcCur"] = 1

        results_day["RcFCur"] = (
            results_day["FungEffcCur"]
            if pd.notnull(results_day["FungEffcCur"])
            else 1
        )

        results_day["Residual"] = np.interp(
            results_day["ResSpray"], fungicide_residual[0], fungicide_residual[1]
        )

        results_day["RcRes"] = results_day["FungEffcCur"] * results_day["Residual"]

        results_day["FungEffecRes"] = (
            results_day["RcRes"] if pd.notnull(results_day["FungEffcCur"]) else 1
        )

        results_day["EffRes"] = results_day["FungEffcCur"]

        fung_prod = results_day["RcFCur"] * results_day["FungEffecRes"]
    else:
        fung_prod = 1

    results_day["Rc"] = (
        results_day["RcOpt"]
        * results_day["RcT"]
        * results_day["RcA"]
        * results_day["Rc_W"]
        * fung_prod
    )

    start = inocp if simulation_day > 10 else 0

    results_day["COFR"] = 1 - (
        results_day["DIS"] / (results_day["DIS"] + results_day["H"])
    )

    ri_series.iloc[simulation_day - 1] = (
        results_day["Rc"]
        * results_day["I"]
        * np.power(results_day["COFR"], results_day["Agg"])
        + start
    )

    results_day["RSEN"] = (
        results_day["RRDD"] + results_day["RRSEN"] * results_day["H"]
    )

    results_day["RLEX"] = (
        results_day["RRLEX"] * results_day["I"] * results_day["COFR"]
    )

    results_day["RDI"] = results_day["I"] * results_day["RRDD"]

    results_day["RDL"] = results_day["L"] * results_day["RRDD"]

    if simulation_day > results_day["ip"]:

        if (simulation_day - 1) == len(results_list):
            results_day["REM"] = results_list[math.ceil(simulation_day - results_day["ip"]) - 2]["RT"]
        else:
            results_day["REM"] = results_list[math.ceil(simulation_day - results_day["ip"]) - 1]["RT"]

    if is_fungicide:
        daily_precip = weather_df["precip"].iloc[simulation_day - 1]

        results_day["FlowRes"] = gs.get_spray(simulation_day, daily_precip, fungicide)

    rt_count[simulation_day - 1] = round(results_day["p"])

    results_day["RT"] = ri_series[rt_count == 0].sum()

    # Counting
    rt_count -= 1

    results_list.append(results_day)

    return results_list, ri_series
