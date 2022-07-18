"""
    Disease Severity Estimate From Weather Data And Crop Specific Tuning Parameters.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from . import utils


def estimate_disease_severity_day_one(
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
    ri_series: pd.Series,
    is_fungicide: bool = False,
    fungicide: pd.DataFrame = pd.DataFrame(),
) -> Tuple[pd.DataFrame, pd.Series]:
    """E stimate Disease Severity From Weather Data And Crop-specific Tuning Parameters For Day One.

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
        ri_series (pd.Series): _description_
        is_fungicide (bool, optional): Whether Or Not Fungicide Was Applied. Defaults to False.
        fungicide (pd.DataFrame, optional): _description_. Defaults to pd.DataFrame(). Necessary Columns:
            `spray_number`
            `spray_moment`
            `spray_eff`

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Dataframe and Series.
    """

    # Input variables
    simulation_day = 1

    # First Day initialization.
    results_day: Dict[str, Union[float, int]] = dict()
    results_day["p_opt"] = p_opt
    results_day["ip_opt"] = ip_opt
    results_day["Temp"] = weather_df["Temperature"].iloc[0]
    results_day["ip"] = ip_opt * np.interp(
        results_day["Temp"], ip_t_cof[0], ip_t_cof[1]
    )
    results_day["RRDD"] = 0.0001
    results_day["I"] = results_day["ip"]
    results_day["p"] = p_opt / np.interp(results_day["Temp"], p_t_cof[0], p_t_cof[1])
    results_day["L"] = 0
    results_day["AUDPC"] = 0
    results_day["GDUsum"] = 0
    results_day["H"] = 2500000
    results_day["HSEN"] = 0
    results_day["LeakI"] = 0
    results_day["LeakL"] = 0
    results_day["R"] = 0
    results_day["LAT"] = 0

    # add ResSpray variables
    if is_fungicide:
        results_day["ResSpray"] = 0

    results_day["CumuLeak"] = results_day["LeakL"] + results_day["LeakI"]

    results_day["DIS"] = (
        results_day["R"] + results_day["I"] + results_day["CumuLeak"] + results_day["L"]
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
    results_day["SITEmax"] = 10000000
    results_day["RRG"] = 0.173
    results_day["RG"] = (
        results_day["RRG"]
        * results_day["H"]
        * (1 - (results_day["TOTSITES"] / results_day["SITEmax"]))
    )
    results_day["Rc_W"] = utils.calculate_antecedent_precipitation_conditions_score(
        days_after_planting=simulation_day,
        precipitation_occur=weather_df["precip_occur"]
    )
    results_day["RRLEX"] = rrlex_par
    results_day["inocp"] = inocp
    results_day["RcOpt"] = rc_opt_par - results_day["RRLEX"]
    results_day["RcT"] = np.interp(results_day["Temp"], rc_t_input[0], rc_t_input[1])
    results_day["RcA"] = np.interp(results_day["DVS8"], rc_a_input[0], rc_a_input[1])

    if is_fungicide:
        results_day["Residual"] = results_day["ResSpray"]
        results_day["RcFCur"] = 1
        results_day["FungEffcCur"] = 1
        results_day["EffRes"] = 1
        results_day["FungEffecRes"] = utils.calculate_fungicide_effective_residual(
            days_after_planting=simulation_day,
            fungicide=fungicide,
            residual=results_day["Residual"],
            spray_interval=7
        )
        results_day["RcRes"] = 1
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

    results_day["COFR"] = 1 - (results_day["DIS"] / (2 * results_day["H"]))

    results_day["Agg"] = 1

    ri_series.iloc[simulation_day - 1] = (
        results_day["Rc"]
        * results_day["I"]
        * np.power(results_day["COFR"], results_day["Agg"])
        + 1
    )

    results_day["RRSEN"] = 0.002307
    results_day["RSEN"] = results_day["RRDD"] + results_day["RRSEN"] * results_day["H"]
    results_day["RLEX"] = results_day["RRLEX"] * results_day["I"] * results_day["COFR"]
    results_day["RDI"] = 0.000100004821661
    results_day["REM"] = 0.999948211789

    if is_fungicide:
        daily_precip = weather_df["precip"].iloc[simulation_day - 1]
        results_day["FlowRes"] = utils.calculate_flow_residual(
            days_after_planting=simulation_day,
            daily_precipitation=daily_precip,
            fungicide=fungicide,
            spray_interval=7
        )

    results_day["RT"] = 0
    results_day["RDL"] = 0

    return results_day, ri_series


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
    """Estimate Disease Severity From Weather Data And Crop-specific Tuning Parameters For Day N.

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

    results_day["Rc_W"] = utils.calculate_antecedent_precipitation_conditions_score(
        days_after_planting=simulation_day,
        precipitation_occur=weather_df["precip_occur"]
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

        results_day["FlowRes"] = utils.calculate_flow_residual(
            days_after_planting=simulation_day,
            daily_precipitation=daily_precip,
            fungicide=fungicide,
            spray_interval=7
        )

    rt_count[simulation_day - 1] = round(results_day["p"])

    results_day["RT"] = ri_series[rt_count == 0].sum()

    # Counting
    rt_count -= 1

    results_list.append(results_day)

    return results_list, ri_series


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
    """Disease Severity Estimate From Weather Data And Crop-specific Tuning Parameters.

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
