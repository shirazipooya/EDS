# -----------------------------------------------------------------------------
# Calculates Disease Severity From Weather Data And Crop Specific Tuning Parameters
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List
import math

import numpy as np
import pandas as pd

from .calc_rain_score import *
from .get_rc_res import *
from .get_spray import *
from .eds_day_one import *
from .eds_day_n import *


def eds(
    one_field_weather: pd.DataFrame,
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
    is_fungicide: bool = False,
    fungicide: pd.DataFrame = pd.DataFrame(),
    fungicide_residual: pd.DataFrame = pd.DataFrame(),
    days_after_planting: int = 140
):
    
    """
    Calculates disease severity from weather data and crop-specific tuning parameters.

    Parameters
    ----------
    one_field_weather : pd.DataFrame
        Daily weather dataset for a single field. Necessary columns:
            `Temperature`: Degrees C
            `Rain`: Boolean
            `precip`: mm
    ip_t_cof : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    p_t_cof : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    rc_t_input : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    dvs_8_input : pd.DataFrame
        Crop specific lookup table, indexed on Cumulative GDUs
    rc_a_input : pd.DataFrame
        Crop-specific lookup table, indexed on DVS 8
    p_opt : int
    inocp : int
    rrlex_par: float
    rc_opt_par: float
    ip_opt: int
    is_fungicide : bool
        Whether or not fungicide was applied
    fungicide : pd.DataFrame
        columns: `spray_number`, `spray_moment`, `spray_eff`
    fungicide_residual : pd.DataFrame
        Crop-specific lookup table

    Returns
    -------
    results : pd.DataFrame
    day : int
        final day after planting of model
    """
    
    # Input Variables
    
    one_field_weather = one_field_weather.set_index("Day", drop=True)
    
    total_days = len(one_field_weather)
    
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
            
            results_day, ri_series = eds_day_one(
                one_field_weather = one_field_weather,
                ip_t_cof = ip_t_cof,
                p_t_cof = p_t_cof,
                rc_t_input = rc_t_input,
                dvs_8_input = dvs_8_input,
                rc_a_input = rc_a_input,
                p_opt = p_opt,
                inocp = inocp,
                rrlex_par = rrlex_par,
                rc_opt_par = rc_opt_par,
                ip_opt = ip_opt,
                ri_series = ri_series,
                is_fungicide = is_fungicide,
                fungicide = fungicide,
            )
            
            results_list = [results_day]
            
        else:
            
            results_day = results_day.copy()
            
            results_day["I"] += (
                results_day["RT"]
                + results_day["RLEX"]
                - results_day["REM"]
                - results_day["RDI"]
            )
            
            results_day["L"] += (
                ri_series.loc[day - 2] - results_day["RT"] - results_day["RDL"]
            )
            
            results_day["AUDPC"] += results_day["RAUPC"]
            
            results_day["GDUsum"] += results_day["RTinc"]
            
            if day > days_after_planting:
                
                for col in set(output_columns) - {"I", "L", "AUDPC", "GDUsum"}:
                    
                    results_day[col] = 0
                    
                results_list.append(results_day)
                
                rt_count = rt_count[: len(results_list) + 1]
                
                break
            
            results_list, ri_series = eds_day_n(
                one_field_weather = one_field_weather,
                ip_t_cof = ip_t_cof,
                p_t_cof = p_t_cof,
                rc_t_input = rc_t_input,
                dvs_8_input = dvs_8_input,
                rc_a_input= rc_a_input,
                p_opt = p_opt,
                inocp = inocp,
                ip_opt = ip_opt,
                day = day,
                results_day = results_day,
                ri_series = ri_series,
                results_list = results_list,
                rt_count = rt_count,
                is_fungicide = is_fungicide,
                fungicide = fungicide,
                fungicide_residual = fungicide_residual,
            )
    
    results = pd.DataFrame.from_dict(results_list)
    
    results["RI"] = ri_series
    
    return results, day