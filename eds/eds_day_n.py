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


def eds_day_n(
    one_field_weather: pd.DataFrame,
    ip_t_cof: pd.DataFrame,
    p_t_cof: pd.DataFrame,
    rc_t_input: pd.DataFrame,
    dvs_8_input: pd.DataFrame,
    rc_a_input: pd.DataFrame,
    p_opt: int,
    inocp: int,
    ip_opt: int,
    day: int,
    results_day: Dict,
    ri_series: pd.Series,
    results_list: List,
    rt_count: np.array,
    is_fungicide: bool = False,
    fungicide: pd.DataFrame = pd.DataFrame(),
    fungicide_residual: pd.DataFrame = pd.DataFrame(),
):
    results_day["H"] += (
        results_day["RG"]
        - ri_series.iloc[day - 2]
        - results_day["RSEN"]
        - results_day["RLEX"]
    )
    
    results_day["HSEN"] += results_day["RSEN"]
    
    results_day["LeakI"] += results_day["RDI"]
    
    results_day["LeakL"] += results_day["RDL"]
    
    results_day["R"] += results_day["REM"]
    
    if is_fungicide:
        results_day["ResSpray"] += results_day["FlowRes"]
        
    results_day["Temp"] = one_field_weather["Temperature"].iloc[day - 1]
    
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
    
    results_day["GDU"] = results_day["Temp"] - 14
    
    results_day["RTinc"] = results_day["GDU"]
    
    results_day["RG"] = (
        results_day["RRG"]
        * results_day["H"]
        * (1 - (results_day["TOTSITES"] / results_day["SITEmax"]))
    )
    
    results_day["Rc_W"] = calc_rain_score(
        day=day, one_field_precip_bool=one_field_weather["Rain"]
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
                (fungicide["spray_moment"] <= day) & (day <= fungicide["V4"])
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
    
    start = inocp if day > 10 else 0
    
    results_day["COFR"] = 1 - (
        results_day["DIS"] / (results_day["DIS"] + results_day["H"])
    )
    
    ri_series.iloc[day - 1] = (
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
    
    if day > results_day["ip"]:
        
        if (day - 1) == len(results_list):
            results_day["REM"] = results_list[math.ceil(day - results_day["ip"]) - 2]["RT"]
        else:
            results_day["REM"] = results_list[math.ceil(day - results_day["ip"]) - 1]["RT"]
    
    if is_fungicide:
        daily_rainfall = one_field_weather["precip"].iloc[day - 1]
        
        results_day["FlowRes"] = get_spray(day, daily_rainfall, fungicide)
        
    rt_count[day - 1] = round(results_day["p"])
    
    results_day["RT"] = ri_series[rt_count == 0].sum()
    
    # Counting
    rt_count -= 1
    
    results_list.append(results_day)
    
    return results_list, ri_series