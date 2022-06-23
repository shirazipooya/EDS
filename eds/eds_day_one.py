# -----------------------------------------------------------------------------
# Calculates Disease Severity From Weather Data And Crop Specific Tuning Parameters For Day One
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List
import math

import numpy as np
import pandas as pd

from .calc_rain_score import *
from .get_rc_res import *
from .get_spray import *


def eds_day_one(
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
    ri_series: pd.Series,
    is_fungicide: bool = False,
    fungicide: pd.DataFrame = pd.DataFrame(),
):
    
    """
    Calculates disease severity from weather data and crop-specific tuning parameters for Day One.

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

    Returns
    -------
    results : pd.DataFrame
    day : int
        final day after planting of model
    """
    # Input variables
    day = 1
    
    # First Day initialization.
    results_day: Dict[str, Union[float, int]] = dict()
    results_day["p_opt"] = p_opt
    results_day["ip_opt"] = ip_opt
    results_day["Temp"] = one_field_weather["Temperature"].iloc[0]
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
    results_day["GDU"] = results_day["Temp"] - 14
    results_day["RTinc"] = results_day["GDU"]
    results_day["SITEmax"] = 10000000
    results_day["RRG"] = 0.173
    results_day["RG"] = (
        results_day["RRG"]
        * results_day["H"]
        * (1 - (results_day["TOTSITES"] / results_day["SITEmax"]))
    )
    results_day["Rc_W"] = calc_rain_score(
        day=day, one_field_precip_bool=one_field_weather["Rain"]
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
        results_day["FungEffecRes"] = get_rc_res(
            day, fungicide, results_day["Residual"]
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
    
    ri_series.iloc[day - 1] = (
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
        daily_rainfall = one_field_weather["precip"].iloc[day - 1]
        results_day["FlowRes"] = get_spray(day, daily_rainfall, fungicide)

    results_day["RT"] = 0
    results_day["RDL"] = 0
    
    return results_day, ri_series