# -----------------------------------------------------------------------------
# Calculate Disease Severity
# -----------------------------------------------------------------------------

import itertools
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import estimate_disease_severity as eds
from . import location_data as ld


def calculate_disease_severity(
    weather_df: str,
    plantings_df: str,
    parameters: Dict,
    number_of_repeat_years: int = 1,
    daily_precip_threshold: float = 2,
    number_applications_list: List[int] = [0, 1, 2, 3],
    genetic_mechanistic_list: List[str] = ["Susceptible", "Moderate", "Resistant"],
    user_crop_parameters: bool = False,
    output_columns: List[str] = ["Sev50%", "SevMAX", "AUC"],
    path_result=None
):
    """Calculates Disease Severity.

    Args:
        weather_df (str): Path To The Data File.
        plantings_df (str): Path To The Info File.
        parameters (Dict): _description_.
        number_of_repeat_years (int, optional): Number Of Repeat Data. Defaults to 1.
        daily_precip_threshold (float, optional): Daily Precipitation Threshold (mm). Defaults to 2 mm.
        number_applications_list (List[int], optional): _description_. Defaults to [0, 1, 2, 3].
        genetic_mechanistic_list (List[str], optional): _description_. Defaults to ["Susceptible", "Moderate", "Resistant"].
        user_crop_parameters (bool, optional): _description_. Defaults to False.
        output_columns (List[str], optional): _description_. Defaults to ["Sev50%", "SevMAX", "AUC"].
        path_result (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    all_results = []

    spray_para = parameters.spray()
    genetic_mechanistic_para = parameters.genetic_mechanistic()

    data = ld.location_data(
        weather_df=weather_df,
        plantings_df=plantings_df,
        number_of_repeat_years=number_of_repeat_years,
        daily_precip_threshold=daily_precip_threshold
    )

    if user_crop_parameters:
        crop_para = parameters.corp_from_file()
    else:
        crop_para = parameters.crop()

    for id in data["info_id"].unique():
        df = data[data["info_id"] == id]
        planting_date_list = [pd.to_datetime(dt).strftime(
            "%Y%m%d") for dt in df["planting_date"].unique()]

        for number_applications, genetic_mechanistic, date in itertools.product(number_applications_list, genetic_mechanistic_list, planting_date_list):

            if number_applications > 0:
                using_fungicide = True
                fungicide_inputs = spray_para[spray_para["spray_number"]
                                              <= number_applications]
            else:
                using_fungicide = False
                fungicide_inputs = pd.DataFrame()

            crop_para_selected = crop_para[df["Crop"].unique()[0]]

            df = df[df["date"] >= date].copy()

            if len(df) == 0:
                print(f"Location {id} NOT USED. No dates in range.")
                continue

            df["Day"] = (
                df["DOY"] - df["DOY"].iloc[0]
            ).dt.days + 1

            field_results, n_day = eds.estimate_disease_severity(
                weather_df=df,
                ip_t_cof=crop_para_selected["ip_t_cof"],
                p_t_cof=crop_para_selected["p_t_cof"],
                rc_t_input=crop_para_selected["rc_t_input"],
                dvs_8_input=crop_para_selected["dvs_8_input"],
                rc_a_input=crop_para_selected["rc_a_input"],
                p_opt=genetic_mechanistic_para[genetic_mechanistic]["p_opt"],
                rrlex_par=genetic_mechanistic_para[genetic_mechanistic]["rrlex_par"],
                rc_opt_par=genetic_mechanistic_para[genetic_mechanistic]["rc_opt_par"],
                inocp=10,
                ip_opt=14 if df["Crop"].unique()[0] == "Corn" else 28,
                GDU_treshhold=10 if df["Crop"].unique()[0] == "Corn" else 14,
                is_fungicide=using_fungicide,
                fungicide=fungicide_inputs,
                fungicide_residual=crop_para_selected["fungicide_residual"],
                days_after_planting=df["obs_planting_delta"].unique()[0],
            )

            # Output information
            start_date = df["date"].iloc[0]
            end_date = df["date"].iloc[n_day - 1]
            result_location = {}
            result_location["locationId"] = id
            result_location["Date1"] = start_date
            result_location["Date2"] = end_date
            result_location["N_Days"] = (
                pd.to_datetime(end_date) - pd.to_datetime(start_date)
            ).days
            result_location["latitude"] = df["latitude"].iloc[0]
            result_location["longitude"] = df["longitude"].iloc[0]
            result_location["Sev50%"] = field_results["Sev"].median()
            result_location["SevMAX"] = field_results["Sev"].max()
            nonzero_sev = field_results[field_results["Sev"] != 0]["Sev"]

            if len(nonzero_sev):
                result_location["AUC"] = np.trapz(nonzero_sev)
            else:
                result_location["AUC"] = 0

            result_location["number_applications"] = number_applications
            result_location["genetic_mechanistic"] = genetic_mechanistic
            result_location["crop"] = df["Crop"].unique()[0]

            all_results.append(result_location)

    all_results = pd.DataFrame.from_dict(all_results)

    all_results.sort_values(
        by=["locationId", "crop", "Date1",
            "number_applications", "genetic_mechanistic"],
        inplace=True
    )

    all_results = all_results[["locationId", "Date1", "Date2", "N_Days", "latitude",
                               "longitude", "number_applications", "genetic_mechanistic", "crop"] + output_columns]

    return all_results
