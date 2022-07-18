# -----------------------------------------------------------------------------
# Calculate Disease Severity
# -----------------------------------------------------------------------------

import itertools
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from .estimate_disease_severity import estimate_disease_severity
from .field_data_preparation import field_data_preparation


def calculation_crop_disease_severity(
    weather_df_path: str,
    plantings_df_path: str,
    crop_parameters: Dict,
    spray_parameters: pd.DataFrame,
    genetic_mechanistic_parameters: Dict,
    number_of_repeat_years: int = 1,
    daily_precip_threshold: float = 2,
    number_applications_list: List[int] = [0, 1, 2, 3],
    genetic_mechanistic_list: List[str] = ["Susceptible", "Moderate", "Resistant"],
    output_columns: List[str] = ["Sev50%", "SevMAX", "AUC"]
):

    all_results = []

    data = field_data_preparation(
        weather_df_path=weather_df_path,
        plantings_df_path=plantings_df_path,
        number_of_repeat_years=number_of_repeat_years,
        daily_precip_threshold=daily_precip_threshold
    )

    for id in data["info_id"].unique():
        df = data[data["info_id"] == id]
        planting_date_list = [pd.to_datetime(dt).strftime(
            "%Y%m%d") for dt in df["planting_date"].unique()]

        for number_applications, genetic_mechanistic, date in itertools.product(number_applications_list, genetic_mechanistic_list, planting_date_list):

            if number_applications > 0:
                using_fungicide = True
                fungicide_inputs = spray_parameters[spray_parameters["spray_number"]
                                                    <= number_applications]
            else:
                using_fungicide = False
                fungicide_inputs = pd.DataFrame()

            crop_parameters_selected = crop_parameters[df["Crop"].unique()[0]]

            df = df[df["date"] >= date].copy()

            if len(df) == 0:
                print(f"Location {id} NOT USED. No dates in range.")
                continue

            df["Day"] = (
                df["DOY"] - df["DOY"].iloc[0]
            ).dt.days + 1

            field_results, n_day = estimate_disease_severity(
                weather_df=df,
                ip_t_cof=crop_parameters_selected["ip_t_cof"],
                p_t_cof=crop_parameters_selected["p_t_cof"],
                rc_t_input=crop_parameters_selected["rc_t_input"],
                dvs_8_input=crop_parameters_selected["dvs_8_input"],
                rc_a_input=crop_parameters_selected["rc_a_input"],
                p_opt=genetic_mechanistic_parameters[genetic_mechanistic]["p_opt"],
                rrlex_par=genetic_mechanistic_parameters[genetic_mechanistic]["rrlex_par"],
                rc_opt_par=genetic_mechanistic_parameters[genetic_mechanistic]["rc_opt_par"],
                inocp=10,
                ip_opt=14 if df["Crop"].unique()[0] == "Corn" else 28,
                GDU_treshhold=10 if df["Crop"].unique()[0] == "Corn" else 14,
                is_fungicide=using_fungicide,
                fungicide=fungicide_inputs,
                fungicide_residual=crop_parameters_selected["fungicide_residual"],
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
