# -----------------------------------------------------------------------------
# Read Location Data
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def field_data_preparation(
    weather_df_path: str,
    plantings_df_path: str,
    number_of_repeat_years: int = 1,
    daily_precip_threshold: float = 2,
    GDU_treshhold: Dict[str, List] = {
        "Corn": [10, 30],
        "Soy": [14, 40]
    },
) -> pd.DataFrame:
    """Data Preparation.

    Args:
        weather_df (str): Path To The Data File.
        plantings_df (str): Path To The Info File.
        number_of_repeat_years (int, optional): Number Of Repeat Data. Defaults to 1.
        daily_precip_threshold (float, optional): Daily Precipitation Threshold (mm). Defaults to 2 mm.

    Returns:
        pd.DataFrame: Location Data.
    """

    info = (
        pd.read_csv(plantings_df_path, encoding="utf-8", index_col=None)
        .groupby(["ID", "Field", "year", "planting_date", "obs_planting_delta", "Crop"])
        .size()
        .reset_index(name="count")
        .drop(columns=["count"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    info["model_origin"] = info["year"] - 1
    info["model_origin"] = info["model_origin"].astype(str) + "-12-31"

    data = pd.read_csv(weather_df_path, encoding="utf-8", index_col=None)

    data = data[data["ID"].isin(info["ID"])]
    data = data.drop_duplicates(["ID", "DOY"]).reset_index(drop=True)

    all_data = pd.DataFrame()

    for row in info.iterrows():

        df = data[data["ID"] == row[1]["ID"]]

        # df["info_id"] = row[1]["ID"] + "_" + str(row[1]["year"]) + "_" + str(row[1]["planting_date"]) + "_" + \
        #     str(row[1]["obs_planting_delta"]) + "_" + row[1]["Crop"]
        df["info_id"] = "_".join(
            (
                str(row[1]["ID"]),
                str(row[1]["year"]),
                str(row[1]["planting_date"]),
                str(row[1]["obs_planting_delta"]),
                str(row[1]["Crop"])
            )
        )

        df["info_id"].astype(str)

        df["year"] = row[1]["year"]
        df["planting_date"] = row[1]["planting_date"]
        df["Crop"] = row[1]["Crop"]
        df["obs_planting_delta"] = row[1]["obs_planting_delta"]

        model_origin = pd.to_datetime(row[1]["model_origin"])

        df["DOY"] = pd.to_timedelta(df["DOY"], unit="d")

        df["time"] = (df["DOY"] + model_origin).dt.strftime("%Y-%m-%d")

        df = (
            df.drop_duplicates(subset=["info_id", "time"])
            .reset_index(drop=True)
            .sort_values(["ID", "time"])
        )

        df_new_raw = df.copy()

        for i in range(1, number_of_repeat_years + 1):

            df_new = df_new_raw.copy()

            df_new["time"] = (
                pd.to_datetime(df_new["time"]) + pd.Timedelta(i * 365, "d")
            ).dt.strftime("%Y-%m-%d")

            df = pd.concat([df, df_new], axis=0, ignore_index=True)

            df = (
                df.drop_duplicates(subset=["info_id", "time"])
                .reset_index(drop=True)
                .sort_values(["ID", "time"])
            )

        df = df.rename(
            columns={
                "ID": "locationId",
                "time": "date",
                "precipitation": "precip",
                "maximum_temperature": "maxtemp",
                "minimum_temperature": "mintemp",
                "wind_speed": "avgwindspeed",
            }
        )

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

        df["GDU"] = np.where(
            df["Crop"] == "Corn", (df["maxtemp"] + df["mintemp"]) / 2 - GDU_treshhold['Corn'][0], np.nan
        )

        df["GDU"] = np.where(
            df["Crop"] == "Soy", (df["maxtemp"] + df["mintemp"]) / 2 - GDU_treshhold['Soy'][0], df["GDU"]
        )

        df["GDU"] = np.where(df["Crop"] == "Corn", df["GDU"].clip(GDU_treshhold['Corn'][0], GDU_treshhold['Corn'][1]), df["GDU"])

        df["GDU"] = np.where(df["Crop"] == "Soy", df["GDU"].clip(GDU_treshhold['Soy'][0], GDU_treshhold['Soy'][1]), df["GDU"])

        df = df[df["GDU"].notnull()]

        df["Temperature"] = df[["maxtemp", "mintemp"]].mean(axis=1)

        df["precip_occur"] = df["precip"] >= daily_precip_threshold  # Set precip occur as boolean

        all_data = pd.concat([all_data, df])

    return all_data
