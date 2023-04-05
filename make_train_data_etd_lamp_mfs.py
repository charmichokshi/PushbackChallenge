from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import re

airports = [
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA",
]

DATA_DIRECTORY = Path("data")

submission_format = pd.read_csv(
    DATA_DIRECTORY / "submission_format.csv", parse_dates=["timestamp"]
)

from typing import Dict, List


def get_all_features(airport: str, exclude: List[str] = []) -> Dict[str, pd.DataFrame]:
    """Get all features for a given airport."""
    airport_features = {}
    for feature_path in DATA_DIRECTORY.glob(f"{airport}/{airport}_*.csv.bz2"):
        feature_name = re.search(rf"{airport}_(.*).csv.bz2", feature_path.name).group(1)
        if feature_name in exclude:
            continue
        print(feature_name)
        df = pd.read_csv(feature_path)
        for col in df.columns:
            if "time" in col:
                df[col] = pd.to_datetime(df[col])
        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp")
        airport_features[feature_name] = df
    return airport_features


for airport in airports:

    all_features = get_all_features(airport, exclude=["tbfm", "tfm"])

    train_labels = pd.read_csv(
        DATA_DIRECTORY / f"train_labels_{airport}.csv.bz2", parse_dates=["timestamp"]
    ).sort_values(by="timestamp")

    def get_etd_features(now: pd.Timestamp) -> pd.DataFrame:
        etd = all_features["etd"]
        # for feature in ['year', 'month', 'day', 'hour', 'minute']:
        #     etd_features[f'etd_{feature}'] = np.nan
        now_etd = etd.loc[
            (etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now)
        ]
        latest_now_etd = now_etd.groupby("gufi").last()
        latest_now_etd["gufi"] = latest_now_etd.index
        latest_now_etd["timestamp"] = now

        return latest_now_etd

    def get_lamp_features(now: pd.Timestamp) -> pd.DataFrame:
        lamp = all_features["lamp"]
        lamp_features = {"timestamp": now}
        for lamp_feature in lamp.columns:
            if "time" in lamp_feature:
                continue
            lamp_features[lamp_feature] = np.nan
        # get the row with closest timestamp to now
        most_recent_forecast = lamp.loc[
            (lamp.timestamp <= now) & (lamp.timestamp > now - timedelta(minutes=60))
        ].sort_values(by="forecast_timestamp")
        assert (
            len(most_recent_forecast.timestamp.unique()) <= 1
        ), f"More than one timestamp for {now}"
        if most_recent_forecast.shape[0] == 0:
            return pd.DataFrame(lamp_features, index=[0])
        most_recent_forecast = most_recent_forecast.iloc[0]
        for lamp_feature in lamp_features.keys():
            if lamp_feature == "timestamp":
                continue
            lamp_features[lamp_feature] = most_recent_forecast[lamp_feature]
        return pd.DataFrame(lamp_features, index=[0])

    print("getting etd features")
    etd_dataframes = process_map(
        get_etd_features,
        pd.to_datetime(train_labels.timestamp.unique()),
        max_workers=128,
        chunksize=20,
    )
    etd_dataframe = pd.concat(etd_dataframes, ignore_index=True)

    print("getting lamp features")
    lamp_series = process_map(
        get_lamp_features,
        pd.to_datetime(train_labels.timestamp.unique()),
        max_workers=64,
        chunksize=20,
    )
    lamp_dataframe = pd.concat(lamp_series, ignore_index=True)

    final_dataframe = etd_dataframe.join(
        lamp_dataframe.set_index("timestamp"),
        on="timestamp",
    )

    mfs = all_features["mfs"]
    final_dataframe = final_dataframe.join(
        mfs.set_index("gufi"),
        on="gufi",
    )

    final_dataframe = final_dataframe.join(
        train_labels.set_index(["timestamp", "gufi"]),
        on=["timestamp", "gufi"],
    )
    final_dataframe = final_dataframe.loc[
        ~final_dataframe.minutes_until_pushback.isnull()
    ]

    final_dataframe["hour"] = final_dataframe["timestamp"].dt.hour
    final_dataframe["minute"] = final_dataframe["timestamp"].dt.minute
    final_dataframe["day"] = final_dataframe["timestamp"].dt.day
    final_dataframe["month"] = final_dataframe["timestamp"].dt.month
    final_dataframe["year"] = final_dataframe["timestamp"].dt.year

    final_dataframe["etd_hour"] = final_dataframe[
        "departure_runway_estimated_time"
    ].dt.hour
    final_dataframe["etd_minute"] = final_dataframe[
        "departure_runway_estimated_time"
    ].dt.minute
    final_dataframe["etd_day"] = final_dataframe[
        "departure_runway_estimated_time"
    ].dt.day
    final_dataframe["etd_month"] = final_dataframe[
        "departure_runway_estimated_time"
    ].dt.month
    final_dataframe["etd_year"] = final_dataframe[
        "departure_runway_estimated_time"
    ].dt.year

    del final_dataframe["departure_runway_estimated_time"]
    del final_dataframe["timestamp"]

    final_dataframe.to_csv(
        f"data/{airport}/{airport}_features.csv.bz2", index=False, compression="bz2"
    )
