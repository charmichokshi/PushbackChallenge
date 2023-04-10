from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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

def estimate_pushback(now: pd.Timestamp) -> pd.Series:

    # subset submission format to the current prediction time
    now_submission_format = airport_submission_format.loc[
        airport_submission_format.timestamp == now
    ].reset_index(drop=True)

    # filter features to 30 hours before prediction time to prediction time
    now_etd = etd.loc[(etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now)]

    # get the latest ETD for each flight
    latest_now_etd = now_etd.groupby("gufi").last().departure_runway_estimated_time

    # merge the latest ETD with the flights we are predicting
    departure_runway_estimated_time = now_submission_format.merge(
        latest_now_etd, how="left", on="gufi"
    ).departure_runway_estimated_time

    now_prediction = now_submission_format.copy()

    now_prediction["minutes_until_pushback"] = (
        (departure_runway_estimated_time - now_submission_format.timestamp).dt.total_seconds() / 60
    ) - 15

    return now_prediction

for airport in airports:
    print(f"Processing {airport}")
    airport_predictions_path = Path(f"validation_predictions_{airport}.csv.bz2")
    if airport_predictions_path.exists():
        print(f"Predictions for {airport} already exist.")
        continue

    # subset submission format to current airport
    airport_submission_format = submission_format.loc[
        submission_format.airport == airport
    ]

    # load airport's ETD data and sort by timestamp
    etd = pd.read_csv(
        DATA_DIRECTORY / airport / f"{airport}_etd.csv.bz2",
        parse_dates=["departure_runway_estimated_time", "timestamp"],
    ).sort_values("timestamp")

    # process all prediction times in parallel
    predictions = process_map(
        estimate_pushback,
        pd.to_datetime(airport_submission_format.timestamp.unique()),
        chunksize=50,
    )

    # concatenate individual prediction times to a single dataframe
    predictions = pd.concat(predictions, ignore_index=True)
    
    predictions["minutes_until_pushback"] = predictions.minutes_until_pushback.clip(
        lower=0
    ).astype(int)

    # reindex the predictions to match the expected ordering in the submission format
    predictions = (
        predictions.set_index(["gufi", "timestamp", "airport"])
        .loc[
            airport_submission_format.set_index(["gufi", "timestamp", "airport"]).index
        ]
        .reset_index()
    )

    # save the predictions for the current airport
    predictions.to_csv(airport_predictions_path, index=False)

predictions = []

for airport in airports:
    airport_predictions_path = Path(f"validation_predictions_{airport}.csv.bz2")
    predictions.append(pd.read_csv(airport_predictions_path, parse_dates=["timestamp"]))

predictions = pd.concat(predictions, ignore_index=True)
predictions["minutes_until_pushback"] = predictions.minutes_until_pushback.astype(int)

predictions.to_csv("validation_predictions.zip", index=False)
