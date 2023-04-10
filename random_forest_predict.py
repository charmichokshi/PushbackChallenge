from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import re

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


filename = 'finalized_model_default_settings.sav'
with open(filename, 'rb') as file:
    model = pickle.load(file)

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

for airport in airports:

    print(airport)

    airport_predictions_path = Path(f"validation_predictions_{airport}.csv.bz2")
    if airport_predictions_path.exists():
        print(f"Predictions for {airport} already exist.")
        continue

    airport_submission_format = submission_format.loc[
        submission_format.airport == airport
    ]

    test_df = pd.read_csv(DATA_DIRECTORY / f"{airport}" / f"{airport}_test_features.csv.bz2")

    test_df = test_df.fillna(test_df.mode().iloc[0])
    # test_df = test_df[
    #         ["gufi",
    #         "airport",
    #         "hour",
    #         "minute",
    #         "day",
    #         "month",
    #         "year",
    #         "etd_hour",
    #         "etd_minute",
    #         "etd_day",
    #         "etd_month",
    #         "etd_year",
    #         "minutes_until_pushback"]
    #     ]

    le = LabelEncoder()
    test_df["gufi"] = le.fit_transform(test_df["gufi"])
    test_df["airport"] = le.fit_transform(test_df["airport"])

    test_df["cloud"] = le.fit_transform(test_df["cloud"])
    # lightning_prob
    test_df["lightning_prob"] = le.fit_transform(test_df["lightning_prob"])
    # aircraft_engine_class
    test_df["aircraft_engine_class"] = le.fit_transform(test_df["aircraft_engine_class"])
    # aircraft_type major_carrier flight_type isdeparture
    test_df["aircraft_type"] = le.fit_transform(test_df["aircraft_type"])
    test_df["major_carrier"] = le.fit_transform(test_df["major_carrier"])
    test_df["flight_type"] = le.fit_transform(test_df["flight_type"])
    test_df["isdeparture"] = le.fit_transform(test_df["isdeparture"])

    test_df = test_df.drop(["minutes_until_pushback"], axis=1)

    airport_submission_format["minutes_until_pushback"] = model.predict(test_df)
    airport_submission_format["minutes_until_pushback"] = airport_submission_format["minutes_until_pushback"].astype(int)

    print("Predictions complete. Saving to disk...")
    airport_submission_format.to_csv(airport_predictions_path, index=False)
    print("Done.")

predictions = []
for airport in airports:
    airport_predictions_path = Path(f"validation_predictions_{airport}.csv.bz2")
    predictions.append(pd.read_csv(airport_predictions_path, parse_dates=["timestamp"]))

predictions = pd.concat(predictions, ignore_index=True)
predictions["minutes_until_pushback"] = predictions.minutes_until_pushback.astype(int)

predictions.to_csv("validation_predictions.zip", index=False)

