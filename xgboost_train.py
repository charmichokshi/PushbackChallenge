from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import re

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

import xgboost as xgb


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

all_dfs = []
for airport in airports:
    print(airport)

    df = pd.read_csv(DATA_DIRECTORY / f"{airport}" / f"{airport}_features.csv.bz2")

    # fill in missing values with most frequent value in that column
    df = df.fillna(df.mode().iloc[0])

    #  keep only  specific columns
    # df = df[
    #     ["gufi",
    #     "airport",
    #     "hour",
    #     "minute",
    #     "day",
    #     "month",
    #     "year",
    #     "etd_hour",
    #     "etd_minute",
    #     "etd_day",
    #     "etd_month",
    #     "etd_year",
    #     "minutes_until_pushback"]
    # ]

    print("Preprocessing data...")
    le = LabelEncoder()
    df["gufi"] = le.fit_transform(df["gufi"])
    df["airport"] = le.fit_transform(df["airport"])

    df["cloud"] = le.fit_transform(df["cloud"])
    df["lightning_prob"] = le.fit_transform(df["lightning_prob"])
    df["aircraft_engine_class"] = le.fit_transform(df["aircraft_engine_class"])
    df["aircraft_type"] = le.fit_transform(df["aircraft_type"])
    df["major_carrier"] = le.fit_transform(df["major_carrier"])
    df["flight_type"] = le.fit_transform(df["flight_type"])
    df["isdeparture"] = le.fit_transform(df["isdeparture"])

    all_dfs.append(df)

all_dfs = pd.concat(all_dfs)

y = all_dfs["minutes_until_pushback"]

all_dfs = all_dfs.drop(["minutes_until_pushback"], axis=1)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    all_dfs, y, test_size=0.10, random_state=5
)

# Rfc = RandomForestRegressor(
#     random_state=42,
#     max_depth=100,
#     n_estimators=10,
#     min_samples_split=5,
#     max_features="sqrt",
# )

# # set min_impurity_split to 1 to avoid warning
# Rfc.set_params(min_impurity_split=1)

xg_boost = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    max_depth=7,
    eta=0.1,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
)


print("Training model...")
fitResultR = xg_boost.fit(X_train, y_train)
predictedValues = fitResultR.predict(X_test)
mae = mean_absolute_error(y_test, predictedValues)
print("MAE:", mae)

filename = "finalized_model_xgboost_default_settings.sav"
pickle.dump(xg_boost, open(filename, "wb"))
