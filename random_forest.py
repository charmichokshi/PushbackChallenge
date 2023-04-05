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
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle


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

    df = pd.read_csv(DATA_DIRECTORY / f"{airport}" / f"{airport}_features.csv.bz2")


    le = LabelEncoder()
    df['gufi'] = le.fit_transform(df['gufi'])
    df['airport'] = le.fit_transform(df['airport'])
    df['cloud'] = le.fit_transform(df['cloud'])
    df['lightning_prob'] = le.fit_transform(df['lightning_prob'])
    df['aircraft_engine_class'] = le.fit_transform(df['aircraft_engine_class'])
    df['aircraft_type'] = le.fit_transform(df['aircraft_type'])
    df['major_carrier'] = le.fit_transform(df['major_carrier'])
    df['flight_type'] = le.fit_transform(df['flight_type'])
    df['isdeparture'] = le.fit_transform(df['isdeparture'])



    y = df['minutes_until_pushback']

    df = df.drop(['minutes_until_pushback'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.10, random_state = 5)

    Rfc = RandomForestRegressor(random_state=2, max_depth=9, n_estimators=7, warm_start=True, max_samples=0.8)

    print('Training model...')
    fitResultR = Rfc.fit(X_train, y_train)
    predictedValues = fitResultR.predict(X_test)
    mae = mean_absolute_error(y_test, predictedValues)
    print ('MAE:' ,  mae)
