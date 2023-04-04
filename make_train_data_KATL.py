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


def get_all_features(airport: str, exclude : List[str] = []) -> Dict[str, pd.DataFrame]:
    """Get all features for a given airport."""
    airport_features = {}
    for feature_path in DATA_DIRECTORY.glob(f"{airport}/{airport}_*.csv.bz2"):
        feature_name = re.search(rf"{airport}_(.*).csv.bz2", feature_path.name).group(1)
        if feature_name in exclude:
            continue
        print(feature_name)
        df = pd.read_csv(feature_path)
        for col in df.columns:
            if 'time' in col:
                df[col] = pd.to_datetime(df[col])
        if 'timestamp' in df.columns:
            df = df.sort_values(by='timestamp')
        airport_features[feature_name] = df
    return airport_features

airport = 'KATL'

all_features = get_all_features(airport, exclude=['tbfm', 'tfm'])

train_labels = pd.read_csv(DATA_DIRECTORY / f"train_labels_{airport}.csv.bz2", parse_dates=["timestamp"]).sort_values(by='timestamp')

def get_etd_features(now: pd.Timestamp, gufi: str, etd: pd.DataFrame) -> pd.Series:
    etd_features = {}
    for feature in ['year', 'month', 'day', 'hour', 'minute']:
        etd_features[f'etd_{feature}'] = np.nan
    now_etd = etd.loc[(etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now) & (etd.gufi == gufi)]
    last_row_index = now_etd.shape[0] - 1
    if last_row_index < 0:
        return pd.Series(etd_features, index=etd_features.keys())
    last_row = now_etd.iloc[last_row_index]
    for feature in ['year', 'month', 'day', 'hour', 'minute']:
        etd_features[f'etd_{feature}'] = getattr(last_row.departure_runway_estimated_time, feature)
    return pd.Series(etd_features, index=etd_features.keys())

def get_mfs_features(gufi: str, mfs: pd.DataFrame) -> pd.Series:
    mfs_features = {}
    for feature in ['aircraft_engine_class', 'aircraft_type', 'major_carrier', 'flight_type', 'isdeparture']:
        mfs_features[f'mfs_{feature}'] = np.nan
    now_mfs = mfs.loc[(mfs.gufi == gufi)]
    last_row_index = now_mfs.shape[0] - 1
    if last_row_index < 0:
        return pd.Series(mfs_features, index=mfs_features.keys())
    last_row = now_mfs.iloc[last_row_index]
    for feature in ['aircraft_engine_class', 'aircraft_type', 'major_carrier', 'flight_type', 'isdeparture']:
        mfs_features[f'{feature}'] = getattr(last_row, feature)
    return pd.Series(mfs_features, index=mfs_features.keys())

def get_first_position_features(gufi: str, first_position: pd.DataFrame) -> pd.Series:
    # TODO: gufi values mismatch as arrival has different gufi values than departure
    return None

def get_lamp_features(now: pd.Timestamp, lamp: pd.DataFrame) -> pd.Series:
    lamp_features = {}
    for lamp_feature in lamp.columns:
        if 'time' in lamp_feature:
            continue
        lamp_features[lamp_feature] = np.nan
    # get the row with closest timestamp to now
    if now.minute == 15:
        now = now.replace(minute=0)
    elif now.minute == 45:
        now = now.replace(minute=30)
    most_recent_forecast = lamp.loc[(lamp.timestamp == now)].sort_values(by='forecast_timestamp')
    if most_recent_forecast.shape[0] == 0:
        return pd.Series(lamp_features, index=lamp_features.keys())
    most_recent_forecast = most_recent_forecast.iloc[0]
    for lamp_feature in lamp_features.keys():
        lamp_features[lamp_feature] = most_recent_forecast[lamp_feature]
    return pd.Series(lamp_features, index=lamp_features.keys())

def make_training_data(now: pd.Timestamp, gufi: str) -> pd.Series:
    features = {}
    # merge pd.Series from different features
    now_features = {}
    now_features['year'] = now.year
    now_features['month'] = now.month
    now_features['day'] = now.day
    now_features['hour'] = now.hour
    now_features['minute'] = now.minute
    features.update(now_features)
    
    features.update(get_etd_features(now, gufi, all_features['etd']))
    features.update(get_mfs_features(gufi, all_features['mfs']))
    # features.extend(get_first_position_features(gufi, all_features['first_position']))
    features.update(get_lamp_features(now, all_features['lamp']))
    return pd.Series(features, index=features.keys())

# for each row in train_labels, we'll construct a row merging the features
# train_labels = train_labels.iloc[:50000]
train_data = process_map(
    make_training_data,
    train_labels['timestamp'],
    train_labels['gufi'],
    max_workers=200,
    chunksize=100,
)

train_data = pd.DataFrame(train_data)
train_data.to_csv(f'{airport}_train_data.csv.bz2', index=False, compression='bz2')
