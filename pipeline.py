import pandas as pd
import numpy as np

from pathlib import Path
import os

import seaborn as sns
import matplotlib.pyplot as plt

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

data_version = 1    
LOG_DIR = Path('..') / "logs"

def collect_pipeline(LOG_DIR):
    RAW_DATA_DIR = Path('..') / "data" / "raw"
    filename = list(RAW_DATA_DIR.glob('*.csv'))[-1]

    # Check if file has already been loaded
    with open(Path('..') / "logs" 'log_raw_data.txt', 'a+') as f:
        f.seek(0)
        if str(filename) in f.read():
            print(f"File {filename} has already been loaded, skipping...")
            return
        else:
            # Log filename
            f.write(str(filename) + '\n')

    print(f"Loading data from file: {filename}")
    
    players = pd.read_csv(filename, encoding='Windows-1252')

    return players

    
def preprocess_pipeling(players):    
    if players.isnull().values.any():
        print("There are missing values in the dataset, let's drop them")
        players.dropna(inplace=True,axis=0)
    
    else :
        print("There are no missing values in the dataset")
    
    # Feature engineering of relevant columns
    players["EFF"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - (players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV
    players['TS%'] = np.where((2 * (players['FGA'] + 0.44 * players['FTA'])) != 0, players['PTS'] / (2 * (players['FGA'] + 0.44 * players['FTA'])), 0)


    ages = players.Age.describe().round(decimals=1) # used to specify the first 25%, defining what is a young player
    points = players.PTS.describe().round(decimals=1)
    effs = players.EFF.describe().round(decimals=1)

    # Define the criteria for a young player to be a future superstar
    young_age = ages["25%"]
    futur_super_star_def = f"(EFF >= 12) & (PTS >= 15) & (Age <= {young_age})"

    # Deal with the position outliers
    players["position"] = players['position'].apply(lambda x: x.replace('-')[0])

    CURATED_DATA_DIR = Path('..') / "data" / "curated"

    players.to_csv(CURATED_DATA_DIR / "2023-2024.csv", index=False)







