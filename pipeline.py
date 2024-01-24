import pandas as pd
import numpy as np

from pathlib import Path
import os
import datetime

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
DATA_DIR = Path('..') / "data" / f"version_{data_version}" 

def folder(DATA_DIR):
    # Create data folder if it doesn't exist
    if not DATA_DIR.exists():
        os.makedirs(DATA_DIR)
        os.makedirs(DATA_DIR / "raw")
        os.makedirs(DATA_DIR / "curated")
        #create a new log file in data_dir
        log = os.path.join(DATA_DIR, "log.txt")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log, 'w') as f:
            f.write("Log file created : {timestamp} \n")
    else:
        print(f"Folder {DATA_DIR} already exists, skipping...")
    return

folder(DATA_DIR)
"""
def collect_pipeline(DATA_DIR):

    filename = list(DATA_DIR.glob('*.csv'))[-1]
    log = 

    # Check if file has already been loaded
    with open(f'{LOG_DIR}log_raw_data.txt', 'a+') as f:
        f.seek(0)
        if str(filename) in f.read():
            print(f"File {filename} has already been loaded, skipping...")
            return
        else:
            # Log filename
            f.write(str(filename) + '\n')

    print(f"Loading data from file: {filename}")
    
    players = pd.read_csv(filename, encoding='Windows-1252')

    return players,filename

    
def preprocess_pipeling(players,filename,LOG_DIR):    
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

    CUR_DATA_DIR = Path('..') / "data" / "curated"
    filename = list(CUR_DATA_DIR.glob('*.csv'))[-1]

    with open(f'{LOG_DIR}log_raw_data.txt', 'a+') as f:
    f.seek(0)
    if str(filename) in f.read():
        print(f"File {filename} has already been loaded, skipping...")
        return
    else:
        # Log filename
        f.write(str(filename) + '\n')

    players.to_csv(CURATED_DATA_DIR / "2023-2024.csv", index=False)







"""