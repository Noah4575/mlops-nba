import pandas as pd
import numpy as np

from pathlib import Path
import os
import datetime
import shutil

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

from potential_stars.extract import create_nba_features

data_version = 1
DATA_DIR = Path('..')/ "mlops-nba" / "data" 
VERSION_DIR = DATA_DIR / f"version_{data_version}" 

def folder(version_dir):
    # Create data folder if it doesn't exist
    if not version_dir.exists():
        os.makedirs(version_dir / "raw")
        os.makedirs(version_dir / "curated")

        #create a new log file in data_dir
        log = os.path.join(version_dir, "log.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log, 'w') as f:
            f.write(f"Log file created : {timestamp} \n")
        print('Data folder created')
    else:
        print(f"Folder {version_dir} already exists, skipping...")

folder(VERSION_DIR)


def collect_pipeline(data_dir,version_dir):

    filename = list(data_dir.glob('*.csv'))[-1]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if file has already been loaded
    with open(f'{data_dir}/log.txt', 'a+') as f:
        f.seek(0)
        if str(filename) in f.read():
            print(f"File {filename} has already been loaded, skipping...")
        else:
            # Log filename
            f.write(f'Loaded file {filename} at {timestamp}\n')
            print(f"Loading data from file: {filename}")
            players = pd.read_csv(filename, encoding='Windows-1252')
    
    # Rename and move data file to version_dir
    shutil.copy(filename, version_dir / "raw" / filename.name)
    os.remove(filename)

    return players,filename

players,filename = collect_pipeline(DATA_DIR,VERSION_DIR)



def preprocess_pipeling(players,filename,version_dir):    
    if players.isnull().values.any():
        print("There are missing values in the dataset, let's drop them")
        players.dropna(inplace=True,axis=0)
    
    else :
        print("There are no missing values in the dataset")
    
    # Feature engineering of relevant columns
    players = create_nba_features(players=players)
    
    # Define the criteria for a young player to be a future superstar
    ages = players.Age.describe().round(decimals=1) # used to specify the first 25%, defining what is a young player
    young_age = ages["25%"]
    
    futur_super_star_def = f"(EFF >= 12) & (PTS >= 10) & (Age <= {young_age})"

    #Create pred column
    players["future_super_star"] = players.eval(futur_super_star_def)

    # Deal with the position outliers
    players["position"] = players['position'].apply(lambda x: x.replace('-')[0])

    #Move file to curated folder
    players.to_csv(version_dir / "curated" / filename.name, index=False)
    os.remove(filename)

    return players  

players = preprocess_pipeling(players,filename,VERSION_DIR)