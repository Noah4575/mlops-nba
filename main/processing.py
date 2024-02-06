import pandas as pd
import os
import datetime
import shutil
from utils import folder

def collect_pipeline(data_dir, version_dir):
    """
    Collects the latest CSV file from the data directory, checks if it has already been loaded,
    and if not, loads the data into a pandas DataFrame. Then moves the original file to the version directory.
    """
    try:
        filename = list(data_dir.glob('*.csv'))[-1]
    except IndexError:
        print("No CSV files found in the data directory.")
        return pd.DataFrame(), None

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    players = pd.DataFrame()

    with open(f'{data_dir}/log.txt', 'a+') as f:
        f.seek(0)
        if str(filename) in f.read():
            print(f"File {filename} has already been loaded, skipping...")
        else:
            f.write(f'Loaded file {filename} at {timestamp}\n')
            print(f"Loading data from file: {filename}")
            try:
                players = pd.read_csv(filename, encoding='Windows-1252')
            except Exception as e:
                print(f"Error reading CSV file: {e}")

    shutil.copy(filename, version_dir / "raw" / filename.name)
    os.remove(filename)

    return players, filename

def preprocess_pipeling(players, filename, version_dir):
    """
    Preprocesses the player data. Drops any rows with missing values, calculates a new 'efficiency' feature,
    identifies potential future superstars based on certain criteria, deals with position outliers,
    and saves the processed data to a new CSV file in the version directory.
    """
    if players.empty:
        return players

    if players.isnull().values.any():
        print("There are missing values in the dataset, let's drop them")
        players.dropna(inplace=True, axis=0)
    else:
        print("There are no missing values in the dataset")

    players["efficiency"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - (players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV

    ages = players.Age.describe().round(decimals=1)
    young_age = ages["25%"]

    players["future_super_star"] = (players["efficiency"] >= 12) & (players["PTS"] >= 10) & (players["Age"] <= young_age)
    players["future_super_star"] = players["future_super_star"].astype(int)

    players["Pos"] = players['Pos'].apply(lambda x: x.split('-')[0] if '-' in x else x)

    name = filename.name.split(' ')[0]
    players.to_csv(version_dir / "curated" / name, index=False)

    return players

def data_pipeline(DATA_DIR,DATA_VERSION_DIR):
    """
    Calls the other functions in order to execute the entire pipeline.
    """
    folder(DATA_VERSION_DIR)
    players, filename = collect_pipeline(DATA_DIR, DATA_VERSION_DIR)
    players = preprocess_pipeling(players, filename, DATA_VERSION_DIR)
    return players
