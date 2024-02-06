from processing import data_pipeline
from prediction import model_pipeline
from config import DATA_DIR, MODEL_DIR
from utils import get_version, increment_version


if __name__ == "__main__":
    version = get_version()

    DATA_VERSION_DIR = DATA_DIR / f"version_{version}"
    MODEL_VERSION_DIR = MODEL_DIR / f"version_{version}"

    players = data_pipeline(DATA_DIR,DATA_VERSION_DIR)
    print(players)
    model = model_pipeline(players,MODEL_VERSION_DIR)

    increment_version()

