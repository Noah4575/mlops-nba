from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from config import MODEL_DIR
from utils import folder
from joblib import dump


def create_model():
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['PTS','FG%','Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['Pos', 'Tm'])
        ])
    # Define model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    return model


# Split data into training and test sets
def split_data(players):
    X = players.drop(['Rk','Player'], axis=1)
    y = players['future_super_star']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def predict(model,players,version_dir):
    # Train model to predict PTS
    X_train, X_test, y_train, y_test = split_data(players)
    model.fit(X_train, y_train)
   
    # Predict the values for the test set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    # Save the model to a file
    dump(model, f'{version_dir}/model.joblib')

    log_predict(accuracy, precision, recall, f1, version_dir)

    return model

def log_predict(accuracy, precision, recall, f1, model_dir):

    with open(f'{model_dir}/log.txt', 'a+') as f:
        f.seek(0)
        f.write(f'Model accuracy {accuracy}\n')
        f.write(f'Model precision {precision}\n')
        f.write(f'Model recall {recall}\n')
        f.write(f'Model f1 {f1}\n')


    
def model_pipeline(players,MODEL_VERSION_DIR):
    folder(MODEL_VERSION_DIR)
    m = create_model()
    model = predict(m,players,MODEL_VERSION_DIR)
    return model
    

