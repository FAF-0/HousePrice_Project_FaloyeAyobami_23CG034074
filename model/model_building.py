import pandas as pd
import numpy as np
import requests
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
DATA_URL = "https://raw.githubusercontent.com/eddiexunyc/DATA605_FINAL/main/Resources/train.csv"
DATA_PATH = "train.csv" # Relative to project root
MODEL_PATH = "model/house_price_model.pkl" # Save inside model dir
SELECTED_FEATURES = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt', 'SalePrice']

def download_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset to {os.path.abspath(DATA_PATH)}...")
        sys.stdout.flush()
        try:
            response = requests.get(DATA_URL, timeout=30)
            if response.status_code == 200:
                with open(DATA_PATH, 'wb') as f:
                    f.write(response.content)
                print("Dataset downloaded successfully.")
            else:
                print(f"Failed to download dataset. Status code: {response.status_code}")
                raise Exception(f"Failed to download dataset.")
        except Exception as e:
            print(f"Error downloading: {e}")
            raise
    else:
        print(f"Dataset already exists at {DATA_PATH}.")

def train_model():
    print("Loading data...")
    sys.stdout.flush()
    if not os.path.exists(DATA_PATH):
        print("Data file missing!")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Feature Selection
    print(f"Selecting features: {SELECTED_FEATURES}")
    df = df[SELECTED_FEATURES]
    
    # Preprocessing
    print("Preprocessing...")
    # Handling missing values - filling with median
    df = df.fillna(df.median())
    
    # Splitting Data
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    print("Training Random Forest Regressor...")
    sys.stdout.flush()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    
    # Save Model
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved.")

if __name__ == "__main__":
    download_data()
    train_model()
