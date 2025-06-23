import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

def train_regressor(input_path="../data/processed/training_data.csv",
                    model_path="../models/linear_regressor_model.pkl"):
    
    #   load the tarining data
    df = pd.read_csv(input_path)

    # Filter only customers who actually made a purchase
    df = df[df["Label"] == 1]

    # Defining features and x, y
    features = ["NumPurchases", "TotalSpent", "AverageSpent", "DaysSinceLastPurchase"]
    X = df[features]
    y = df["TotalPrice"]

    # data normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #  split x, y to test and train
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    #  model creation
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict and mse+r2
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(" ביצועי מודל רגרסיה:")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}")

    # saving 
    joblib.dump(model, model_path)
    joblib.dump(scaler, "../models/regressor_scaler.pkl")
    print(f" המודל נשמר בהצלחה: {model_path}")

    return model


if __name__ == "__main__":
    train_regressor()


