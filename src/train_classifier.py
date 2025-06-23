import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def train_classifier(input_path="../data/processed/training_data.csv",
                     model_path="../models/logistic_model.pkl"):
    
    # load tarinig data
    df = pd.read_csv(input_path)

    # define target label
    y = df["Label"]

    # select relevant columns
    features = ["NumPurchases", "TotalSpent", "AverageSpent", "DaysSinceLastPurchase"]
    X = df[features]

    # data normalization 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # model creation
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)

    # y prediction and report
    y_pred = model.predict(X_test)
    print(" דו״ח המודל:\n")
    print(classification_report(y_test, y_pred))

    # saving to folder
    joblib.dump(model, model_path)
    joblib.dump(scaler, "../models/scaler.pkl")
    print(f" המודל נשמר ב: {model_path}")
    return model


#  test
if __name__ == "__main__":
    train_classifier()


