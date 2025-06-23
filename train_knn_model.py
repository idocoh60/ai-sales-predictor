
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# load clean data
df = pd.read_csv("data/clean_data.csv")

# basic cleaning 
df = df.dropna(subset=["Description"])
df = df[df["TotalPrice"] > 0]

# encoder
encoder = LabelEncoder()
df["ProductEncoded"] = encoder.fit_transform(df["Description"])

# split to x and y
features = ["Quantity", "TotalPrice", "Hour", "DayOfWeek"]
X = df[features]
y = df["ProductEncoded"]

# traing to knn model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# saveing
joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(encoder, "models/knn_label_encoder.pkl")
