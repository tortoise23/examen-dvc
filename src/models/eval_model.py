import pandas as pd
import joblib
import json
from sklearn.metrics import r2_score, mean_squared_error

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()

model = joblib.load("models/best_model.joblib")

predictions = model.predict(X_test)

pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
}).to_csv("data/predictions.csv", index=False)


r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

with open("metrics/scores.json", "w") as file_json:
    json.dump({"r2": r2, "mse": mse}, file_json)