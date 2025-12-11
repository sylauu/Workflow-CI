import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# 1. Argument parser 
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# 2. Load dataset
df = pd.read_csv(args.data_path)

X = df.drop(columns=["stroke"])
y = df["stroke"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Autolog
mlflow.sklearn.autolog()

# 4. MLflow run
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")

print("Training selesai dan artefak tersimpan di MLflow.")