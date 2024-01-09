from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# DB 연결
df = pd.read_csv("data.csv")
X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 저장된 모델 불러오기
pipeline_load = joblib.load("db_pipeline.joblib")

# 검증
load_train_pred = pipeline_load.predict(X_train)
load_valid_pred = pipeline_load.predict(X_valid)
load_train_acc = accuracy_score(y_true=y_train, y_pred=load_train_pred)
load_valid_acc = accuracy_score(y_true=y_valid, y_pred=load_valid_pred)
print("Load Model Train Accuracy :", load_train_acc)
print("Load Model Valid Accuracy :", load_valid_acc)