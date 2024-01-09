from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# iris 데이터 불러오기
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 저장된 모델 불러오기
model_pipeline_load = joblib.load("model_pipeline.joblib")

# 검증
load_train_pred = model_pipeline_load.predict(X_train)
load_valid_pred = model_pipeline_load.predict(X_valid)
load_train_acc = accuracy_score(y_true=y_train, y_pred=load_train_pred)
load_valid_acc = accuracy_score(y_true=y_valid, y_pred=load_valid_pred)
print("Load Model Train Accuracy :", load_train_acc) # 0.9833333333333333
print("Load Model Valid Accuracy :", load_valid_acc) # 0.9666666666666667