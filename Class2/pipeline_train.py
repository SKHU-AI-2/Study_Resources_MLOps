from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# iris 데이터 불러오기
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 모델 개발 및 학습
model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())]) # 파이프라인
model_pipeline.fit(X_train, y_train)
train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)
train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)
print("Train Accuracy :", train_acc) # 0.9833333333333333
print("Valid Accuracy :", valid_acc) # 0.9666666666666667

# 학습된 모델 저장
joblib.dump(model_pipeline, "model_pipeline.joblib")