import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


X, y = load_iris(return_X_y = True, as_frame = True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 2024)

scaler = StandardScaler()
classifier = SVC()


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_valid = scaler.transform(X_valid)
classifier.fit(scaled_X_train, y_train)

train_pred = classifier.predict(scaled_X_train)
valid_pred = classifier.predict(scaled_X_valid)


train_acc = accuracy_score(y_true = y_train, y_pred = train_pred)
valid_acc = accuracy_score(y_true = y_valid, y_pred = valid_pred)


print('Train accuracy: ', train_acc)
print('valid accuracy: ', valid_acc)


joblib.dump(scaler, 'scaler.joblib')
joblib.dump(classifier, 'classifier.joblib')