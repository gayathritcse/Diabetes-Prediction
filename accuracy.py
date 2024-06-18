import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:/Users/aravi/Desktop/diabetes.csv')

print(data.head())

print(data.isnull().sum())

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)
nb = GaussianNB()

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
nb.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
nb_pred = nb.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
nb_accuracy = accuracy_score(y_test, nb_pred)

print(f'Random Forest Accuracy: {rf_accuracy}')
print(f'SVM Accuracy: {svm_accuracy}')
print(f'Naive Bayes Accuracy: {nb_accuracy}')
