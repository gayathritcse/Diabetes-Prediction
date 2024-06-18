from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def homepage(request):
    return render(request,"homepage.html")
def predict(request):

    glucose = float(request.GET["Glucose"])
    bmi = float(request.GET["BMI"])
    age = float(request.GET["Age"])
    bloodpressure = float(request.GET["Blood Pressure"])
    insulin = float(request.GET["Insulin"])
    skinthickness = float(request.GET["Skin Thickness"])
    diabetespedigreefunction = float(request.GET["Diabetes Pedigree Function"])
    pregnancies = float(request.GET["Pregnancies"])

    df = pd.read_csv('diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = SVC(kernel='linear', C=1, random_state=42)
    svm.fit(X_train, y_train)
    result1 = int(svm.predict([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]]))
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    result2 = int(nb.predict([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]]))
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    result3 = int(rf.predict([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]]))
    

        
    if result1==1:
        final_result1 = "You may have diabetes"
    else:
        final_result1 = "You may not have diabetes"
    if result2==1:
        final_result2 = "You may have diabetes"
    else:
        final_result2 = "You may not have diabetes"
    if result3==1:
        final_result3 = "You may have diabetes"
    else:
        final_result3 = "You may not have diabetes"
        
    return render(request,"predict.html",{'Print1':final_result1,'Print2':final_result2,'Print3':final_result3})
