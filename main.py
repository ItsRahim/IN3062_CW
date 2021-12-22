import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# commented for now
#from imblearn.over_sampling import SMOTE
#from imblearn import over_sampling


df = pd.read_csv("stroke.csv")
x = pd.DataFrame(df.groupby(['stroke'])['stroke'].count())
print(x)

# removing unneccesary columns
to_drop = ['id', 'work_type']
for columns in to_drop:
    df = df.drop(columns, axis=1)

# filling NaN values in the csv files with median from column
df['bmi'].fillna(value=df['bmi'].median(), inplace=True)

# removing unknown variables from data frame which may affect results
df = df.drop(df[df.gender == 'Other'].index)
df = df.drop(df[df.smoking_status == 'Unknown'].index)

df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})
df['ever_married'] = df['ever_married'].replace({'No': 0, 'Yes': 1})
df['smoking_status'] = df['smoking_status'].replace(
    {'formerly smoked': -1, 'smokes': 1, 'never smoked': 0})

# Adding residence type allowe dthe accuracy to be increase fractionally.
df['Residence_type'] = df['Residence_type'].replace({'Urban': 1, 'Rural': 0})

# Testing if accuracy increased by adding in work_type
"""
df['work_type'] = df['work_type'].replace({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children' : 1})
Adding work_type did not increase the accuracy as it stayed the same.
"""

# converting df to csv to check if data is clean
df.to_csv('stroke_clean_data.csv', encoding='utf-8', index=False)

# quick summary of the data
stats = df.describe()
print(stats)

# TODO: Smote to balance dataset - Rahim
X = df[['gender', 'age', 'hypertension',
        'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']

# splitting the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=1/3, random_state=42)


def randomForestClassifier(X_train, X_test, y_train, y_test):
    print("Random Forest Classifier")
    clf = RandomForestClassifier(n_estimators=100, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rfc = confusion_matrix(y_test, y_pred)
    print(rfc)
    print(classification_report(y_test, y_pred))


def decisionTree(X_train, X_test, y_train, y_test):
    print("Decision Tree Classifier")
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    dt = confusion_matrix(y_test, y_pred)
    print(dt)
    print(classification_report(y_test, y_pred))


def logisticRegression(X_train, X_test, y_train, y_test):
    print("Logistic Regression")
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(classification_report(y_test, y_pred))


# TODO: Gaussian instead of convolutional neural network to train AI - Jeeves
"""
# training the algorithm
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
y_pred = linearModel.predict(X_test)

accuracyData = (pd.DataFrame(
    {'Actual': y_test, 'Predicted': y_pred}))

accuracyData.to_csv("stroke_accuracy.csv", encoding='utf-8', index=False)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

We cannot use linear regression model as we are using multiple independent variable
which is not possible with linear regression. 
Since, we attend to use multiple indpenedent varibale to predict a dependent variable
"""
randomForestClassifier(X_train, X_test, y_train, y_test)
decisionTree(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
