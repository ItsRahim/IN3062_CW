from os import name
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from visualise import graph
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

df = pd.read_csv("stroke.csv")
graph(df)

# removing unneccesary columns
to_drop = ['id', 'work_type', 'Residence_type']
for columns in to_drop:
    df = df.drop(columns, axis=1)

# filling NaN values in the csv files with median from column
df = df.fillna(df.median())

# removing unknown variables from data frame which may affect results
df = df.drop(df[df.gender == 'Other'].index)
df = df.drop(df[df.smoking_status == 'Unknown'].index)

df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})
df['ever_married'] = df['ever_married'].replace({'No': 0, 'Yes': 1})
df['smoking_status'] = df['smoking_status'].replace(
    {'formerly smoked': -1, 'smokes': 1, 'never smoked': 0})
# converting df to csv to check if data is clean
df.to_csv('stroke_clean_data.csv', encoding='utf-8', index=False)

# quick summary of the data
stats = df.describe()
print(stats)

# TODO: Complete training for machine. Try Linear Regression, K Cluster, RandomForest
# TODO: Train Using all above and use most accurate one
X = df[['gender', 'age', 'hypertension',
        'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']

# splitting teh dataset into tarining and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=1/3, random_state=42)

"""
# training the algorithm
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
y_pred = linearModel.predict(X_test)

accuracyData = (pd.DataFrame(
    {'Actual': y_test, 'Predicted': y_pred}))

accuracyData.to_csv("stroke_accuracy.csv", encoding='utf-8', index=False)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
We cannot use linear regression model as we are using multiple independent variable which is not possible with linear regression. 
Since, we attend to use multiple indpenedent varibale to predict a dependent variable
"""
