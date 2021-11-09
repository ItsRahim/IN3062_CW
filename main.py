from os import name
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("stroke.csv")

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

# converting df to csv to check if data is clean
df.to_csv('stroke_clean_data.csv', encoding='utf-8', index=False)

# quick summary of the data
stats = df.describe()
print(stats)

"""
strokePatient = pd.DataFrame(df.loc[df['stroke'] == 1])
non_strokePatient = pd.DataFrame(df.loc[df['stroke'] == 0])

strokePatient.to_csv("_strokePatient.csv", encoding='utf-8', index=False)
non_strokePatient.to_csv("_non_strokePatient.csv",
                         encoding='utf-8', index=False)
"""
# TODO: Complete training for machine. Try Linear Regression, K Cluster, RandomForest
# TODO: Train Using all above and use most accurate one
X = df[['gender', 'age', 'hypertension',
        'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.33, random_state=42)
