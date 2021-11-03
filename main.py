from os import name
import numpy as np
import pandas as pd

df = pd.read_csv("stroke.csv")

# removing unneccesary columns
to_drop = ['id', 'ever_married', 'work_type', 'Residence_type']
for columns in to_drop:
    df = df.drop(columns, axis=1)

# filling NaN values in the csv files with median from column
df = df.fillna(df.median())

# removing unknown variables from data frame which may affect results
df = df.drop(df[df.smoking_status == 'Unknown'].index)

# converting df to csv to check if data is clean
df.to_csv('stroke_new.csv', encoding='utf-8', index=False)
