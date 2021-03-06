import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay

"""
Below are functions created which take in test and training data as paramenters from the train test split.
These functions are used to:
-create and train machine learning models
-create a report regarding accuracy of the models
-visualise the predictions made by the model via the confusion matrix
"""


def randomForestClassifier(X_train, X_test, y_train, y_test):
    print("Random Forest Classifier\n")

    # creating the machine learning model
    clf = RandomForestClassifier(n_estimators=1500, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rfc = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(rfc)
    print(classification_report(y_test, y_pred))

    # creating confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    disp.ax_.set_title("Random Forest Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()


def decisionTree(X_train, X_test, y_train, y_test):
    print("Decision Tree Classifier\n")
    decisionTreeModel = DecisionTreeClassifier(criterion='gini')
    decisionTreeModel.fit(X_train, y_train)
    y_pred = decisionTreeModel.predict(X_test)
    dt = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(dt)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=decisionTreeModel.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=decisionTreeModel.classes_)
    disp.plot()
    disp.ax_.set_title("Decision Tree Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()


def logisticRegression(X_train, X_test, y_train, y_test):
    print("Logistic Regression\n")
    logreg = LogisticRegression(fit_intercept=True)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(cnf_matrix)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=logreg.classes_)
    disp.plot()
    disp.ax_.set_title("Logistic Regression")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()


def linearRegression(X_train, X_test, y_train, y_test):
    print("Linear Regression\n")
    linearModel = LinearRegression(fit_intercept=True)
    linearModel.fit(X_train, y_train)
    y_pred = linearModel.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix\n")
    print(cnf_matrix)
    print(classification_report(y_test, y_pred))


def naiveBayes(X_train, X_test, y_train, y_test):
    print("Naive Bayes\n")
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    y_pred = gaussian.predict(X_test)
    gaussian_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix\n")
    print(gaussian_matrix)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=gaussian.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=gaussian.classes_)
    disp.plot()
    disp.ax_.set_title("Naive Bayes")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()


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

over_sampling = SMOTE()

X = df[['gender', 'age', 'hypertension',
        'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']

# splitting the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=1/3, random_state=42)
X_train_smote, y_train_smote = over_sampling.fit_resample(X_train, y_train)

randomForestClassifier(X_train, X_test, y_train, y_test)
decisionTree(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
naiveBayes(X_train, X_test, y_train, y_test)
#linearRegression(X_train, X_test, y_train, y_test)
