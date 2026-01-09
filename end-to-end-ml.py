import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import pickle

df = pd.read_csv("students_placement_data.csv")
df.head() # shows overview of the data

df.info() # used to check if there is any missing value or not

df = df.iloc[:,1:]

# EDA
plt.scatter(df["IQ"],df["CGPA"],c=df["Placement"])

# Extracting input and output columns
X = df.iloc[:,:2]
y = df.iloc[:,-1]

# Train-Test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

# Scaling the value
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model
clf = LogisticRegression()
clf.fit(X_train,y_train)

# Scoring the acuracy or evaluating the model
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)

plot_decision_regions(X_train,y_train.values, clf = clf, legend = 2)

pickle.dump(clf, open("placement_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))