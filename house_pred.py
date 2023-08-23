import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:\csm33\divya\house_pred_data\House_Rent_Dataset.csv")
#reading the dataset from csv file and storing it in a dataframe df

y = df.iloc[:,3:4]

selected_col = df.drop(columns = ["Rent"])
x = selected_col.iloc[:,:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# 80% training data and rest is testing data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
#fitting on training set only to avoid over fitting of model
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=15)
# n_neighbours means k value
classifier.fit(x_train,y_train)
#training our model with training data
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))