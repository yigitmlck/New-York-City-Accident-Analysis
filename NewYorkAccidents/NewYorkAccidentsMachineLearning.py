#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:16:58 2024

@author: yigitmlck
"""


# https://www.kaggle.com/datasets/melodyyiphoiching/nyc-traffic-accidents



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Loading the dataset
veri = pd.read_csv("PrepearedData.csv")

# One-hot encoding 
encveri = pd.get_dummies(veri, columns=['ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME','CRASH DATE', 'CRASH TIME' ])

# Delete extra Date column
encveri.drop('YearMonth', axis=1, inplace=True)

# Creating a Class column
encveri['class'] = encveri['NUMBER OF PERSONS KILLED'].apply(lambda x: 1 if x > 0 else 0)

# Separating training and test datasets
sutunlar = encveri.iloc[:, 1:1033].values
sonuc = encveri.iloc[:, 1033:].values
x_train, x_test, y_train, y_test = train_test_split(sutunlar, sonuc, test_size=79, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. KNN (Euclidean Distance)
knn_euclidean = KNeighborsClassifier()
knn_euclidean.fit(x_train, y_train.ravel())
tahmin_euclidean = knn_euclidean.predict(x_test)

# 2. KNN with Special Distance Metric
# Classification algorithm that takes the target data as the center and creates an outer tangent circle to the data in the center, and decides according to the data density within this circle
# If there is no data within the circle, it returns the label of the closest data
def manuel_mesafe_metriği(x, y):
    import numpy as np

class CustomKNNWithTangentCircle:
    def __init__(self, radius=1.0):
        self.radius = radius

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.center = np.mean(self.X_train, axis=0)  # Merkez hesapla

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Distance to target data point
        distance_to_center = np.linalg.norm(x - self.center)

        # Select data points within the circle
        points_inside_circle = self.X_train[np.linalg.norm(self.X_train - self.center, axis=1) <= self.radius]

        # Get labels of data points within the circle
        labels_inside_circle = self.y_train[np.linalg.norm(self.X_train - self.center, axis=1) <= self.radius]

        # Find the class with the most data points within the circle
        most_common_label = np.bincount(labels_inside_circle).argmax()

        # If the distance to the target data point is smaller than the circle radius,
        # return the most common class as the prediction
        if distance_to_center <= self.radius:
            return most_common_label
        else:
            # If outside the circle, return the label of the nearest data point
            nearest_index = np.argmin(np.linalg.norm(self.X_train - x, axis=1))
            return self.y_train[nearest_index]



knn_ozel = KNeighborsClassifier(metric=manuel_mesafe_metriği)
knn_ozel.fit(x_train, y_train.ravel())
tahmin_ozel = knn_ozel.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Definition that allows us to calculate and print performance metrics in bulk
def metrikleri_hesapla(y_test, tahmin):
    conf_matrix = confusion_matrix(y_test, tahmin)
    dn, yp = conf_matrix[0, 0], conf_matrix[0, 1]
    yn, dp = conf_matrix[1, 0], conf_matrix[1, 1]
    duyarlılık = dp / (dp + yn)
    özgüllük = dn / (dn + yp)
    kesinlik = dp / (dp + yp)
    f_ölçütü = 2 * (duyarlılık * kesinlik) / (duyarlılık + kesinlik)

    return {
        "Success Rate": accuracy_score(y_test, tahmin),
        "True Negative": dn,
        "False Positive": yp,
        "False Negative": yn,
        "True Positive": dp,
        "Sensitivity": duyarlılık,
        "Specifity": özgüllük,
        "Precision": kesinlik,
        "F-Criteria": f_ölçütü
    }

# KNN (Euclidean Distance) Metrics
print("KNN (Euclidean Mesafe) Performans:")
print(confusion_matrix(y_test, tahmin_euclidean))
print(classification_report(y_test, tahmin_euclidean))
print(metrikleri_hesapla(y_test, tahmin_euclidean))

# KNN Metrics with Special Distance Metric
print("\Special Mesafe Metriği ile KNN Performans:")
print(confusion_matrix(y_test, tahmin_ozel))
print(classification_report(y_test, tahmin_ozel))
print(metrikleri_hesapla(y_test, tahmin_ozel))

from sklearn.ensemble import RandomForestClassifier

# 3. Random Forest (voting)
rf_oylama = RandomForestClassifier(random_state=0)
rf_oylama.fit(x_train, y_train.ravel())
tahmin_rf_oylama = rf_oylama.predict(x_test)

# 4. Random Forest (Gini)
rf_ozel = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=23, min_samples_split=7, random_state=0)
rf_ozel.fit(x_train, y_train.ravel())
tahmin_rf_ozel = rf_ozel.predict(x_test)

# Performans metrikleri
print("Random Forest (voting) Performance:")
print(confusion_matrix(y_test, tahmin_rf_oylama))
print(metrikleri_hesapla(y_test, tahmin_rf_oylama))

print("Special Random Forest Performance:")
print(confusion_matrix(y_test, tahmin_rf_ozel))
print(metrikleri_hesapla(y_test, tahmin_rf_ozel))



import numpy as np
from collections import Counter

def calculate_twoing(y, split_indices):
   
    left_class_counts = Counter(y[:split_indices])
    right_class_counts = Counter(y[split_indices:])
    
    total_samples = len(y)
    total_left = len(y[:split_indices])
    total_right = len(y[split_indices:])
    
    left_gini = 1.0 - sum((count / total_left) ** 2 for count in left_class_counts.values())
    right_gini = 1.0 - sum((count / total_right) ** 2 for count in right_class_counts.values())
    
    twoing = total_left * left_gini + total_right * right_gini
    
    return twoing

def find_best_split(X, y):

    num_samples, num_features = X.shape
    best_split_feature = None
    best_split_index = None
    best_twoing = float('inf')

    for feature in range(num_features):
        feature_values = X[:, feature]
        unique_values = np.unique(feature_values)

        for value in unique_values:
            split_indices = feature_values < value
            current_twoing = calculate_twoing(y, split_indices)

            if current_twoing < best_twoing:
                best_twoing = current_twoing
                best_split_feature = feature
           



