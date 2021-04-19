# Bryce Allen
# Applied Machine Learning
# Project 3

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import dataset
data = load_breast_cancer()
data.target[[10, 50, 85]]
df_data = pd.DataFrame(data['data'], columns=data['feature_names'])
df_data['target'] = data['target']

df_data.head()

# NUERAL NETWORK FOR DATA

# Different Hidden Layers (1,2)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(1,2), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# One Hidden Layer (5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# One Hidden Layer (10)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(10), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Two Hidden Layer (5,10)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(5,10), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Two Hidden Layer (10,5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(10,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Three Hidden Layer (5,10,5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(5,10,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Three Hidden Layer (10,20,5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Gradient Descent Solver (10,20,5) - lbfgs
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Gradient Descent Solver (10,20,5) - sgd
clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Gradient Descent Solver (10,20,5) - adam
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Activation Function (10,20,5) - logistic
clf = MLPClassifier(activation='logistic', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Activation Function (10,20,5) - relu
clf = MLPClassifier(activation='relu', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()

# Different Regularization Parameter (0.01,0.1)
clf = MLPClassifier(alpha=(0.01,0.1), 
                    hidden_layer_sizes=(10,20,5), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()