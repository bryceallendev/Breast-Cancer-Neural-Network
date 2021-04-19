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

# Neural Network for Data
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10,3,3), random_state=1)
cv_score = cross_val_score(clf,
                           X=df_data.iloc[:, :-1],
                           y=df_data['target'],
                           cv=5)
plt.plot(cv_score);
plt.show()