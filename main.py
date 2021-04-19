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
from sklearn.preprocessing import StandardScaler

# Import dataset
data = load_breast_cancer()
data.target[[10, 50, 85]]
df_data = pd.DataFrame(data['data'], columns=data['feature_names'])
df_data['target'] = data['target']

df_data.head()

X_train, X_test, Y_train, Y_Test = train_test_split(data.data, data.target,stratify=data.target, random_state=1)

# NUERAL NETWORK FOR DATA

# Different Hidden Layers (1,2)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(1,2), max_iter=600, random_state=1)
#cv_score = cross_val_score(clf,
#                           X=df_data.iloc[:, :-1],
#                           y=df_data['target'],
 #                          cv=5)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Different Hidden Layers (1,2)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

#plt.plot(cv_score);
plt.show()

# One Hidden Layer (5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(5), max_iter=750, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("One Hidden Layer (5)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))


# One Hidden Layer (10)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(10), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("One Hidden Layer (10)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))


# Two Hidden Layer (5,10)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(5,10), max_iter=600, random_state=1)


clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Two Hidden Layer (5,10)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Two Hidden Layer (10,5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(10,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Two Hidden Layer (10,5)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Three Hidden Layer (5,10,5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(5,10,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Three Hidden Layer (5,10,5)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Three Hidden Layer (10,20,5)
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Three Hidden Layer (10,20,5)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Gradient Descent Solver (10,20,5) - lbfgs
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Gradient Descent Solver (10,20,5) - lbfgs")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Gradient Descent Solver (10,20,5) - sgd
clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Gradient Descent Solver (10,20,5) - sgd")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Gradient Descent Solver (10,20,5) - adam
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Gradient Descent Solver (10,20,5) - adam")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Activation Function (10,20,5) - logistic
clf = MLPClassifier(activation='logistic', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Activation Function (10,20,5) - logistic")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Activation Function (10,20,5) - relu
clf = MLPClassifier(activation='relu', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Activation Function (10,20,5) - relu")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Different Regularization Parameter (0.01)
clf = MLPClassifier(alpha=0.01,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Different Regularization Parameter (0.01)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))


# Different Regularization Parameter (0.1)
clf = MLPClassifier(alpha=0.1,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Different Regularization Parameter (0.1)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))

# Activation Function (10,20,5) - relu (Scaled)
scaler = StandardScaler()
scaler.fit(X_train)

X_train_S = scaler.transform(X_train)
X_test_S = scaler.transform(X_test)

clf = MLPClassifier(activation='relu', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train_S, Y_train)
prediction = clf.predict(X_test_S)
print("Activation Function (10,20,5) - relu (Scaled)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train_S,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test_S,Y_Test)))

# Activation Function (10,20,5) - relu (Unscaled)
clf = MLPClassifier(activation='relu', alpha=1e-5,
                    hidden_layer_sizes=(10,20,5), max_iter=600, random_state=1)

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
print("Activation Function (10,20,5) - relu (Unscaled)")
print("Training Set Accuracy: {:.2f}".format(clf.score(X_train,Y_train)))
print("Test Set Accuracy: {:.2f}".format(clf.score(X_test,Y_Test)))