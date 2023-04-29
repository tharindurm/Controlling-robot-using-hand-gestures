from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

# Path to the dataset
path = "keypoint.csv"

# Reading dataset
df = pd.read_csv(path)

# Printing 1st 5 rows of the data set to ensure file and content loaded correctly
print( df.head() )

# Checking for empty parameters in records
df.isna().any()

# Separating column numbers from 1 to the last column as input for the MLP classifier
X = df.iloc[:, 1:]

#Separating 1st column as the output
y = df.iloc[:, :1]

# Splitting data in to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Making the MLP model
model = MLPClassifier(hidden_layer_sizes=(10),
activation='logistic', solver='sgd',
learning_rate='constant', learning_rate_init=0.001,
max_iter=1000)

# Training the model on dataset
model.fit(X_train, y_train.values.ravel())

# Testing the model using test data
predictions = model.predict(X_test)

# Printing model accuraccy on training dataset and testing dataset
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# Filename/Path for the model to be saved
filename = 'finalized_model.sav'

# Saving model
pickle.dump(model, open(filename, 'wb'))
