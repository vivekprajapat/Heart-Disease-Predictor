import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv("C:/Users/abc/OneDrive/Documents/Python Project/heart_disease_data.csv")

# print first 5 rows of the dataset
print (heart_data.head())
print()

# print last 5 rows of the dataset
print (heart_data.tail())
print()

# number of rows and columns in the dataset
print (heart_data.shape)
print()

# getting some info about the data
print (heart_data.info())
print()

# checking for missing values
print (heart_data.isnull().sum())
print()

# statistical measures about the data
print (heart_data.describe())
print()

# checking the distribution of Target Variable
print (heart_data['target'].value_counts())
print()

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)
print()
print(Y)
print()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print()


model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)
print()

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)
print()


input_data = (57,1,2,150,168,0,1,174,0,1.6,2,0,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
print()

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
  print()
else:
  print('The Person has Heart Disease')
#   print()
