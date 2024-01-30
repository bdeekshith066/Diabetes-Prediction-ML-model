#Importing Dependies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


 #Data Collection and Analysis

 #IMA Diabetes Dataset
.   diabetes_dataset = pd.read_csv('/delete ml csv.csv')  


# printing the first 5 rows of the dataset
.  diabetes_dataset.head()

#number of rows and columns in this dataset
.  diabetes_dataset.shape


.  diabetes_dataset['Outcome'].value_counts()


#printing the first 5 rows of the dataset
.  diabetes_dataset.head()


#number of rows and columns in this dataset
.  diabetes_dataset.shape


.  diabetes_dataset['Outcome'].value_counts()



#getting the  statistical measures of the data
.  diabetes_dataset.describe()


.  diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
.  X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
.  Y = diabetes_dataset['Outcome']

.  print(X)

.  print(Y)

.  scaler = StandardScaler()

.  scaler.fit(X)

.  standardized_data = scaler.transform(X)

.  print(standardized_data)

.  X = standardized_data
.  Y = diabetes_dataset['Outcome']

.  print(X)
.  print(Y)

.  X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2 , stratify = Y , random_state = 2 )


.  print(X.shape , X_train.shape, X_test.shape)

# Training the model

.  classifier = svm.SVC(kernel = 'linear')

#training the support vector Machine Classifier
.  classifier.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score

# accuracy score on the training data
.  X_train_prediction = classifier.predict(X_train)
.  training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


.  print('Accuracy score of the training data: ',training_data_accuracy)

# Making a Predictive System

.input_data = (4,110,92,0,0,37.6,0.191,30)

input_data_as_numpy_array = np.asarray(input_data)  #chaning the inpit data to numpy arrray

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)      #reshape the array as we are predicting for one

std_data = scaler.transform(input_data_reshaped)    #standarize the input device
print(std_data)

prediction = classifier.predict(std_data)

print(prediction)

if(prediction[0] == 0):
  print('The person is not diabetic')
else :
  print('the person is diabetic')








                 
