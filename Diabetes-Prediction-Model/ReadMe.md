### Author

* [Salim Olanrewaju Oyinlola](https://twitter.com/salimopines)

### Project: Diabetes-ML-Model

### Project Description: This is a classification problem that uses a set of eight medical values to determine/predict if a patient is diabetic or not. 

### URL to Dataset: [Download here](https://www.dropbox.com/s/uh7o7uyeghqkhoy/diabetes.csv?dl=0)

### Line-by-line explanation of Code
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

The block of codes above imports the third party libraries used in the model. 
`import numpy as np` imports the numpy library which can be used to perform a wide variety of mathematical operations on arrays.

`import pandas as pd` imports the pandas library which is used to analyze data.

`from sklearn.preprocessing import StandardScaler` imports the StandardScaler from sklearn's preprocessing library. It standardizes features by removing the mean and scaling to unit variance. 

`from sklearn.model_selection import train_test_split` imports the train_test_split from sklearn's model_selection library. It will be used in spliting arrays or matrices into random train and test subsets.

`from sklearn import svm` imports the Support Vector Machine (SVM) Machine Learning model from the sklearn library. This model will be used in training the model. 

`from sklearn.metrics import accuracy_score` imports the accuracy_score function from sklearn's metrics library. This model is used to ascertain the performance of our model. 

- The Support Vector Machine(SVM) is a supervised machine learning algorithm used for both classification and regression. Though we say regression problems as well its best suited for classification.

```
salim_diabetes_dataset = pd.read_csv(r'C:\Users\OYINLOLA SALIM O\Downloads\diabetes.csv') 
```
This reads the content of the used `.csv` file. The header is set as none, as such, we do not see the name of each property of the dataset. 


```
salim_diabetes_dataset.head()
```

This displays the first five rows of the dataset for better understanding of the dataset. 

It is seen that the dataset has 9 columns - 8 representing the x-values and 1 representing the y-value. 

The x properties are as follows:
- Pregnancies: The number of months of pregnancy of the patient (if applicable).

- Glucose: Glucose levels in patient. 

- BloodPressure: Blood Pressure of the patient.

- SkinThickness: The skin thickness of the patient. 

- Insulin: Insulin level of the patient. 

- BMI: The Body mass index, or BMI, is a measure of body size. It combines a person's weight with their heigh

- DiabetesPedigreeFunction: It provides some data on diabetes mellitus history in relatives and the genetic relationship of those relatives to the patient.

- Age: The age of the patient. 

```
salim_diabetes_dataset.shape
```

This line of code returns the number of rows and Columns in this dataset. It is seen that the data is 768 by 9

```
salim_diabetes_dataset.describe()
```

This displays the statistical measure of the data (i.e.  the mean, median, max, min 25th, 50th and 75th percentile values)

```
salim_diabetes_dataset['Outcome'].value_counts()
```

This line of code returns the amount of the total number of outcome that is 0 or 1 (False or True). 
It is shown that in this dataset, 500 of the 768 samples are False and 200 are True. 

```
salim_diabetes_dataset.groupby('Outcome').mean()
```
This is a very powerful line of code that is able to look at the respective values of the outcome and compare to the mean of individual x-values. 

```
X = salim_diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = salim_diabetes_dataset['Outcome']
```

This block of code separates the data and labels

```
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)
```
 Initiating an instance of the standardize the x values. 

 ```
X = standardized_data
Y = salim_diabetes_dataset['Outcome']
 ```

 Hold on...

 ```
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
 ```

 The train_test_split method that was used earlier is hence called and used to divide the dataset into train set and test set. 

- The 0.2 value of test_size implies that 20% of the dataset is kept for testing whilst 80% is used to train the model. 

```
classifier = svm.SVC(kernel='linear')
```

Initializing an instance of the Support Vector Classifier part of the Support Vector Machine (Since SV, encpmapsses classification and regression) 

```
classifier.fit(X_train, Y_train)
```
This trains the support vector Machine Classifier with the train dataset. 

```
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
```
Evaluating the accuracy score on the training data to see how the model is doing.

- We see an accuracy score of 78.6% which is ONLY FAIR.


```
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
```

Assessing the accuracy score on the test data

- We see an accuracy score of 78.6% which is ONLY FAIR. 

```
# Step 1
input_data = (5,166,72,19,175,25.8,0.587,51)

# Step 2
input_data_as_numpy_array = np.asarray(input_data)

# Step 3
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Step 4
std_data = scaler.transform(input_data_reshaped)
print(std_data)

#Step 5
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
```

This is the most complex part of the code and will be explain in steps. 

Step 1 - This is the user inputted value for all 8 x values (i.e. Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

Step 2 - This changes the input_data to a numpy array

Step 3 - This reshapes the np array as we are predicting for one instance

Step 4 - This standardises the data. 

Step 5 - This prints the result of the prediction. 

