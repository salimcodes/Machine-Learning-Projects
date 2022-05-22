

### Author

* [Salim Olanrewaju Oyinlola](https://twitter.com/salimopines)

### Project: Rock-vs-Mine-ML-Model

### Project Description: This is a classification problem that uses a set of 60 properties to determine/predict if a sample is mine or rock. 

### URL to Dataset: [Download here](https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view)

### Line-by-line explanation of Code

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
The block of codes above imports the third party libraries used in the model.  
`import numpy as np` imports the numpy library which can be used to perform a wide variety of mathematical operations on arrays.

`import pandas as pd` imports the pandas library which is used to analyze data.

`from sklearn.model_selection import train_test_split` imports the train_test_split function from sklearn's model_selection library. It will be used in spliting arrays or matrices into random train and test subsets.

`from sklearn.linear_model import LogisticRegression` will import the LogisticRegression Machine Learning model from sklearn's linear_model library. This model will be used in training the model. 

`from sklearn.metrics import accuracy_score` imports the accuracy_score function from sklearn's metrics library. This model is used to ascertain the performance of our model. 

- N.B: The logistic regression model which is a classification model was used because the problem is a classification problem. We are trying to group samples into Mine or Rock based on certain properties.  

```
salim_data = pd.read_csv('/content/sonar data.csv', header=None)
```

This reads the content of the used `.csv` file. The header is set as none, as such, we do not see the name of each property of the dataset. 

```
salim_data.head()
```
This displays the first five rows of the dataset for better understanding of the dataset. 

```
salim_data.shape
```
This displays the number of rows and columns in the dataset. 

```
salim_data.describe() 
```
This displays the statistical measure of the data (i.e.  the mean, median, max, min 25th, 50th and 75th percentile values.)

```
salim_data.groupby(60).mean()
```
This is a very powerful line of code that is able to look at the respective values of the output (in column 60) and compare to the mean of individual x-values. 

It is important to note that M represents Mine and R represents Rock.

```
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```

This helps in separating data and Labels. 
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=1)
```
The train_test_split method that was used earlier is hence called and used to divide the dataset into train set and test set. 

- The 0.2 value of test_size implies that 20% of the dataset is kept for testing whilst 80% is used to train the model. 

```
salim_model = LogisticRegression()
```

The logistic regression model is called and an instance is saved under the variable name, `salim_model`. 

```
salim_model.fit(X_train, Y_train)
```
This trains the Logistic Regression model with training data.

```
X_train_prediction = salim_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 
print('Accuracy on training data : ', training_data_accuracy)
```
This evaluates and prints the accuracy on the train set.

A 83% accuracy is observed. 
```
X_test_prediction = salim_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)
```
This evaluates and prints the accuracy on the test set. 

A 76% accuracy was gotten.  
```

# Step 1
input_data = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)

# Step 2
input_data_as_numpy_array = np.asarray(input_data)

# Step 3
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Step 4
prediction = salim_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')

```

This is the most complex part of the code and will be explain in steps. 

Step 1 - This is the user inputted value for all 60 x values. 

Step 2 - This changes the input_data to a numpy array

Step 3 - This reshapes the np array as we are predicting for one instance

Step 4 - This prints the result of the prediction. 
