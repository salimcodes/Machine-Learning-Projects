## Author

* [Salim Olanrewaju Oyinlola](https://twitter.com/salimopines)

### Project: Credit Card Fraud Detection using Machine Learning with Python.

### Project Description: 

### URL to Dataset: [Download here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Line-by-line explanation of Code

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

`import numpy as np` imports the numpy library which can be used to perform a wide variety of mathematical operations on arrays.

`import pandas as pd` imports the pandas library which is used to analyze data.

`from sklearn.model_selection import train_test_split` imports the train_test_split function from sklearn's model_selection library. It will be used in spliting arrays or matrices into random train and test subsets.

`from sklearn.linear_model import LogisticRegression` imports the `LogisticRegression` Machine Learning model from sklearn's linear_model library. This model will be used in training the model. 


> The logistic regression model, which is a classification model was used in this problem since the problem statement requires us to classify samples into Mine or Rock based on the given attributes.

`from sklearn.metrics import accuracy_score` imports the `accuracy_score` function from sklearn's metrics library. This model is used to ascertain the performance of our model. 

```
salim_credit_card_data = pd.read_csv(r'C:\Users\OYINLOLA SALIM O\Downloads\creditcard.csv')
```
This loads the dataset to a Pandas DataFrame. 

```
salim_credit_card_data.head()
```
This displays the first 5 rows of the dataset. 

```
salim_credit_card_data.info()
```
This displays the dataset informations.

```
salim_credit_card_data['Class'].value_counts()
```
This shows the distribution of legit transactions & fraudulent transactions in the 'outcome' column. 

We see;

0    284315

1       492

It is seen that this Dataset is highly unblanced

`0` --> Normal Transaction

`1` --> fraudulent transaction. 

```
legit = salim_credit_card_data[salim_credit_card_data.Class == 0]
fraud = salim_credit_card_data[salim_credit_card_data.Class == 1]
```

This separates the data for analysis according to their outcomes. 

```
legit.Amount.describe()
```

This shows a statistical measures of the data that comes out as legit (i.e. 0)

```
fraud.Amount.describe()
```

This shows a statistical measures of the data that comes out as legit (i.e. 1)

```
salim_credit_card_data.groupby('Class').mean()
```

This line of code compares the mean values for both fraud and legit transactions. 

- All steps taken thus far is taken to understand the dataset. 

```
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
```

This block of code first builds a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions and then, concatenates the two DataFrames

- Note that we see 492 number of legit samples because the number of Fraudulent Transactions --> 492

```
new_dataset.head()
```

Displaying the new dataset showing the randomness of the dataset sample. 

```
new_dataset['Class'].value_counts()
```
 We see equal number of samples with outcomes, `0` and `1`. 


```
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
```

This splits the data into Features & Targets. 

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=1)
```
The `train_test_split` method which was imported earlier is hence called and used to divide the dataset into train set and test set. 

NOTE: The `0.2` value of test_size implies that 20% of the dataset is kept for testing whilst 80% is used to train the model. 


```
model = LogisticRegression()
model.fit(X_train, Y_train)
```

This block of code creates an instance of the LogisticRegression models and then, trains the Logistic Regression Model with Training Data. 

```
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
```

This displays the accuracy on the training data. 

Accuracy on Training data :  `0.9415501905972046`. 

```
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
```

This displays the accuracy on the training data. 

Accuracy on Training data :  `0.8883248730964467`. 



