### Author

* [Salim O. Oyinlola](https://twitter.com/salimopines)

### Project: Car-Prices-Prediction-Model

### Project Description: This project involves the binary classification of wines into good and bad quality based on a set of properties.

### URL to Dataset: [Download here](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv)

### Line-by-line explanation of Code
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics
```
The block of codes shown above imports the third party libraries used in the model.  

`import pandas as pd` is used to import the pandas library which is used to analyze data.

`import matplotlib.pyplot as plt` imports the PyPlot function from the MatPlotLib library which is used to visualize data and trends in the data.

`import seaborn as sns` imports the seaborn library which is used for making statistical graphics. It builds on top of matplotlib and integrates closely with pandas data structures. Seaborn helps you explore and understand your data.

`from sklearn.model_selection import train_test_split` imports the train_test_split function from sklearn's model_selection library. It will be used in spliting arrays or matrices into random train and test subsets.

`from sklearn.linear_model import Lasso` imports the Lasso linear regresssion machine learning model from sklearn's linear_model library. 

`from sklearn import metrics` imports the metrics library from the sklearn library. This model is used to ascertain the performance of our model. 

```
salim_car_dataset = pd.read_csv(r'C:\Users\OYINLOLA SALIM O\Downloads\car data.csv')
```
This line of code loads the data from csv file to pandas dataframe named `salim_car_dataset`.

```
salim_car_dataset.head()
```
This line of code displays the first 5 rows of the dataframe. 

```
salim_car_dataset.shape
```
This line of code checks the number of rows and columns. The observed output is `(301, 9)`. 

```
salim_car_dataset.isnull().sum()
```
This line of code checks the number of missing values. It is seen that there is no missing value in the dataset. 

```
print(salim_car_dataset.Fuel_Type.value_counts())
print(salim_car_dataset.Seller_Type.value_counts())
print(salim_car_dataset.Transmission.value_counts()) 
```
This block of code checks the distribution of categorical data in the `Fuel_Type`, `Seller_Type` and `Transmission` columns of the dataset. 

In the `Fuel_Type` column, we see:
`Petrol`    239
`Diesel`     60
`CNG`         2

In the `Seller_Type` column, we see:
`Dealer`        195
`Individual`    106

In the `Transmission` column, we see: 
`Manual`       261
`Automatic`     40

```
salim_car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
```
This line of code encodes the `Fuel_Type` Column. 

```
salim_car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
```
This line of code encodes the `Seller_Type` Column. 

```
salim_car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
```
This line of code encodes the `Transmission` Column. 

```
salim_car_dataset.head()
```
This line of code would print the first five rows of the dataset to show the label encoded dataset. 

```
X = salim_car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = salim_car_dataset['Selling_Price']
```
This block of code separates the data and Label i.e. into `X` and `Y`. 

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)
```
The train_test_split method is hence called and used to divide the dataset into train set and test set. 

```
lass_reg_model = Lasso()
```
This line of code loads the linear regression model. 

```
lass_reg_model.fit(X_train,Y_train)
```
This line of code trains the model with the train dataset. (i.e. `X_train` and `Y_train`)

```
training_data_prediction = lass_reg_model.predict(X_train)
```
This line of code predicts on the Training data.

```
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)
```
This block of codes displays the R squared Error. The R squared Error is given as `0.8427856123435794`.

```
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
```
This block of codes helps in visualizing the actual prices and Predicted prices. 

```
test_data_prediction = lass_reg_model.predict(X_test)
```
This line of code predicts on the test data.

```
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)
```
This block of codes displays the R squared Error. The R squared Error is given as `0.8709167941173195`.

```
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
```

This block of codes helps in visualizing the actual prices and Predicted prices. 
