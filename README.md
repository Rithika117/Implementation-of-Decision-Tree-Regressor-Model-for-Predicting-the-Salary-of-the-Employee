# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import pandas
2.Import Decision Tree classifier
3.Fit the Data in the model
4.Fit the accuracy score
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RITHIKA K
RegisterNumber:212224230230

import pandas as pd
data=pd.read_csv("/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

*/
```

## Output:

![Screenshot 2024-11-28 205254](https://github.com/user-attachments/assets/5e4dbbe0-f8bf-4d64-9aec-59866bdb6769)
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Position  10 non-null     object
 1   Level     10 non-null     int64 
 2   Salary    10 non-null     int64 
dtypes: int64(2), object(1)
memory usage: 372.0+ bytes
```

```
Position    0
Level       0
Salary      0
dtype: int64
```
![Screenshot 2024-11-28 205416](https://github.com/user-attachments/assets/b110f165-4d93-44cd-b2f4-d0779fe7adbb)
```
0     45000
1     50000
2     60000
3     80000
4    110000
Name: Salary, dtype: int64
```
array([80000., 45000.])

462500000.0

0.48611111111111116

![Screenshot 2024-11-28 205622](https://github.com/user-attachments/assets/dc4e2d85-8008-4597-85ba-a11678d7095f)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
