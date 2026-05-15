# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import NumPy, Pandas, and Matplotlib libraries. Load the Startup.csv dataset and read R&D Spend as input X and Profit as output Y.

Step 2: Normalize the input data X using mean and standard deviation. Initialize slope m = 0, intercept b = 0, learning rate 0.1, epochs 1000, and sample size n.

Step 3: For each epoch, calculate predicted values using Y_predict = mX + b, find gradients dm and db, then update m and b using gradient descent formula.

Step 4: Print the final slope and intercept values. Plot the scatter graph of dataset points and regression line using Matplotlib.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SRINITHI.S
RegisterNumber:  212225040428
*/
```
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Startup.csv")
X=data['R&D Spend'].values
Y=data['Profit'].values
X=(X-X.mean())/X.std()
m=0
b=0
learning_rate=0.1
epoches=1000
n=len(X)
for i in range(epoches):
    Y_predict=m*X+b
    dm=(-2/n)*np.sum(X*(Y-Y_predict))
    db=(-2/n)*np.sum(Y-Y_predict)
    m=m-learning_rate*dm
    b=b-learning_rate*db
print("Slope(m):",m)
print("Intercept(b):",b)
y_predict=m*X+b
plt.scatter(X,Y)
plt.plot(X,Y_predict)
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()

```

## Output:
<img width="1043" height="641" alt="image" src="https://github.com/user-attachments/assets/b0c711bf-5b05-4de4-a9b6-ec1023e5791d" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
