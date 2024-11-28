# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Yokesh I
RegisterNumber: 24013623 
*/
```
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Step 2: Load the dataset
data = pd.read_csv("Salary.csv")

# Step 3: Display dataset information
print("Dataset Preview:")
print(data.head())
print("\nDataset Info:")
data.info()
print("\nMissing Values:")
print(data.isnull().sum())

# Step 4: Encode categorical variables (if any)
le = LabelEncoder()
if "Position" in data.columns:
    data["Position"] = le.fit_transform(data["Position"])
else:
    print("Warning: 'Position' column not found in the dataset.")

# Step 5: Define features (X) and target (y)
x = data[["Position", "Level"]]  # Feature columns
y = data["Salary"]  # Target variable

# Step 6: Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Step 7: Train the Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=2)
dt.fit(x_train, y_train)

# Step 8: Make predictions on the test set
y_pred = dt.predict(x_test)

# Step 9: Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score of the model: {r2:.2f}")

# Step 10: Make a sample prediction
sample_prediction = dt.predict([[5, 6]])  # Adjust sample values as needed
print(f"\nPrediction for sample input [5, 6]: {sample_prediction[0]}")

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![11](https://github.com/user-attachments/assets/be1f781e-8a79-47ad-97fb-ac1c7252719f)
![12](https://github.com/user-attachments/assets/4a2c3160-f2b6-4057-bb65-95e9f55ed9ef)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
