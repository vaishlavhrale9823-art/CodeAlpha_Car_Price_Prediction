# ==========================================
# Car Price Prediction - CodeAlpha Task 3
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("car data.csv")

print("----- Dataset Preview -----")
print(df.head())

print("\n----- Dataset Info -----")
print(df.info())

print("\n----- Missing Values -----")
print(df.isnull().sum())

# Drop Car_Name (not useful)
df.drop("Car_Name", axis=1, inplace=True)

# Convert categorical data into numbers
df = pd.get_dummies(df, drop_first=True)

# Feature & Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

print("\n----- Model Evaluation -----")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()