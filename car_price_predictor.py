import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('car_data.csv')

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Data Cleaning & Feature Engineering
df['Current_Year'] = 2026
df['Car_Age'] = df['Current_Year'] - df['Year']
df.drop(['Car_Name', 'Current_Year'], axis=1, inplace=True)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: ₹{mae:.2f} Lakhs")
print("="*50)

# Save the model
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model saved as 'car_price_model.pkl'")

# Example Prediction
example = X_test.iloc[0:1]
pred_price = model.predict(example)[0]
print(f"\nExample Prediction: ₹{pred_price:.2f} Lakhs")
