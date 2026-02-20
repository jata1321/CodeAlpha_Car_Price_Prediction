import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =====================================
# 1️⃣ Load Dataset (Professional Way)
# =====================================
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "car_data.csv")

df = pd.read_csv(file_path)

print("First 5 rows of dataset:\n")
print(df.head())

print("\nColumns in dataset:\n")
print(df.columns)

# =====================================
# 2️⃣ Basic Cleaning
# =====================================
print("\nMissing Values:\n")
print(df.isnull().sum())

# =====================================
# 3️⃣ Convert Categorical to Numeric
# =====================================
df = pd.get_dummies(df, drop_first=True)

# =====================================
# 4️⃣ Define Target Column
# =====================================
# 🔥 IMPORTANT: Change this if needed
target_column = "Selling_Price"   # Change if your dataset has different name

X = df.drop(target_column, axis=1)
y = df[target_column]

# =====================================
# 5️⃣ Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 6️⃣ Train Model
# =====================================
model = LinearRegression()
model.fit(X_train, y_train)

# =====================================
# 7️⃣ Predictions
# =====================================
y_pred = model.predict(X_test)

# =====================================
# 8️⃣ Evaluation
# =====================================
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# =====================================
# 9️⃣ Visualization
# =====================================
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()