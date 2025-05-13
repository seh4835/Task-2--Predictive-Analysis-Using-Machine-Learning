# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score
)

# Load dataset
df = pd.read_csv("Data_set 2.csv")

# Drop missing values
df = df.dropna()

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# CLASSIFICATION TASK

# Predicting the investment Avenue (categorical)
X_class = df.drop(columns=["Avenue"])
y_class = df["Avenue"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(Xc_train, yc_train)

# Predict and evaluate
yc_pred = clf.predict(Xc_test)
acc = accuracy_score(yc_test, yc_pred)
print("=== Classification Report ===")
print(f"Accuracy: {acc:.2f}")
print(classification_report(yc_test, yc_pred, target_names=label_encoders['Avenue'].classes_, zero_division=0))

# Plot feature importance
importances = pd.Series(clf.feature_importances_, index=X_class.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(
    x=importances.values[:10],
    y=importances.index[:10],
    hue=importances.index[:10],    
    palette="viridis",
    legend=False                    
)
plt.title("Top 10 Feature Importances (Classification)")
plt.tight_layout()
plt.show()



# REGRESSION TASK

# Predicting Mutual Fund score (numeric)
X_reg = df.drop(columns=["Mutual_Funds"])
y_reg = df["Mutual_Funds"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train Linear Regression
reg = LinearRegression()
reg.fit(Xr_train, yr_train)

# Predict and evaluate
yr_pred = reg.predict(Xr_test)
mse = mean_squared_error(yr_test, yr_pred)
r2 = r2_score(yr_test, yr_pred)

print("\n=== Regression Report ===")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted with ideal line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=yr_test, y=yr_pred)
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--', label='Ideal Prediction Line (y=x)')
plt.xlabel("Actual Mutual Fund Score")
plt.ylabel("Predicted Score")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
