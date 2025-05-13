#Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("/content/Data_set 2.csv")

# Display the first few rows (optional)
# df.head()

# Drop any rows with missing values to simplify processing
df = df.dropna()

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target variable (y)
X = df.drop("Avenue", axis=1)
y = df["Avenue"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")

# Decode target labels for readable classification report
target_names = label_encoders['Avenue'].classes_
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Plot feature importances
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances.values[:10],
    y=importances.index[:10],
    hue=importances.index[:10],
    palette="viridis",
    legend=False
)
plt.title("Top 10 Important Features for Investment Avenue Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
