import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv("MS_Survey.csv")

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.replace("\\", "", regex=False)
    .str.replace("\n", " ")
    .str.replace("  ", " ")
)

# Display the first few columns to verify
print(df.columns[:10])

# Convert to numeric and fill NaNs
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(0).astype(int)

# Distribute the target variable
print(df["Preliminary_Diagnosis"].value_counts())

# Display unique values of the target variable
print(df["Preliminary_Diagnosis"].unique())

df["Preliminary_Diagnosis"].unique()

X = df.drop(columns=["Preliminary_Diagnosis"])
y = df["Preliminary_Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind="barh")
plt.title("Top 15 Important Features")
plt.show()
