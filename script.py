# script.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("dataset.csv")

# 2. Define features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

<<<<<<< HEAD
# 3. Identify categorical and numerical columns
categorical_cols = ['Geography', 'Gender']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Predictions
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

# 8. Metrics
=======
# 3. Drop non-predictive identifiers
X = X.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 4. Identify categorical and numerical columns
categorical_cols = ['Geography', 'Gender']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)
X_processed = preprocessor.fit_transform(X)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 8. Predictions
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)

# 9. Metrics calculation
>>>>>>> 4633841 (Fixed script.py for identifiers and categorical columns)
metrics = {
    "Train Accuracy": accuracy_score(y_train, y_train_pred),
    "Train Precision": precision_score(y_train, y_train_pred),
    "Train Recall": recall_score(y_train, y_train_pred),
    "Train F1": f1_score(y_train, y_train_pred),
    "Test Accuracy": accuracy_score(y_test, y_pred),
    "Test Precision": precision_score(y_test, y_pred),
    "Test Recall": recall_score(y_test, y_pred),
    "Test F1": f1_score(y_test, y_pred)
}

<<<<<<< HEAD
# 9. Save metrics
=======
# 10. Save metrics
>>>>>>> 4633841 (Fixed script.py for identifiers and categorical columns)
with open("metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

<<<<<<< HEAD
# 10. Confusion matrix
=======
# 11. Confusion matrix
>>>>>>> 4633841 (Fixed script.py for identifiers and categorical columns)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("conf_matrix.png")
plt.close()
