import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger dataset
data = pd.read_csv("dataset.csv")

# Suppression colonnes inutiles si besoin
# data = data.drop(columns=["id"])  # exemple

# Séparer features / target
X = data.drop("churn", axis=1)
y = data["churn"]

# Identifier colonnes numériques et catégorielles
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 2. Préprocessing
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Gestion déséquilibre classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 5. Modèle
clf = Pipeline(steps=[("preprocessor", preprocessor),
                      ("classifier", RandomForestClassifier(random_state=42))])

clf.fit(X_train_res, y_train_res)

# 6. Prédictions
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# 7. Metrics
report_train = classification_report(y_train, y_pred_train)
report_test = classification_report(y_test, y_pred_test)

with open("metrics.txt", "w") as f:
    f.write("=== TRAIN METRICS ===\n")
    f.write(report_train)
    f.write("\n=== TEST METRICS ===\n")
    f.write(report_test)

# 8. Matrice de confusion (test)
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("conf_matrix.png")
