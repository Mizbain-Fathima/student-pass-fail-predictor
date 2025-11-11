# student_pass_fail_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1) Load dataset ------------------------------------------------------------
df = pd.read_csv("student-por.csv")

# 2) Create Pass/Fail column (1 = Pass, 0 = Fail)
df["Pass"] = (df["G3"] >= 10).astype(int)

# Target & features
y = df["Pass"]
X = df.drop(columns=["G3", "Pass"])

# 3) Identify categorical & numeric features --------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 4) Preprocessor ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# 5) Model pipeline ----------------------------------------------------------
clf = DecisionTreeClassifier(random_state=42)
pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])

# 6) Train-test split --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7) Hyperparameter tuning ---------------------------------------------------
param_grid = {
    "model__max_depth": [4, 6, 8, 10, 12],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("\nBest Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 8) Evaluate ----------------------------------------------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9) Save model --------------------------------------------------------------
joblib.dump(best_model, "best_pass_fail_model.joblib", compress=3)
print("\nSaved: best_pass_fail_model.joblib")
