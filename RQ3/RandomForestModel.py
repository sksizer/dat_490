# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Sklearn imports
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Select plot formatting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Import Data
path = "/content/sample_data/LeanData.csv"
df = pd.read_csv(path)


# Shuffle the data set
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split Variables into X and y

X = df.drop('_HLTHPL1', axis=1)
y = df['_HLTHPL1']


# Check for and handle missing values in X
print(f"Number of missing values in X before imputation: {np.isnan(X).sum()}")

from sklearn.impute import SimpleImputer

# Use SimpleImputer to fill missing values, use median to prevent skewing
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Now use X_imputed for train-test split and further processing
X = X_imputed
print(f"Number of missing values in X after imputation: {np.isnan(X).sum()}")

# Train test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# Hyper parameter tuning

param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],             # Number of trees in the forest
    'max_features': ['sqrt', 'log2', 0.5, None],          # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30, 40, 50],             # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10, 15],                 # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4, 6],                    # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]                           # Method of selecting samples for training each tree
}

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=RANDOM_STATE)

# RandomizedSearchCV performs cross validation

random_search = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_dist,
    n_iter=20,        # Number of parameter settings that are sampled

    cv=3,             # Number of CV folds
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring='accuracy'# Metric to optimize for
)


#  Fit CV to the training data
print("\n--- Starting Hyperparameter Tuning with RandomizedSearchCV ---")
random_search.fit(X_train, y_train)

# Generate best estimator
best_rf_model = random_search.best_estimator_
print("\n--- Best Hyperparameters Found ---")
print(random_search.best_params_)

# Evaluate the best model on the test Set
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Evaluation on Test Set ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_rf_model.classes_,
            yticklabels=best_rf_model.classes_)
plt.title("Confusion Matrix for Best Random Forest Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Feature importance metric
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1] # Sort features by importance
sorted_feature_names = [df.drop('_HLTHPL1', axis=1).columns[i] for i in indices]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances from Best Random Forest Model")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), sorted_feature_names, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()

# Plotting tree from the model
single_tree = best_rf_model.estimators_[0]

plt.figure(figsize=(25, 15)) # Adjust tree size
plot_tree(
    single_tree,
    feature_names=df.drop('_HLTHPL1', axis=1).columns.tolist(),
    class_names=[str(c) for c in best_rf_model.classes_],
    filled=True,
    rounded=True,
    precision=2,
    fontsize=7,
    max_depth=5
)
plt.title(f"Visualization of a Single Decision Tree (estimator 0, max_depth=5) from the Best Random Forest", fontsize=16)
plt.tight_layout()
plt.show()

