# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Sklearn imports
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Load the data

path = "/content/sample_data/LeanData.csv"
df = pd.read_csv(path)

# Shuffle the data set
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Seperate dependent from independant variables

X = df.drop('_HLTHPL1', axis=1)
y = df['_HLTHPL1']

# Check for and handle missing values in X
print(f"Number of missing values in X before imputation: {np.isnan(X).sum()}")

from sklearn.impute import SimpleImputer

# Use SimpleImputer to fill missing values of each column with a median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Use X_imputed for train-test split and further processing
X = X_imputed
print(f"Number of missing values in X after imputation: {np.isnan(X).sum()}")

# Check for and handle missing values in y
print(f"Number of missing values in y before imputation: {np.isnan(y).sum()}")

# Use SimpleImputer to fill missing values of y with a median
imputer_y = SimpleImputer(strategy='median')
y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1)) # Reshape y for imputer

# Use y_imputed for train-test split
y = y_imputed.ravel() # Flatten y_imputed back to a 1D array
print(f"Number of missing values in y after imputation: {np.isnan(y).sum()}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y # Stratify for imbalanced classes
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use transform only on test set

# Hyperparameter Tuning using GridSearchCV
print("\n--- Hyperparameter Tuning ---")
log_reg = LogisticRegression(random_state=42, max_iter=1000) # Increased max iterations

# Define the parameter grid
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Use StratifiedKFold for cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)


# Get the best estimator
best_log_reg = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best cross-validation ROC AUC score: {grid_search.best_score_:.4f}")

# Model Evaluation on Test Set
print("\n--- Model Evaluation on Test Set ---")
y_pred = best_log_reg.predict(X_test_scaled)
# Probabilities for all classes
y_pred_proba = best_log_reg.predict_proba(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Visualizations

# Get unique classes
classes = np.unique(y_test)
n_classes = len(classes)

# Binarize the output for OvR
y_test_bin = label_binarize(y_test, classes=classes)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Predicted Class {c}' for c in classes],
            yticklabels=[f'Actual Class {c}' for c in classes])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC Curve (One-vs-Rest)
plt.figure(figsize=(10, 8))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Precision-Recall Curve (One-vs-Rest)
plt.figure(figsize=(10, 8))
precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(recall[i], precision[i], lw=2, color=colors[i],
             label=f'PR curve of class {classes[i]} (area = {average_precision[i]:.2f})')


for i in range(n_classes):
     no_skill = np.sum(y_test == classes[i]) / len(y_test)
     plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color=colors[i], lw=1, alpha=0.5,
              label=f'No Skill (Class {classes[i]})')


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (One-vs-Rest)')
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Visual of Top 5 Correlating Attributes

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Get the correlations with the target variable '_HLTHPL1'
target_correlation = correlation_matrix['_HLTHPL1'].sort_values(ascending=False)

# Drop the correlation of '_HLTHPL1' with itself
target_correlation = target_correlation.drop('_HLTHPL1')

# Get the top 5 attributes with the highest absolute correlation
top_5_correlated_attributes = target_correlation.abs().nlargest(5)

# Get the actual correlation values for the top 5 attributes
top_5_correlation_values = target_correlation[top_5_correlated_attributes.index]

print("Top 5 attributes with the highest correlation to _HLTHPL1:")
print(top_5_correlation_values)

# Create a bar plot of the top 5 correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=top_5_correlation_values.index, y=top_5_correlation_values.values, palette="viridis")
plt.title('Top 5 Attribute Correlations with _HLTHPL1')
plt.xlabel('Attribute')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
