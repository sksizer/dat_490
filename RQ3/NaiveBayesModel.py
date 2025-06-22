# Import Libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns


# Load dataset
# Import Data
path = "/content/sample_data/Datafile/LeanData.csv"
df = pd.read_csv(path)

X = df.drop('_HLTHPL1', axis=1)
y = df['_HLTHPL1']

# Check for and handle missing values in X
print(f"Number of missing values in X before imputation: {np.isnan(X).sum()}")

from sklearn.impute import SimpleImputer

# Use SimpleImputer to fill missing values, for example, with the mean of each column
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Use X_imputed for train-test split and further processing
X = X_imputed
print(f"Number of missing values in X after imputation: {np.isnan(X).sum()}")

# Initialize Naive Bayes model
model = GaussianNB()

# Define number of folds for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracy_scores = []
confusion_matrices = []

# Convert X and y to NumPy arrays for KFold
X_np = X
y_np = y

for fold, (train_index, test_index) in enumerate(kf.split(X_np)):
    X_train, X_test = X_np[train_index], X_np[test_index]
    y_train, y_test = y_np[train_index], y_np[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

print(f"\nAverage Accuracy across {n_splits} folds: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")

# Visualizations

# Get feature names from DataFrame columns
feature_names = df.columns.tolist()[:-1] # Exclude the target column

# Get target names from unique values in the target column
target_names = sorted(df['_HLTHPL1'].unique().tolist())


# Distribution of top features
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x=feature_names[0], hue='_HLTHPL1', kde=True, palette='viridis')
plt.title(f'Distribution of {feature_names[0]}')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x=feature_names[1], hue='_HLTHPL1', kde=True, palette='viridis')
plt.title(f'Distribution of {feature_names[1]}')


plt.tight_layout()
plt.show()

# Visualize the Confusion Matrix from the last fold 
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrices[-1], annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Last Fold)')
plt.show()

# ROC Curve and AUC 

model.fit(X, y)
y_prob = model.predict_proba(X)

plt.figure(figsize=(10, 8))

for i in range(len(target_names)):
    # Use the original y for roc_curve
    fpr, tpr, _ = roc_curve(y, y_prob[:, i], pos_label=target_names[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve for {target_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
