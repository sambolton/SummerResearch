from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
#retrieving Cleaned Data
merged = pd.read_csv('/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv')

#defining features vs labels
X = merged.drop(columns=['LXRa', 'LXRb', 'Molecule Name'], axis=1)

# Making label Classification thresholds
lxra_threshold = 75
lxrb_threshold = 50

# Creating binary labels based on thresholds (if LXRa > threshold and LXRb < threshold)
y = merged.apply(lambda row: 1 if (row['LXRa'] > lxra_threshold and row['LXRb'] < lxrb_threshold) else 0, axis=1)


#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
randomforest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
randomforest.fit(X_train, y_train)

import numpy as np

#evaluating the model
predictions = randomforest.predict(X_test)

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# For binary classification

tn, fp, fn, tp = cm.ravel()

# Calculate ratios

print(f"\nTrue Positive ratio: {tp:.4f}")
print(f"True Negative ratio: {tn:.4f}")
print(f"False Positive ratio: {fp:.4f}")
print(f"False Negative ratio: {fn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# print("Feature Importance: ", randomforest.feature_importances_)
import matplotlib.pyplot as plt

# Plotting feature importances
importances = randomforest.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center", label=features[indices])
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()
