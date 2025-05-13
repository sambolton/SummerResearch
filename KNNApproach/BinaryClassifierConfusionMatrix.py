import pandas as pd 
import numpy as np
from sklearn import neighbors
from sklearn import model_selection

CleanedData = pd.read_csv("/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv")

# Binary Classification Threshold
lxra_threshold = 80.0
lxrb_threshold = 50.0

# Thresholding ~top 1/3 LXRb inhibitors and bottom 1/3 LXRa inhibitors
CleanedData['LXRa'] = np.where(CleanedData['LXRa'] > lxra_threshold, 1, 0)
CleanedData['LXRb'] = np.where(CleanedData['LXRb'] < lxrb_threshold, 1, 0)

forbidden = ['LXRa', 'LXRb', 'Molecule Name']
X = CleanedData.drop(columns=['LXRb', 'LXRa', 'Molecule Name'])
y = CleanedData.apply(lambda row: 1 if (row['LXRa'] == 1 and row['LXRb'] == 1) else 0, axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

# Create and train a KNN Classifier model
clf = neighbors.KNeighborsClassifier(weights='distance', n_jobs=-1)
clf.fit(x_train, y_train) # fitting (training)

accuracy = clf.score(x_test, y_test) # testing
print(f"Accuracy: {accuracy:.2f}")

from sklearn.metrics import confusion_matrix, classification_report
# Generating the confusion matrix
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

print(y_test)

# Evaluating the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# positive_total = fp + tp
# negtive_total = tn + fn
tp_ratio = tp #/ positive_total
tn_ratio = tn #/ negtive_total
fp_ratio = fp #/ positive_total
fn_ratio = fn #/ negtive_total

print(f"True Positive ratio: {tp_ratio:.4f}")
print(f"True Negative ratio: {tn_ratio:.4f}")
print(f"False Positive ratio: {fp_ratio:.4f}")
print(f"False Negative ratio: {fn_ratio:.4f}")

# For more detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))