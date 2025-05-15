from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CleanedData = pd.read_csv('/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv')


#defining features vs labels
X = CleanedData.drop(columns=['LXRa', 'LXRb', 'Molecule Name'], axis=1)

# Making label Classification thresholds
lxra_threshold = 75 #+ merged['LXRa'].std()
lxrb_threshold = 50 #+ merged['LXRb'].std()

# Creating binary labels based on thresholds (if LXRa > threshold and LXRb < threshold)
ya = CleanedData.apply(lambda row: 1 if (row['LXRa'] > lxra_threshold) else 0, axis=1)
yb = CleanedData.apply(lambda row: 1 if (row['LXRb'] < lxrb_threshold) else 0, axis=1)

#splitting the data into training and testing sets
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X, ya, test_size=0.2, random_state=42)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X, yb, test_size=0.2, random_state=42)

clfa = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42)
clfa.fit(Xa_train, ya_train)
clfb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42)
clfb.fit(Xb_train, yb_train)

print(f'LXRa score {clfa.score(Xa_test, ya_test)}')
print(f'LXRb score {clfb.score(Xb_test, yb_test)}')

from sklearn.metrics import confusion_matrix, classification_report

#evaluating the model
predictionsa = clfa.predict(Xa_test)
predictionsb = clfb.predict(Xb_test)

cma = confusion_matrix(ya_test, predictionsa)
cmb = confusion_matrix(yb_test, predictionsb)
print("Confusion Matricies:")

tn, fp, fn, tp = cma.ravel()

print(f'LXRa True Negatives = {tn}')
print(f'LXRa False Positives = {fp}')
print(f'LXRa False Negatives = {fn}')
print(f'LXRa True Positives = {tp}')
print(classification_report(ya_test, predictionsa))

tn, fp, fn, tp = cmb.ravel()

print(f'LXRb True Negatives = {tn}')
print(f'LXRb False Positives = {fp}')
print(f'LXRb False Negatives = {fn}')
print(f'LXRb True Positives = {tp}')
print(classification_report(yb_test, predictionsb))