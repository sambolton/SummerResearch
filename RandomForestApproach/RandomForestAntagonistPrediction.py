from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
#retrieving Cleaned Data
merged = pd.read_csv('/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv')

#defining features vs labels
X = merged.drop(columns=['LXRa', 'LXRb', 'Molecule Name'], axis=1)
y = merged['LXRa'] - merged['LXRb']

#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and training the random forest model
randomforest = RandomForestRegressor(n_estimators=100, random_state=42)
randomforest.fit(X_train, y_train)

#evaluating the model
predictions = randomforest.predict(X_test)
print("MAE: ", mean_absolute_error(y_test, predictions))

print("Feature Importance: ", randomforest.feature_importances_)
import matplotlib.pyplot as plt
import numpy as np

# Plotting feature importances
importances = randomforest.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()




