import pandas as pd 
import numpy as np
from sklearn import neighbors
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, r2_score

CleanedData = pd.read_csv("/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv")


# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a KNN Regressor model
# knn_regressor = neighbors.KNeighborsRegressor(n_neighbors=3)
# knn_regressor.fit(X_train, y_train)

accuracy_list = []
for i in range(25):
    forbidden = ['LXRa', 'LXRb', 'Molecule Name']
    X = CleanedData.drop(columns=['LXRb', 'LXRa', 'Molecule Name', 'log S'])
    y = CleanedData['LXRb'] - CleanedData['LXRa']
# Split the data into training and testing sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

# Create and train a KNN Regressor model
    clf = neighbors.KNeighborsRegressor(weights='distance', n_jobs=-1)
    clf.fit(x_train, y_train) # fitting (training)

    accuracy = clf.score(x_test, y_test) # testing
    accuracy_list.append(accuracy)
# Evaluate the model
print("Average accuracy: ", np.average(accuracy_list))
