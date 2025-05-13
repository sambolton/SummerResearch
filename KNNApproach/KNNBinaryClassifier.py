import pandas as pd 
import numpy as np
from sklearn import neighbors
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, r2_score

CleanedData = pd.read_csv("/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv")

# Binary Classification Threshold
lxra_threshold = 75
lxrb_threshold = 50
CleanedData['LXRa'] = np.where(CleanedData['LXRa'] > lxra_threshold, 1, 0)
CleanedData['LXRb'] = np.where(CleanedData['LXRb'] < lxrb_threshold, 1, 0)

accuracy_list = []
for i in range(25):
    forbidden = ['LXRa', 'LXRb', 'Molecule Name']
    X = CleanedData.drop(columns=['LXRb', 'LXRa', 'Molecule Name'])
    y = CleanedData.apply(lambda row: 1 if (row['LXRa'] == 1 and row['LXRb'] == 1) else 0, axis=1)
    
# Split the data into training and testing sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

# Create and train a KNN Classifier model
    clf = neighbors.KNeighborsClassifier(weights='distance', n_jobs=-1)
    clf.fit(x_train, y_train) # fitting (training)

    accuracy = clf.score(x_test, y_test) # testing
    accuracy_list.append(accuracy)
    
# model evaluation
print("Average accuracy: ", np.average(accuracy_list))