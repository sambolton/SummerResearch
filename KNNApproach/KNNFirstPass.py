import DataProcessing as data
import numpy as np
import pandas as pd
from sklearn import neighbors, model_selection
from itertools import combinations

Total_list = []

# merged = data.merged_df.drop(['Molecule Name', 'Heavy atom count', 'H-bond donors', 'Rotatable bonds'], axis = 1)

merged = data.merged_df.rename(columns={'Data Summary: % LXRb 100nM (%)': 'LXRb', 'Data Summary: % LXRa 100nM (%)': 'LXRa'})

# print(merged.drop(['LXRa', 'LXRb', 'Topological polar surface area (Å²)', 'Molecular weight (g/mol)', 'Lipinski violations', 'Fsp3', 'CNS MPO score'], axis=1).head())


catagories = list(merged.columns)
catagories = [e for e in catagories if e not in ('Molecule Name', 'LXRa', 'LXRb')]
catagory_superlist = []

for size in range(13, 1, -1):
    for subset in combinations(catagories, size):
        catagory_superlist.append(subset)
        
catagory_superlist = [list(subset) for subset in catagory_superlist if len(subset) < 13]
subset_list = []

for subset in catagory_superlist:
    accuracy_list = []
    for i in range(25):
        forbidden = ['LXRa', 'LXRb', 'Molecule Name'] + subset
        X = np.array(merged.drop(forbidden, axis=1))
        Y = np.array(merged['LXRa'])

        x_train, x_test, y_train, y_test = model_selection.train_test_split(X,Y, test_size=0.2)

        clf = neighbors.KNeighborsRegressor(weights='distance', n_jobs=-1)
        clf.fit(x_train, y_train) # fitting (training)

        accuracy = clf.score(x_test, y_test) # testing
        accuracy_list.append(accuracy)
    Total_list.append(np.average(accuracy_list))
    subset_list.append(subset)
print(max(Total_list))
print(subset_list[np.argmax(Total_list)])


