import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = pd.read_csv('/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/CDD CSV Export - Labels.csv')
labels = pd.DataFrame(labels)
labels = labels.dropna(axis=1, how='any')
labels = labels.drop(labels=['Chemical formula', 'Isotope formula', 'Composition', 'Isotope composition', 'Batch Name', 'Batch Formula weight'], axis=1)
labels = labels.drop_duplicates()


features = pd.read_csv('/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/CDD CSV Export - Features.csv')
features = pd.DataFrame(features)
features = features.dropna(axis=1, how='any')
features = features.drop_duplicates()

dup_list = []
none_list = []

for name in labels['Molecule Name']:
    positive = features['Molecule Name'] == name
    positive = positive[positive].index.tolist()
    if len(positive) > 1:
        dup_list.append((name, positive))
    elif len(positive) < 1:
        none_list.append((name, positive))

for element in dup_list:
    features.drop(index = element[1][1])
    
merged_df = pd.merge(labels, features, on='Molecule Name', how='inner')

merged_df = merged_df.drop_duplicates(subset='Molecule Name')


# for name in labels['Molecule Name']:
#     positive = features['Molecule Name'] == name
#     positive = positive[positive].index.tolist()
#     if len(positive) > 1:
#         dup_list.append((name, positive))
#     elif len(positive) < 1:
#         none_list.append((name, positive))
        
# print(dup_list)
# print(none_list)