import pandas as pd
# Feature Set
features = pd.DataFrame(pd.read_csv('/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/CDD CSV Export - Features.csv'))
features = features.drop('CDD Number', axis=1)
main = features[['Molecule Name', 'Data Summary: FC LXRa Agonist', 'Data Summary: FC LXRb Agonist']]
main = main.rename(columns={'Data Summary: FC LXRa Agonist': 'LXRa', 'Data Summary: FC LXRb Agonist': 'LXRb'})
main = main.drop_duplicates(subset='Molecule Name')
# print(main.shape)
# print(main.columns)

# Label Set
labels = pd.DataFrame(pd.read_csv('/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/CDD CSV Export - Labels.csv'))
labels = labels.drop(['CDD Number', 'pKa', 'pKa type', 'pKa (Acidic)', 'pKa (Basic)', 'Chemical formula', 'Isotope formula', 'Composition', 'Isotope composition', 'Batch Name', 'Batch Formula weight', 'Place', 'Vendor', 'Note', 'Date Received', 'Serial Number'], axis=1)
labels = labels.drop_duplicates(subset='Molecule Name')
# print(labels.shape)
# print(labels.columns)

merged = pd.merge(main, labels, on='Molecule Name', how='inner')
merged.dropna(axis=0, how='any', inplace=True)

merged.to_csv('/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv', index=False)

