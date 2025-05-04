import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Path to the docking data and the agonist data
docking_pathA = '/Users/sambolton/Desktop/Cummins Lab/Molecular Docking/4EXS_VS/DOCKING/ranked_result.csv'
docking_dataB = '/Users/sambolton/Desktop/Cummins Lab/1PQ6_Docking/DOCKING/ranked_result.csv'
agonist_path = '/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/LXRAgonistDataUpdated.csv'


# Read the data into pandas dataframes
docking_data_A = pd.read_csv(docking_pathA)
docking_data_B = pd.read_csv(docking_dataB)

agonist_data = pd.read_csv(agonist_path)

docking_data_A['ID'] = docking_data_A['ID'].str.slice(0,-2)
docking_data_B['ID'] = docking_data_B['ID'].str.slice(0,-2)

# take absolute value of the docking scores
docking_data_A['SCORE'] = docking_data_A['SCORE'].abs()
docking_data_B['SCORE'] = docking_data_B['SCORE'].abs()

# Taking the natural logarithm of LXRa and LXRb activity
agonist_data['Data Summary: % LXRa 100nM (%)'] = np.log(agonist_data['Data Summary: % LXRa 100nM (%)'])
agonist_data['Data Summary: % LXRb 100nM (%)'] = np.log(agonist_data['Data Summary: % LXRb 100nM (%)'])

# rename the 'Molecule Name' column to 'ID' for both dataframes
agonist_data.rename(columns={'Molecule Name': 'ID'}, inplace=True)

print(docking_data_B.head())

merged_A = pd.merge(docking_data_A,agonist_data, on='ID', how='inner')
merged_B = pd.merge(docking_data_B,agonist_data, on='ID', how='inner')
# Select only the required columns

# Print the resulting dataframe
# print(filtered_merged.shape)

def linear(m, b, x):
    return m*x + b

# params, cov = curve_fit(linear, merged_A['SCORE'], merged_A['Data Summary: % LXRa 100nM (%)'], p0=[0.5, 0.75])
# print(params)


plt.plot(merged_A['SCORE'], merged_A['Data Summary: % LXRa 100nM (%)'], 'o')
# plt.plot(merged_A['SCORE'], linear(params[0], params[1], merged_A['SCORE']))
plt.xlabel('SCORE')
plt.ylabel('LXRa Agonist Activity (%)')
plt.title('Docking Score vs LXRa Agonist Activity')
plt.show()
plt.plot(merged_B['SCORE'], merged_B['Data Summary: % LXRb 100nM (%)'], 'o')
plt.xlabel('SCORE')
plt.ylabel('LXRa Agonist Activity (%)')
plt.title('Docking Score vs LXRb Agonist Activity')
plt.show()
