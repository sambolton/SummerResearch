import pandas as pd

# importing cleaned data
CleanedData = pd.read_csv('/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv')

# Standardizing the data
sd = CleanedData.std()
sd_df = pd.DataFrame([sd], columns=CleanedData.columns)

standardized_data = (CleanedData - CleanedData.mean()) / sd_df.values

from sklearn.decomposition import PCA
# Performing PCA


