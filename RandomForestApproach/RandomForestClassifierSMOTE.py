from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
#retrieving Cleaned Data
CleanedData = pd.read_csv('/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv')

#defining features vs labels
X = CleanedData.drop(columns=['LXRa', 'LXRb', 'Molecule Name'], axis=1)

# Making label Classification thresholds
# lxra_threshold = merged['LXRa'].mean() #+ merged['LXRa'].std()
# lxrb_threshold = merged['LXRb'].mean() #+ merged['LXRb'].std()
lxra_threshold = 80
lxrb_threshold = 50


CleanedData['LXRa'] = np.where(CleanedData['LXRa'] > lxra_threshold, 1, 0)
CleanedData['LXRb'] = np.where(CleanedData['LXRb'] < lxrb_threshold, 1, 0)
y = pd.merge(CleanedData['LXRa'], CleanedData['LXRb'], left_index=True, right_index=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# resampling using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# First dimension
X_res1, y_res1 = smote.fit_resample(X, y['LXRa'])
df_res1 = pd.DataFrame(X_res1)
df_res1['LXRa'] = y_res1
df_res1['is_synthetic1'] = df_res1.index >= len(X)

# Second dimension
X_res2, y_res2 = smote.fit_resample(X, y['LXRb'])
df_res2 = pd.DataFrame(X_res2)
df_res2['LXRb'] = y_res2
df_res2['is_synthetic2'] = df_res2.index >= len(X)

# Keep all original samples
final_df = CleanedData.copy()

# Add synthetic samples from first dimension
synthetic_df1 = df_res1[df_res1['is_synthetic1']].drop('is_synthetic1', axis=1)
synthetic_df1['LXRb'] = np.nan  # We'll fill these later

# Add synthetic samples from second dimension
synthetic_df2 = df_res2[df_res2['is_synthetic2']].drop('is_synthetic2', axis=1)
synthetic_df2['LXRa'] = np.nan  # We'll fill these later

# Combine all data
result = pd.concat([final_df, synthetic_df1, synthetic_df2], ignore_index=True)
result['LXRa'].fillna(result['LXRa'].mean(), inplace=True)
result['LXRb'].fillna(result['LXRb'].mean(), inplace=True)

# Separate features and labels
feature_cols = list(X.columns)
X_resampled = result[feature_cols].values
y_resampled = result[['LXRa', 'LXRb']].values

print(X_resampled)
print(y_resampled)

