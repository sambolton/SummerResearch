import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Importing cleaned data
CleanedData = pd.read_csv('/Users/sambolton/Desktop/Cummins Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv')

# Separate features (X) and targets (y)
X = CleanedData.drop(columns=['LXRa', 'LXRb'])
y = CleanedData[['LXRa', 'LXRb']]

# Standardizing features
sd = X.std()
sd_df = pd.DataFrame([sd], columns=X.columns)
standardized_data = (X - X.mean()) / sd_df.values
standardized_data = standardized_data.dropna(axis=1)

# Store feature names after dropping NA columns
feature_names = standardized_data.columns

# Performing PCA
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(standardized_data)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio for the first 5 components:", explained_variance)

# Get the loadings (components)
loadings = pca.components_

# Create a DataFrame for the loadings
loadings_df = pd.DataFrame(
    loadings.T,  # Transpose to have features as rows
    index=feature_names,  # Original feature names
    columns=[f'PC{i+1}' for i in range(loadings.shape[0])]  # PC names
)

# Display the loadings
print("\nPCA Loadings (Component Matrix):")
print(loadings_df)

# Visualize the loadings with a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, cmap='coolwarm', center=0)
plt.title('PCA Component Loadings')
plt.tight_layout()
plt.show()

# More detailed view of the top contributors to each PC
def get_top_features(loadings_df, n=10):
    """Get the top n contributing features for each principal component"""
    top_features = {}
    
    for pc in loadings_df.columns:
        # Sort features by absolute loading values
        sorted_features = loadings_df[pc].abs().sort_values(ascending=False)
        # Get top n features
        top_n = sorted_features.head(n)
        # Store feature names and loading values
        top_features[pc] = pd.DataFrame({
            'Feature': top_n.index,
            'Loading': loadings_df.loc[top_n.index, pc].values,
            'Absolute Loading': top_n.values
        }).sort_values('Absolute Loading', ascending=False)
    
    return top_features

# Get top 10 contributing features for each PC
top_features = get_top_features(loadings_df, n=10)

# Display top features for each principal component
for pc, features in top_features.items():
    print(f"\nTop 10 features contributing to {pc}:")
    print(features[['Feature', 'Loading']])

# Visualize features with bar plots
for i in range(len(loadings_df.columns)):
    pc = loadings_df.columns[i]
    plt.figure(figsize=(10, 6))
    
    # Get data for plotting
    plot_data = top_features[pc].sort_values('Loading')
    
    # Plot horizontal bar chart
    bars = plt.barh(plot_data['Feature'], plot_data['Loading'], color=plt.cm.coolwarm(
        (plot_data['Loading'] - plot_data['Loading'].min()) / 
        (plot_data['Loading'].max() - plot_data['Loading'].min())
    ))
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'Top 10 Features Contributing to {pc} ({explained_variance[i] * 100:.2f}% Variance)')
    plt.xlabel('Loading Value')
    plt.tight_layout()
    plt.show()

