import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Importing cleaned data
CleanedData = pd.read_csv('/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD CSV Export - CleanedData.csv')

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

# Set the Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Create a figure for the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Extract PC1 values
pc1_values = X_reduced[:, 0]  # First column of X_reduced is PC1
pc2_values = X_reduced[:, 1]  # Second column of X_reduced is PC2 (for PC1 and PC2 vs Delta LXR plot)

# Get LXRa and LXRb values (for PC1 vs LXRa and LXRb plot)
# lxra_values = y['LXRa'].values
# lxrb_values = y['LXRb'].values

# Get LXRa - LXRb values (for PC1 and PC2 vs Delta LXR plot)
lxr_values = y['LXRa'].values - y['LXRb'].values


# Create a color palette using Seaborn
palette = sns.color_palette("viridis", as_cmap=True)

# Create the scatter plot
scatter = ax.scatter(pc1_values, pc2_values, lxr_values, 
                    c=pc1_values,  # Color points by PC1 value
                    cmap=palette, 
                    s=80,  # Point size
                    alpha=0.8,  # Transparency
                    edgecolor='w',  # White edge around points
                    linewidth=0.5)  # Edge thickness

# Add a color bar to show the mapping of colors to PC1 values
cbar = plt.colorbar(scatter)
cbar.set_label('PC1 Value', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Set labels for the axes with Seaborn styling
ax.set_xlabel('Principal Component 1', fontsize=14, labelpad=10)
ax.set_ylabel('Principal Component 2', fontsize=14, labelpad=10)
ax.set_zlabel('\Delta LXR', fontsize=14, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a title with Seaborn styling
plt.title('3D Visualization of PC1 and PC2 vs \Delta LXR Values', fontsize=16, pad=20)

# Add grid lines for better depth perception
ax.grid(True, alpha=0.3)

# Improve the view angle
ax.view_init(elev=25, azim=35)


# Show the plot
plt.tight_layout()
plt.show()