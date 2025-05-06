import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PCACleanedData import X_reduced, y


# Extract PC1 and PC2 values
pc1_values = X_reduced[:, 0]
pc2_values = X_reduced[:, 1]

# Calculate LXRa - LXRb
lxr_diff = y['LXRa'].values - y['LXRb'].values

# Create a figure for the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set the Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Create weights based on the absolute difference
abs_diff = np.abs(lxr_diff)
weights = np.zeros_like(abs_diff)

# Linear Weighting on Absolute Difference
weights = abs_diff.copy()  # Linear weighting

# threshold get weights
threshold = 20
weights = np.zeros_like(abs_diff)
weights[abs_diff > threshold] = abs_diff[abs_diff > threshold]

# Normalize weights to sum to 1 (optional)
weights = weights / weights.sum()

# Create a mask for visualization (points with difference > threshold)
high_diff_mask = abs_diff > threshold
low_diff_mask = ~high_diff_mask

# Plot points with low difference
ax.scatter(pc1_values[low_diff_mask], 
           pc2_values[low_diff_mask], 
           lxr_diff[low_diff_mask],
           color='gray', 
           alpha=0.5, 
           s=50, 
           label=f'|LXRa - LXRb| ≤ {threshold}')

# Plot points with high difference, with size proportional to weight
scatter = ax.scatter(pc1_values[high_diff_mask], 
                    pc2_values[high_diff_mask], 
                    lxr_diff[high_diff_mask],
                    c=lxr_diff[high_diff_mask], 
                    cmap='viridis', 
                    s=80 + 300 * weights[high_diff_mask] / weights[high_diff_mask].max(), 
                    alpha=0.8,
                    edgecolor='k',
                    linewidth=0.5,
                    label=f'|LXRa - LXRb| > {threshold}')

# Perform weighted least squares for the plane
if np.sum(weights > 0) > 3:  # Need at least 3 points with non-zero weights
    # Create a design matrix for all points
    X_plane = np.column_stack((pc1_values, pc2_values, np.ones_like(pc1_values)))
    
    # Use weights for weighted least squares
    from scipy.linalg import lstsq
    
    # We'll use the square root of weights because in the normal equations weights appear squared
    sqrt_weights = np.sqrt(weights)
    
    # Weight both the design matrix and the target
    weighted_X = X_plane * sqrt_weights[:, np.newaxis]
    weighted_y = lxr_diff * sqrt_weights
    
    # Solve the weighted least squares problem
    A, residuals, rank, s = lstsq(weighted_X, weighted_y)
    
    # Create a mesh grid for the plane
    x_min, x_max = pc1_values.min(), pc1_values.max()
    y_min, y_max = pc2_values.min(), pc2_values.max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                         np.linspace(y_min, y_max, 20))
    
    # Calculate z values for the plane using the equation z = a*x + b*y + c
    zz = A[0] * xx + A[1] * yy + A[2]
    
    # Constraining Z values
    zz = np.clip(zz, -100, 100)
    
    # Plot the plane
    plane = ax.plot_surface(xx, yy, zz, alpha=0.3, color='red', 
                           label='Weighted best fit plane')
    
    # Add information about the plane equation
    plane_eq = f'LXR|a - b| = {A[0]:.3f}*PC1 + {A[1]:.3f}*PC2 + {A[2]:.3f}'
    ax.text(x_min, y_max, zz.max(), plane_eq, color='red', fontsize=12)

# Add a color bar
cbar = plt.colorbar(scatter)
cbar.set_label('LXRa - LXRb Difference', fontsize=14)

# Set labels for the axes
ax.set_xlabel('Principal Component 1', fontsize=14)
ax.set_ylabel('Principal Component 2', fontsize=14)
ax.set_zlabel('LXR |a - b|', fontsize=14)

# Add a title
plt.title('PC1 vs PC2 vs LXR|a-b| with Weighted Fit Plane', fontsize=16)

# Add a legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, alpha=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
#ax.legend(custom_lines, [f'|LXRa - LXRb| ≤ {threshold}', f'|LXRa - LXRb| > {threshold}'], loc='upper left')

# Add text explaining the weighting
ax.text2D(0.05, 0.95, f"Weighted plane fit (weight ∝ |LXRa - LXRb|)\n{np.sum(weights > 0)} points used for fitting", transform=ax.transAxes, fontsize=12)
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()