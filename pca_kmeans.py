import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def perform_pca(data, n_components):
  """
  Performs Principal Component Analysis (PCA) on the data.

  Args:
      data: A numpy array of data points.
      n_components: The number of principal components to retain.

  Returns:
      A tuple containing:
          - Transformed data using PCA
          - PCA object
  """
  pca = PCA(n_components=n_components)
  pca_data = pca.fit_transform(data)
  return pca_data, pca

def perform_clustering(data, n_clusters):
  """
  Performs K-Means clustering on the data.

  Args:
      data: A numpy array of data points.
      n_clusters: The number of clusters to create.

  Returns:
      A tuple containing:
          - Cluster labels for each data point
          - KMeans object
  """
  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(data)
  return kmeans.labels_, kmeans

# Sample dataset (replace with your actual data)
data = np.array([[1, 2, 1.5],
                 [3, 4, 2.5],
                 [9, 1, 0.8],
                 [7, 2, 1.2],
                 [4, 0, 3.0]])

# PCA with 2 components
pca_data, pca_obj = perform_pca(data, 2)

# K-Means clustering with original data (3 clusters)
original_labels, kmeans_original = perform_clustering(data, 3)

# K-Means clustering with PCA data (3 clusters)
pca_labels, kmeans_pca = perform_clustering(pca_data, 3)

# Print results (modify to fit your data analysis)
print("Original data cluster labels:", original_labels)
print("PCA data cluster labels:", pca_labels)

# Additional analysis (e.g., visualize clusters using the original features)
# You can use libraries like matplotlib or seaborn to plot the clusters in the original feature space
# This can help you understand how PCA might have affected the clustering results visually.
