### Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is a dimensionality reduction technique used in data analysis and machine learning. It transforms a dataset with many variables into a smaller set of uncorrelated variables called **principal components**, while retaining as much variance as possible.

---

### Key Objectives of PCA

1. **Dimensionality Reduction**:
   - Simplify datasets with many variables while preserving essential information.
   
2. **Feature Extraction**:
   - Identify the most significant features (components) that explain the variance in the data.

3. **Data Visualization**:
   - Reduce the data to 2D or 3D for visualization of high-dimensional datasets.

4. **Noise Reduction**:
   - Remove less significant components to reduce noise in the data.

---

### How PCA Works

1. **Standardize the Data**:
   - Scale the data so that each feature has a mean of 0 and a standard deviation of 1.
   - This ensures that PCA is not biased by the scale of features.

2. **Compute the Covariance Matrix**:
   - Identify relationships between the variables in the dataset.

3. **Calculate Eigenvalues and Eigenvectors**:
   - Compute the eigenvalues and eigenvectors of the covariance matrix to determine the directions (principal components) of maximum variance.

4. **Select Principal Components**:
   - Rank eigenvalues in descending order and select the top \(k\) components that explain the most variance.

5. **Transform the Data**:
   - Project the original data onto the selected principal components to create the reduced-dimension dataset.

---

### Mathematical Representation

1. **Covariance Matrix** (\(C\)):
   \[
   C = \frac{1}{n-1} (X^T X)
   \]
   Where \(X\) is the standardized data matrix.

2. **Eigenvalue Decomposition**:
   - Compute eigenvalues (\(\lambda\)) and eigenvectors (\(v\)):
     \[
     Cv = \lambda v
     \]

3. **Select Principal Components**:
   - Choose \(k\) eigenvectors corresponding to the largest \(k\) eigenvalues.

4. **Transform Data**:
   \[
   Z = X \cdot V_k
   \]
   Where \(Z\) is the reduced data, and \(V_k\) contains the top \(k\) eigenvectors.

---

### Choosing the Number of Components
- Use **explained variance** to decide the number of components.
- Plot a **scree plot** to visualize the cumulative explained variance:
  - Select \(k\) where the cumulative variance reaches a threshold (e.g., 90%).

---

### Example in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
data = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])

# Standardize data
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_std)

# Explained variance
print("Explained Variance Ratios:", pca.explained_variance_ratio_)
print("Reduced Data:\n", data_pca)

# Scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Plot')
plt.show()
```

---

### Applications of PCA

1. **Data Preprocessing**:
   - Remove redundant or correlated features.

2. **Image Compression**:
   - Reduce the number of pixels while preserving key features.

3. **Visualization**:
   - Project high-dimensional data into 2D or 3D.

4. **Noise Reduction**:
   - Focus on components that carry the most variance and discard noisy components.

5. **Clustering and Classification**:
   - Improve the performance of clustering or classification algorithms.

---

Compiled by UdithaWICK
