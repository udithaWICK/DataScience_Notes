Here are some **Python codes** for analyzing and visualizing **data distributions** using libraries like **NumPy**, **Pandas**, **Matplotlib**, and **Seaborn**:

---

## **1. Import Required Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## **2. Generate Sample Data**
```python
# Generate random data
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)  # Normal distribution
```

---

## **3. Visualize the Distribution**

### **a. Histogram**
```python
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Data')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

---

### **b. Density Plot (KDE Plot)**
```python
sns.kdeplot(data, shade=True, color='blue')
plt.title('Density Plot')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
```

---

### **c. Boxplot**
```python
sns.boxplot(data, orient='h', color='orange')
plt.title('Boxplot of Data')
plt.show()
```

---

### **d. Violin Plot**
```python
sns.violinplot(data, color='purple')
plt.title('Violin Plot of Data')
plt.show()
```

---

### **e. QQ Plot (Normality Check)**
```python
import scipy.stats as stats
import pylab

stats.probplot(data, dist="norm", plot=pylab)
pylab.title('QQ Plot')
pylab.show()
```

---

## **4. Statistical Measures**
```python
# Descriptive Statistics
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0][0]
std_dev = np.std(data)
variance = np.var(data)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")
```

---

## **5. Handling Skewed Data (Transformations)**
```python
# Generate right-skewed data
skewed_data = np.random.exponential(size=1000)

# Log Transformation
log_transformed = np.log(skewed_data)

# Square Root Transformation
sqrt_transformed = np.sqrt(skewed_data)

# Visualize Transformations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(skewed_data, ax=axes[0], kde=True, color='red')
axes[0].set_title('Original Data')

sns.histplot(log_transformed, ax=axes[1], kde=True, color='green')
axes[1].set_title('Log Transformed')

sns.histplot(sqrt_transformed, ax=axes[2], kde=True, color='blue')
axes[2].set_title('Square Root Transformed')

plt.tight_layout()
plt.show()
```

---

## **6. Comparing Distributions with Subplots**
```python
# Generate multiple distributions
data1 = np.random.normal(0, 1, 1000)    # Normal Distribution
data2 = np.random.exponential(1, 1000) # Exponential Distribution
data3 = np.random.uniform(0, 1, 1000)  # Uniform Distribution

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(data1, ax=axes[0], kde=True, color='blue')
axes[0].set_title('Normal Distribution')

sns.histplot(data2, ax=axes[1], kde=True, color='green')
axes[1].set_title('Exponential Distribution')

sns.histplot(data3, ax=axes[2], kde=True, color='orange')
axes[2].set_title('Uniform Distribution')

plt.tight_layout()
plt.show()
```

---

## **7. Testing Normality (Shapiro-Wilk Test)**
```python
from scipy.stats import shapiro

stat, p = shapiro(data)
print(f"Shapiro-Wilk Test: Stat={stat}, p={p}")

if p > 0.05:
    print("Data looks normally distributed.")
else:
    print("Data does not look normally distributed.")
```

---

These Python scripts can be directly copied into a Jupyter Notebook or Python IDE for execution.
