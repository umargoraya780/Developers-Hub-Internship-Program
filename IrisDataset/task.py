# Exploring and Visualizing a Dataset (Iris)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Load the dataset from CSV
# -----------------------------
# Replace path with your actual file location
iris = pd.read_csv(r"C:\Users\umern\OneDrive\Desktop\DEvelopers hub\IrisDataset\Iris.csv")

# -----------------------------
# Rename columns to lowercase with underscores
# -----------------------------
iris.columns = [col.strip().lower().replace(" ", "_") for col in iris.columns]

# If dataset has 'Id' column, drop it
if 'id' in iris.columns:
    iris.drop(columns='id', inplace=True)

# -----------------------------
# Inspecting the dataset
# -----------------------------
print("Shape of dataset:", iris.shape)
print("\nColumn names:", iris.columns.tolist())
print("\nFirst five rows:\n", iris.head())

print("\nDataset info:")
print(iris.info())

print("\nStatistical summary:")
print(iris.describe())

# -----------------------------
# Visualization
# -----------------------------

# Scatter plot - relationship between sepal length and petal length
plt.figure(num="Iris Scatter Plot", figsize=(6, 4))
sns.scatterplot(data=iris, x='sepallengthcm', y='petallengthcm', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.show()

# Histograms - distribution of each feature
plt.figure(num="Iris Histograms")
iris.drop(columns='species').hist(figsize=(8, 6), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Feature Distributions', y=1.02)
plt.show()

# Box plots - identify outliers
plt.figure(num="Iris Box Plots", figsize=(8, 6))
sns.boxplot(data=iris.drop(columns='species'))
plt.title('Box Plots for Iris Features')
plt.show()
