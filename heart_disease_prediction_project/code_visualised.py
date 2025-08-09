import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/umern/OneDrive/Desktop/DEvelopers hub/Health Disease Prediction/heart_disease_uci.csv")

# Checking missing values before cleaning

print("Missing values before cleaning:")
for column in df.columns:
    missing_count = df[column].isnull().sum()
    print(f"{column}: {missing_count}")


# Filling missing values (Median for numeric, Mode for categorical)

for column in df.columns:
    if df[column].dtype == "float64" or df[column].dtype == "int64":
        non_null_values = df[column].dropna()
        sorted_values = sorted(non_null_values)
        n = len(sorted_values)

        if n % 2 == 1:
            median_value = sorted_values[n // 2]
        else:
            median_value = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

        for i in range(len(df[column])):
            if pd.isnull(df[column][i]):
                df.at[i, column] = median_value
    else:
        value_counts = {}
        for val in df[column]:
            if pd.isnull(val):
                continue
            if val in value_counts:
                value_counts[val] += 1
            else:
                value_counts[val] = 1
        
        mode_value = max(value_counts, key=value_counts.get)

        for i in range(len(df[column])):
            if pd.isnull(df[column][i]):
                df.at[i, column] = mode_value


print("\nMissing values after cleaning:")
for column in df.columns:
    missing_count = df[column].isnull().sum()
    print(f"{column}: {missing_count}")


df.to_csv("C:/Users/umern/OneDrive/Desktop/DEvelopers hub/Health Disease Prediction/heart_disease_uci_cleaned.csv", index=False)

# EDA

# Basic statistics

print("\nBasic Statistics:")
print(df.describe())

# Distribution of target variable

plt.figure(figsize=(6,4))
sns.countplot(x='num', data=df)
plt.title("Heart Disease Cases (0 = No Disease, 1-4 = Disease Levels)")
fig = plt.gcf()  # get current figure
fig.canvas.manager.set_window_title("Distribution of target variable")
plt.show()

# Age distribution

plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
fig = plt.gcf() 
fig.canvas.manager.set_window_title("Age Distribution")
plt.show()

# Gender vs Heart Disease

plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='num', data=df)
plt.title("Heart Disease by Gender")
fig = plt.gcf()  
fig.canvas.manager.set_window_title("Heart Disease by Gender")
plt.show()

# Correlation heatmap (numeric columns only)

plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=['number']) 
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
fig = plt.gcf()
fig.canvas.manager.set_window_title("Correlation Heatmap")
plt.show()
