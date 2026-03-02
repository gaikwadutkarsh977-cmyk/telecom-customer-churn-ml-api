import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv")

print("Shape of dataset:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# Target distribution
if df.columns[-1] in df.columns:
    target = df.columns[-1]
    print("\nTarget Distribution:")
    print(df[target].value_counts())

    sns.countplot(x=df[target])
    plt.title("Churn Distribution")
    plt.show()

# Correlation heatmap (numeric only)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=False)
plt.title("Correlation Heatmap")
plt.show()