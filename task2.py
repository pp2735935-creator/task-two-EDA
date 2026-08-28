
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows:")
print(df.head())



print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

numeric_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

df[numeric_columns].hist(
    figsize=(14, 10),
    bins=20
)

plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()


numeric_df = df.select_dtypes(
    include=['int64', 'float64']
)

correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix Heatmap")
plt.show()


pairplot_columns = [
    'Survived',
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]

sns.pairplot(
    df[pairplot_columns],
    hue="Survived"
)

plt.show()


print("\nSkewness:")
print(numeric_df.skew())


plt.figure(figsize=(6, 4))

sns.countplot(
    x='Survived',
    data=df
)

plt.title("Survival Count")
plt.show()



plt.figure(figsize=(7, 5))

sns.countplot(
    x='Sex',
    hue='Survived',
    data=df
)

plt.title("Survival by Gender")
plt.show()



plt.figure(figsize=(7, 5))

sns.countplot(
    x='Pclass',
    hue='Survived',
    data=df
)

plt.title("Survival by Passenger Class")
plt.show()


# 

print("\nEDA Completed Successfully!")

