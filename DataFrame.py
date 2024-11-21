import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df)

print("\nFirst few rows:")
print(df.head())

print("\nLast few rows:")
print(df.tail())

print("\nDataFrame Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nSelecting 'Name' column:")
print(df['Name'])

print("\nSelecting 'Name' and 'Age' columns:")
print(df[['Name', 'Age']])

print("\nRows where Age > 30:")
print(df[df['Age'] > 30])

df['Salary'] = [50000, 60000, 70000]
print("\nDataFrame after adding 'Salary' column:")
print(df)

df['Age'] = df['Age'] + 1
print("\nDataFrame after modifying 'Age' column:")
print(df)

df = df.drop(columns=['City'])
print("\nDataFrame after dropping 'City' column:")
print(df)

df = df.drop(index=[0])
print("\nDataFrame after dropping first row:")
print(df)

df = df.sort_values(by='Age', ascending=False)
print("\nDataFrame after sorting by 'Age':")
print(df)

group_data = {'Category': ['A', 'A', 'B', 'B'], 'Values': [10, 20, 30, 40]}
group_df = pd.DataFrame(group_data)
grouped = group_df.groupby('Category').sum()
print("\nGrouped Data:")
print(grouped)

missing_data = {
    'Name': ['Alice', None, 'Charlie'],
    'Age': [25, None, 35]
}
missing_df = pd.DataFrame(missing_data)
print("\nDataFrame with missing values:")
print(missing_df)

filled_df = missing_df.fillna(0)
print("\nDataFrame after filling missing values with 0:")
print(filled_df)

dropped_df = missing_df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(dropped_df)

df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 3], 'Age': [25, 30]})
merged = pd.merge(df1, df2, on='ID', how='inner')
print("\nMerged DataFrame:")
print(merged)

df.to_csv('output.csv', index=False)
print("\nDataFrame exported to 'output.csv'")
