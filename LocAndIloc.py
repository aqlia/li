import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

print("\n--- .loc Examples ---")

print("\nRow with index 0 using .loc:")
print(df.loc[0])

print("\nRows 0 and 2, columns 'Name' and 'City':")
print(df.loc[[0, 2], ['Name', 'City']])

print("\nAll rows, 'Age' column:")
print(df.loc[:, 'Age'])

print("\nNames of people older than 30 using .loc:")
print(df.loc[df['Age'] > 30, 'Name'])

print("\nUpdating Age for index 1 using .loc:")
df.loc[1, 'Age'] = 31
print(df)

print("\n--- .iloc Examples ---")

print("\nFirst row using .iloc:")
print(df.iloc[0])

print("\nRows 0 and 2, columns 0 ('Name') and 2 ('City'):")
print(df.iloc[[0, 2], [0, 2]])

print("\nAll rows, second column ('Age') using .iloc:")
print(df.iloc[:, 1])

print("\nFirst two rows and first two columns using .iloc:")
print(df.iloc[0:2, 0:2])

print("\nUpdating value in second row, second column using .iloc:")
df.iloc[1, 1] = 32
print(df)

print("\n--- Combining .loc and .iloc ---")
filtered_data = df.loc[df['Age'] > 30].iloc[:, :2]
print("\nFiltered data (Age > 30), first two columns:")
print(filtered_data)
