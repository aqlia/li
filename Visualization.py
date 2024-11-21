import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud as wc
import plotly.express as px

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.plot(x, y, marker='o', color='b', label="Prime Numbers")
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 8, 7]
plt.scatter(x, y, color='red', label="Points")
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

categories = ['A', 'B', 'C', 'D']
values = [3, 7, 8, 5]
plt.bar(categories, values, color='orange')
plt.title("Bar Chart")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

data = np.random.randn(1000)
plt.hist(data, bins=30, color='purple', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

data = [7, 8, 9, 10, 10, 11, 15, 19, 21]
sns.boxplot(data=data)
plt.title("Box Plot")
plt.show()

data = [7, 8, 9, 10, 10, 11, 15, 19, 21]
sns.violinplot(data=data, color='cyan')
plt.title("Violin Plot")
plt.show()

data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 6]
sns.kdeplot(data, fill=True, color="green")
plt.title("KDE Plot")
plt.show()

labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gold', 'lightblue', 'pink', 'lightgreen'])
plt.title("Pie Chart")
plt.show()

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gold', 'lightblue', 'pink', 'lightgreen'])
plt.gca().add_artist(plt.Circle((0, 0), 0.5, color='white'))
plt.title("Donut Chart")
plt.show()

data = np.random.rand(5, 5)
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title("Heatmap")
plt.show()

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
sns.pairplot(df)
plt.show()

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
sizes = [100, 200, 300, 400, 500]
plt.scatter(x, y, s=sizes, alpha=0.5, color='blue')
plt.title("Bubble Chart")
plt.show()

text = "Python Data Visualization Machine Learning AI"
wordcloud = wc(width=800, height=400, background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud")
plt.show()

data = {'Date': pd.date_range(start='2024-01-01', periods=10), 'Value': [10, 15, 20, 18, 25, 30, 35, 40, 45, 50]}
df = pd.DataFrame(data)
plt.plot(df['Date'], df['Value'], marker='o')
plt.title("Time Series Plot")
plt.xlabel("Date")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.show()

df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
fig.show()

data = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Gender": ["Female", "Male", "Male"],
    "Score": [85, 90, 95]
})
numeric_data = data.select_dtypes(include=["number"])
corr_matrix = numeric_data.corr()
fig = px.imshow(corr_matrix, color_continuous_scale="Viridis", title="Correlation Heatmap")
fig.show()

