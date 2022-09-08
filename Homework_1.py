# importing libraries

import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/home/osboxes/Downloads/iris.csv")
df1 = pd.DataFrame(data)
df = pd.DataFrame(data)
df.pop("class")

n_array = df.to_numpy()

print(n_array)

mean_value = n_array.mean(axis=0)
min_value = n_array.min(axis=0)
max_value = n_array.max(axis=0)

print("mean = ", mean_value)
print("min = ", min_value)
print("max = ", max_value)


Histogram = px.histogram(df, x="petalLength")
Histogram.show()

scatter_plot = px.scatter(df, x="sepalLength", y="sepalWidth")
scatter_plot.show()

violin_plot = px.violin(df1, x="sepalLength", color="class")
violin_plot.show()

x = df.iloc[:, [0, 1, 2, 3]]
x.head()

scaler = StandardScaler().fit(x)
print(scaler)

x_scaled = scaler.transform(x)
print(x_scaled)

print(df1["class"])
