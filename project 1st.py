import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = sns.load_dataset("iris")

df = pd.DataFrame(data)
df2 = df.to_csv("new_data.csv" , index=False)
print("File saved Successfully as csv")

new_data = pd.read_csv("new_data.csv")

# 150 rows and 5 columns
print("Its shape is ",new_data.shape)

print("Number of missing values\n" , new_data.isnull().sum())

print("Summary\n" , new_data.describe())

plt.plot(new_data.iloc[: , 0] , new_data.iloc[: , 1]) # METHOD 1
# plt.plot(new_data["sepal_length"] , new_data["sepal_width"]) # METHOD 2
plt.title("Visualizing")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

sns.heatmap(new_data.drop(columns="species").corr() , cmap="viridis")
plt.show()