import numpy as np
import pandas as pd
import folium 
from folium import plugins
import matplotlib.pyplot as plt

def generateBaseMap(default_location=[40.693943, -73.985880]):
	base_map = folium.Map(location=default_location) 
	return base_map
base_map = generateBaseMap()
df = pd.read_csv("listings.csv")
df.dropna()
plugins.HeatMap(data=df[[ 'latitude' , 'longitude' , 'price']].groupby ([ 'latitude' , 'longitude' ]).mean().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map) 
base_map.save('Question 3a.html')

#2.
df = pd.read_csv("Question 2 Output.csv")
df.columns = ['latitude', 'longitude','price','kmeans','hierarchical','GMM']
def plot_data(data, form):
	x = df['longitude']
	y = df['latitude']
	z = df['price']
	marker = df[form]
	max_num = max(marker)
	min_num = min(marker)
	color = ['red','blue','green','yellow','black']
	for i in range(int(min_num), int(max_num)+1):
		new_x = [x[j] for j in range(len(x)) if int(marker[j]) == i]
		new_y = [y[j] for j in range(len(y)) if int(marker[j]) == i]
		print("avg price for cluster", i, np.mean([z[j] for j in range(len(z)) if int(marker[j]) == i]))
		plt.scatter(new_x,new_y,color = color[i])
	plt.show()

plot_data(df,'kmeans')
print("---")
plot_data(df,'hierarchical')
print("---")
plot_data(df, 'GMM')


#extra credit
def plot_data_on_Map(data, form, name):
	x = df['longitude']
	y = df['latitude']
	z = df['price']
	marker = df[form]
	max_num = max(marker)
	min_num = min(marker)
	base_map = generateBaseMap()
	color = ['red','blue','green','yellow','black']
	for i in range(int(min_num), int(max_num)+1):
		new_x = [x[j] for j in range(len(x)) if int(marker[j]) == i]
		new_y = [y[j] for j in range(len(y)) if int(marker[j]) == i]
		for index in range(len(new_x)):
			folium.CircleMarker(location = [new_y[index], new_x[index]],color = color[i], fill_color= color[i]).add_to(base_map)
	base_map.save(name)

plot_data_on_Map(df,'kmeans','Question 3d KMeans.html')
plot_data_on_Map(df,'hierarchical','Question 3d Hierarchical.html')
plot_data_on_Map(df,'GMM','Question 3d GMM.html')

