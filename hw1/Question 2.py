import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import mixture

def read_file(file_path):
	file = pd.read_csv(file_path)
	file = file.dropna()
	latitude = np.array(file.iloc[:,6])
	longtitude = np.array(file.iloc[:,7])
	price = np.array(file.iloc[:,9])
	# use min-max normalization here
	latitude = (latitude-min(latitude))/(max(latitude)-min(latitude))
	longtitude = (longtitude - min(longtitude))/(max(longtitude)-min(longtitude))
	price = (price-min(price))/(max(price)-min(price))
	data = np.vstack((latitude, longtitude, price)).transpose()
	return data
def decide_K_cluster_by_kmeans_plus(data):
	distortions = []
	for i in range(1, 11):
		km = KMeans(n_clusters=i, init='k-means++',random_state=0)
		km.fit(data)
		distortions.append(km.inertia_)
	plt.plot(range(1, 11), distortions, marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.show()
	# From plot, we decide number of clusters to be 5
	return 4
def cluster_by_kmeans_plus(data,K):
	km = KMeans(n_clusters=K, init='k-means++',random_state=0)
	km.fit(data)
	Y = km.predict(data)
	return Y
def decide_num_of_cluster_hierarchical(data):
	plt.figure(figsize=(10, 7))
	plt.title("Customer Dendograms")
	dend = shc.dendrogram(shc.linkage(data, method='ward'))
	plt.show()
	#From plot, we decide the number of clusters to be 5
def cluster_by_hierarchical(data,k):
	cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
	Y = cluster.fit_predict(data)
	return Y

def plot_aic_bic(data):
	n_components = np.arange(1, 10)
	models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(data) for n in n_components]
	plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
	plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
	plt.xlabel('n_components')
	plt.show()
	# From the plot, we decide number of clusters to be 5
def cluster_by_GMM(data,k):
	gmm = mixture.GaussianMixture(n_components=k).fit(data)
	labels = gmm.predict(data)
	return labels

def output_data(data,a,b,c):
	file = pd.read_csv(data)
	file = file.dropna()
	latitude = np.array(file.iloc[:,6])
	longtitude = np.array(file.iloc[:,7])
	price = np.array(file.iloc[:,9])
	data = np.vstack((latitude, longtitude, price)).transpose()
	new_data = np.vstack((a,b,c)).transpose()
	new_data = pd.DataFrame(np.concatenate((data, new_data), axis=1))
	new_data.to_csv("Question 2 Output.csv", index = False)
def main():
	data = read_file("listings.csv")
	#num_K = decide_K_cluster_by_kmeans_plus(data)
	Y1 = cluster_by_kmeans_plus(data,5)
	#decide_num_of_cluster_hierarchical(data)
	Y2 = cluster_by_hierarchical(data,5)
	#plot_aic_bic(data)
	Y3 = cluster_by_GMM(data,5)
	output_data("listings.csv", Y1,Y2,Y3)

if __name__ == '__main__':
    main()
