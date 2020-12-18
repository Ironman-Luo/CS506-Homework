import k_means_clustering
import cv2
import numpy as np
img = cv2.imread("test.jpg",)
x,y,z = img.shape
data = []
print(img.shape)
for i in range(x):
	for j in range(y):
		data.append(img[i,j].tolist())
assignment, history = k_means_clustering.run_k_means(data, k_means_clustering.choose_random_centroids(data, K=10),5)
history = history[-1]
for i in range(len(data)):
	data[i] = history[assignment[i]]
data = np.array(data)
data = data.reshape((850,1280,3))
cv2.imwrite('Question 4 Output.png',data)

#cv2.imshow('Display Window', data) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()
