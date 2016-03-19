import numpy as np
import random

class KMeans():
	def init_centroids(self,data,k):
		centroids = []
		if len(data)<k:
			print('error')
		else:
			for i in range(k):
				copy = data.copy()
				random.shuffle(copy)
				centroids.append([round(w,5) for w in copy.pop()])
		return centroids

	def euclidean_distance(self,list1,list2):
		list1 = np.matrix(list1)
		list2 = np.matrix(list2)
		distance = list1 - list2
		distance = (distance.A ** 2).sum()
		return distance

	def cost_function(self,centroids,result,m):
		cost = 0
		for i in range(len(result)):
			for j in range(len(result[i])):
				d = self.euclidean_distance(centroids[i],result[i][j])
				cost += d
		cost = cost / m
		return cost

	def cluster(self,data,data_norm,centroids,cen_norm,result):
		global X_mean,X_std
		for i in range(len(data)):
			distance = -1
			k_class = -1
			for j in range(len(centroids)):
				new_distance = self.euclidean_distance(data_norm[i],cen_norm[j])
				if distance == -1:
					distance = new_distance
					k_class = j
				elif distance > new_distance:
					distance = new_distance
					k_class = j
			result[k_class].append([round(w,5) for w in data[i]])
		new_centroids = centroids.copy()
		for i in range(k):
			result_i = np.array(result[i])
			mean = list(result_i.mean(0))
			new_centroids[i] = [round(w,5) for w in mean]
		# if new_centroids != centroids:
		# 	centroids = new_centroids.copy()
		# 	self.cluster(data,centroids,result)
		return new_centroids,result

if __name__ == '__main__':
	km = KMeans()
	m = 100
	n = 10
	k = 3
	X = [[random.random() for i in range(n)] for i in range(m)]

	X_norm = np.matrix(X)
	X_std = X_norm.std(0)
	X_mean = X_norm.mean(0)
	X_norm = (X_norm - X_mean)/X_std
	centroids = km.init_centroids(X,k)
	C_norm = np.matrix(centroids)
	C_norm = (C_norm - X_mean)/X_std

	X_norm = np.array(X_norm).reshape(m,n).tolist()
	C_norm = np.array(C_norm).reshape(k,n).tolist()

	for i in range(100):
		Y = [[] for i in range(k)]
		centroids,result = km.cluster(X,X_norm,centroids,C_norm,Y)
	cost = km.cost_function(centroids,Y,m)

