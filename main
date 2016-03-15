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

	def cluster(self,data,centroids,result):
		for i in range(len(data)):
			distance = -1
			k_class = -1
			for j in range(len(centroids)):
				new_distance = self.euclidean_distance(data[i],centroids[j])
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
		if new_centroids != centroids:
			centroids = new_centroids.copy()
			self.cluster(data,centroids,result)
		cen = new_centroids.copy()
		return cen,result

if __name__ == '__main__':
	km = KMeans()
	m = 100
	n = 10
	k = 3
	X = [[random.random() for i in range(n)] for i in range(m)]
	'''
	X = np.matrix(X)
	X = (X - X.mean(0))/X.std(0)
	X = np.array(X).reshape(m,n).tolist()
	'''
	Y = [[] for i in range(k)]

	centroids = km.init_centroids(X,k)
	k_centroids,output = km.cluster(X,centroids,Y)
	cost = km.cost_function(k_centroids,output,m)
'''
	for i in range(9):
		Y = [[] for i in range(k)]
		centroids = km.init_centroids(X,k)
		k_centroids_next,output_next = km.cluster(X,centroids,Y)
		cost_next = km.cost_function(k_centroids_next,output_next,m)
		if cost > cost_next:
			k_centroids = k_centroids_next.copy()
			output = output_next.copy()
			cost = cost_next
'''

