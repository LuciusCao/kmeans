import random
import numpy as np
# from matplotlib import pyplot as pp
'''using random library for shuffling data and also making fake data for test
using numpy for vectorization, and matplotlib for debugging algorithm'''

def init_centroids(data,k):
	'''Randomly choose k centroids.'''
	if not isinstance(data,list):
		data = data.tolist()
	data_copy = data.copy()
	centroids = []
	for i in range(k):
		random.shuffle(data_copy)
		centroids.append(data_copy.pop())
	centroids = np.array(centroids)
	return centroids

def normalize(data):
	'''normalize the data set so that all data will be on the same scale.'''
	if isinstance(data,list):
		data = np.array(data)
	data_mean = np.mean(data,0)
	data_stdev = np.std(data,0)
	data_norm = (data-data_mean)/data_stdev
	return data_norm

def assign_centroids(data,centroids,result):
	'''compare the distance and assign the data to the one 
	which is closest to the data set. (using Euclidean distance)'''
	for i in range(len(data)):
		distance = np.sum((np.power((data[i,:] - centroids), 2)),1)
		'''using vectorized calculation to simplifie code and calculation,
		and also speed up the algorithm.'''
		distance = distance.tolist()
		j = distance.index(min(distance))
		result[i] = j
	return result

def new_centroids(data,centroids,result):
	'''after assigning centroids to each row of the data set, calculate the 
	new centroids for each cluster. Again calculation are vectorized here.'''
	for i in range(len(centroids)):
		number = sum(np.equal(result,i))
		centroids[i] = (1/number)*sum(np.multiply(np.equal(result,i),data))
	return centroids

def cost_function(data,centroids,result):
	'''using Euclidean Distance as the optimization object.'''
	cost = 0
	for i in range(len(centroids)):
		d_cluster = np.sum(np.power(np.multiply(np.equal(result,i),(data-centroids[i])),2))
		cost += d_cluster
	return cost

'''Play around the data by changing the settings below'''
if __name__ == '__main__':
	k = 10 # number of clusters
	m = 2000 # number of examples (rows)
	n = 100 # number of features

	X = np.array([[random.random() for i in range(n)] for i in range(m)])
	X_norm = normalize(X)

	all_cost = []
	for k in range(1,10):
		for j in range(10):
			result = np.zeros((m,1))
			centroids = init_centroids(X_norm,k)
			current_cost = []
			for i in range(10):
				result = assign_centroids(X_norm,centroids,result)
				centroids = new_centroids(X_norm,centroids,result)
				cost = cost_function(X_norm,centroids,result)
			current_cost = cost
		all_cost.append(current_cost)
