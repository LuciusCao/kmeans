import random
import numpy as np

def init_centroids(data,k):
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
	if isinstance(data,list):
		data = np.array(data)
	data_mean = np.mean(data,0)
	data_stdev = np.std(data,0)
	data_norm = (data-data_mean)/data_stdev
	return data_norm

def assign_centroids(data,centroids,result):
	for i in range(len(data)):
		distance = np.sum((np.power((data[i,:] - centroids), 2)),1)
		distance = distance.tolist()
		j = distance.index(min(distance))
		result[i] = j
	return result

def new_centroids(data,centroids,result):
	for i in range(len(centroids)):
		number = sum(np.equal(result,i))
		centroids[i] = (1/number)*sum(np.multiply(np.equal(result,i),data))
	return centroids

def cost_function(data,centroids,result):
	cost = 0
	for i in range(len(centroids)):
		d_cluster = np.sum(np.power((np.multiply(np.equal(result,i),data)-centroids[i]),2))
		cost += d_cluster
	return cost

# k = 5
m = 500
n = 10

X = np.array([[random.random() for i in range(n)] for i in range(m)])
X_norm = normalize(X)

global_cost = []
for k in range(1,10):
	for j in range(10):
		result = np.zeros((m,1))
		centroids = init_centroids(X_norm,k)
		local_cost = []
		for i in range(10):
			result = assign_centroids(X_norm,centroids,result)
			centroids = new_centroids(X_norm,centroids,result)
			local_cost.append(cost_function(X_norm,centroids,result))
	global_cost.append(min(local_cost))

