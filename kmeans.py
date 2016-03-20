import random
import numpy as np
# from matplotlib import pyplot as pp

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
		d_cluster = np.sum(np.power(np.multiply(np.equal(result,i),(data-centroids[i])),2))
		cost += d_cluster
	return cost

# k = 5
m = 200
n = 2

X = np.array([[random.random() for i in range(n)] for i in range(m)])
X_norm = normalize(X)
# x1 = X_norm[:,0]
# x2 = X_norm[:,1]

all_cost = []
for k in range(1,10):
	for j in range(10):
		result = np.zeros((m,1))
		centroids = init_centroids(X_norm,k)
		current_cost = []
		for i in range(10):
			result = assign_centroids(X_norm,centroids,result)
			centroids = new_centroids(X_norm,centroids,result)
			# c1 = centroids[:,0]
			# c2 = centroids[:,1]
			# f = pp.figure()
			# f.hold(True)
			# pp.plot(x1,x2,'r+')
			# pp.plot(c1,c2,'bo')
			# f.hold(False)
			# f.show()
			# pp.pause(5)
			# pp.close(f)
			cost = cost_function(X_norm,centroids,result)
		current_cost = cost
	all_cost.append(current_cost)
