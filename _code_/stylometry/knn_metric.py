#!/usr/bin/env

from sklearn.neighbors import NearestNeighbors
import numpy as np 

def knn(vectors, authors, titles, neighbors):

	if type(vectors) == tuple:
		X_train, Y_train, x_test, y_test = vectors[0], vectors[1], vectors[2], vectors[3]
		
		# Reconstruct original X data, disregard the test_train_split that was made during vectorization
		vectors = np.append(X_train, x_test, axis=0)

		nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(vectors)
		distances, indices = nbrs.kneighbors(vectors)
	
		return distances, indices
		
	else:
		nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(vectors)
		distances, indices = nbrs.kneighbors(vectors)
	
		return distances, indices