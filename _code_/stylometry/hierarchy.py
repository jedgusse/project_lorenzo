#!/usr/bin/env

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.neighbors import NearestNeighbors

def dendrogram(vectors, authors, titles, features):

	# We start at the bottom of the tree, where all nodes still represent the original leaf nodes; in each subsequent iteration, we merge one
	# node with another node, until eventually, all nodes have been merged. Note that in each step, the
	# number of nodes is therefore reduced, because of the merging operation.

	# Parameter that determines where the nodes should change colors
	# This modest level of supervision is of course perfectly fine as long as you report it
	# Although Joseph Rudman and Maciej Eder have warned about "cherry picking"

	color_threshold = 680

	distance_matrix = squareform(pdist(vectors, 'cityblock'))
	linkage_object = linkage(distance_matrix, method='single', metric='cityblock')
	d = dendrogram(linkage_object, labels=titles, orientation='top', color_threshold=color_threshold)
	
	plt.show()

def heatmap(vectors, authors, titles, features):

	# The heatmap is a visualization of the symmetrical distance matrix of the frequency vectors
	# Starker colors represents larger distance

	distance_matrix = squareform(pdist(vectors, 'cityblock'))

	fig, ax = plt.subplots()
	heatmap = ax.pcolor(distance_matrix, cmap=plt.cm.Reds)
	ax.set_xticks(np.arange(distance_matrix.shape[0])+0.5, minor=False)
	ax.set_yticks(np.arange(distance_matrix.shape[1])+0.5, minor=False)
	ax.set_xticklabels(titles, minor=False, rotation=90)
	ax.set_yticklabels(authors, minor=False)
	plt.show()

def plot_frequencies(authors, titles, counts, features, feat_amount, plotname):

	# Output a plot to plot.ly that allows to analyze the frequencies of the function words
	# Login credentials to plot.ly

	#tls.set_credentials_file(username='jedgusse', api_key='td4zs19w0a')
	tls.set_credentials_file(username='jedgusse.degussem', api_key='qn9k3mb0ic')

	pca = PCA(n_components=2)
	X_bar = pca.fit_transform(counts)
	var_exp = pca.explained_variance_ratio_
	comps = pca.components_
	comps = comps.transpose()
	loadings = pca.components_.transpose()
	vocab_weights_p1 = sorted(zip(features, comps[:,0]), key=lambda tup: tup[1], reverse=True)
	vocab_weights_p2 = sorted(zip(features, comps[:,1]), key=lambda tup: tup[1], reverse=True)

	#weightiest_loadings = [each[0] for each in vocab_weights_p1[:5]] + [each[0] for each in vocab_weights_p1[-5:]]

	# Or add custom words which need separate observation to the weightiest_loadings list

	weightiest_loadings = ["vester", "vos", "tuus", "tu", "noster", "nos", "meus", "ego"]

	# Managing data
	# Include ALL word frequencies:

	z_list = []
	for sample_counts in counts:
		a_list = []
		for feature, raw_count in zip(features, sample_counts):
			if feature in weightiest_loadings:
				a_list.append((feature, raw_count))
		z_list.append(a_list)

	data = []	
	for feature, i in zip([each_tuple[0] for each_tuple in z_list[0]], range(len(weightiest_loadings))):
		trace = go.Scatter(
			x = [each for each in titles],
			y = [each_list[i][1] for each_list in z_list],
			name = '{}'.format(feature),
			line = dict(width = 1))
		data.append(trace)

	py.iplot(data, filename='{}'.format(plotname))

def plot_loadings(authors, titles, raw_counts, normalized_counts, features, feat_amount, plotname):
	# Output a plot to plot.ly that gives allows to analyze the frequencies of the function words
	# Login credentials to plot.ly

	#tls.set_credentials_file(username='jedgusse', api_key='td4zs19w0a')
	tls.set_credentials_file(username='jedgusse.degussem', api_key='qn9k3mb0ic')

	pca = PCA(n_components=2)
	X_bar = pca.fit_transform(normalized_counts)
	var_exp = pca.explained_variance_ratio_
	comps = pca.components_
	comps = comps.transpose()
	loadings = pca.components_.transpose()
	vocab_weights_p1 = sorted(zip(features, comps[:,0]), key=lambda tup: tup[1], reverse=True)
	vocab_weights_p2 = sorted(zip(features, comps[:,1]), key=lambda tup: tup[1], reverse=True)

	#weightiest_loadings = [each[0] for each in vocab_weights_p1[:5]] + [each[0] for each in vocab_weights_p1[-5:]] + [each[0] for each in vocab_weights_p2[:2]] + [each[0] for each in vocab_weights_p2[-2:]]

	# Or add custom words which need separate observation to the weightiest_loadings list

	weightiest_loadings = ["vester", "vos", "tuus", "tu", "noster", "nos", "meus", "ego"]

	# Managing data
	# Include ALL word frequencies:

	z_list = []
	for sample_counts in normalized_counts:
		a_list = []
		for feature, raw_count in zip(features, sample_counts):
			if feature in weightiest_loadings:
				a_list.append((feature, raw_count))
		z_list.append(a_list)
	
	data = []	
	for feature, i in zip([each_tuple[0] for each_tuple in z_list[0]], range(len(weightiest_loadings))):
		trace = go.Scatter(
			x = [each for each in titles],
			y = [each_list[i][1] for each_list in z_list],
			name = '{}'.format(feature),
			line = dict(width = 1))
		data.append(trace)

	py.iplot(data, filename='{}'.format(plotname))

def gephi_networks(authors, titles, vectors, neighbors):

	# We calculate the distances, but we weight them, which is different from the knn_metric.py document

	nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(vectors)
	distances, indices = nbrs.kneighbors(vectors)

	# Here we make the tab-separated txt files which should go into GEPHI; we instantiate a file object.
	# Calculates Nearest Neighbours of each txt sample.
	# Outputs distances

	fob_nodes = open("/Users/jedgusse/stylofactory/output/gephi_output/gephi_nodes.txt", "w")
	fob_edges = open("/Users/jedgusse/stylofactory/output/gephi_output/gephi_edges.txt", "w")

	fob_nodes.write("Id" + "\t" + "Work" + "\t" + "Author" + "\n")
	fob_edges.write("Source" + "\t" + "Target" + "\t" + "Type" + "\t" + "Weight" + "\n")

	# We normalize the calculated distances to a (0,1) range. However, smaller distances should come with a greater number, since they should
	# receive more weight. That's why we normalize to a range where the highest value is the "lowest" (argmin), and the lowest is the "highest" (argmax).
	# e.g. not (12,18) to (0,1), but (18,12) to (0,1), where higher numbers start at zero and lower numbers reach to 1.0

	all_distances = []
	for distance_vector in distances:
		for value in distance_vector:
			if value != 0.0:
				all_distances.append(value)

	all_distances = np.array(all_distances)
	highest_value = all_distances[np.argmin(all_distances)]
	lowest_value = all_distances[np.argmax(all_distances)]

	distances = (distances - lowest_value) / (highest_value - lowest_value)

	# Group our nearest texts:

	nearest_texts = []

	for dist_lst, ind_lst in zip(distances, indices):
		distances_persample = ()
		for num in range(1, neighbors):
			distances_persample = distances_persample + ((titles[ind_lst[num]], authors[ind_lst[num]], dist_lst[num], ind_lst[num] + 1),)
		nearest_texts.append(distances_persample)

	for index, (author, title, nearest_text) in enumerate(zip(authors, titles, nearest_texts)):
		fob_nodes.write(str(index + 1) + "\t" + str(title) + "\t" + str(author) + "\n")
		for num in range(0, neighbors-1):
			fob_edges.write(str(index+1) + "\t" + str(nearest_text[num][3]) + "\t" + "Undirected" + "\t" + str(nearest_text[num][2]) + "\n")
	
