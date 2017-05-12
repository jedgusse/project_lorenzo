#!/usr/bin/env

from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PrinCompAnal:

	def __init__(self, authors, titles, X,
				 features, n_components):
		self.authors = authors
		self.titles = titles
		self.X = X
		self.features = features
		self.n_components = n_components

	def plot(self, show_samples, show_loadings):

		# Normalizer and Delta perform badly
		# They flatten out all difference in a PCA plot

		colours = {'GilPoit': '#6190FF', 'GuibNog': '#A4CDFF', 'Bern': '#FF4F46', 'div': '#FF9181', '5': '#FFC3A4', 'n': '#3FD121', 'hugo': '#D157B5', 'h': '#D157B5', 
		   'JohnSal': '#6190FF', 'ro': '#000000', 'NicCl': '#dc1818', 'AnsLaon': '#dc1818', '...': '#dc1818', 'lec': '#D157B5', 'JohnSal': '#D157B5', 'WilConch': '#3FD121'}
		
		pca = PCA(n_components=self.n_components)
		X_bar = pca.fit_transform(self.X)
		var_exp = pca.explained_variance_ratio_
		comps = pca.components_
		comps = comps.transpose()
		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(self.features, comps[:,0]), key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(self.features, comps[:,1]), key=lambda tup: tup[1], reverse=True)

		print("Explained variance: ", sum(pca.explained_variance_ratio_)*100)

		if show_samples == True:

			fig = plt.figure(figsize=(8,6))
			ax = fig.add_subplot(111)
			x1, x2 = X_bar[:,0], X_bar[:,1]

			# If anything needs to be invisible in plot, add to exclusion_list

			exclusion_list = []

			ax.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
			for p1, p2, a, title in zip(x1, x2, self.authors, self.titles):
				if a not in exclusion_list:
					ax.text(p1, p2, title[:2] + '_' + title.split("_")[1], ha='center',
				    va='center', color=colours[a], fontdict={'size': 7})
			ax.set_xlabel('PC1')
			ax.set_ylabel('PC2')

			# Legend settings (code for making a legend)

			"""brev_patch = mpatches.Patch(color=colours['1'], label='Bernard\'s intra corpus (brevis)')
			#plt.legend(handles=[brev_patch, new_patch, pre_patch, mid_patch, post_patch, nic_patch], loc=2, prop={'size':9})"""

			if show_loadings == True:
				ax2 = ax.twinx().twiny()
				l1, l2 = loadings[:,0], loadings[:,1]
				ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
				for x, y, l in zip(l1, l2, self.features):
					ax2.text(x, y, l, ha='center', va="center", color="black",
					fontdict={'family': 'Arial', 'size': 10})
			
			elif show_loadings == False:
				print("No loadings in PCA")

			plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
			plt.axvline(x=0, ls="--", lw=0.5, c='0.75')

			plt.show()

			fig.savefig("/Users/jedgusse/compstyl/output/fig_output/pcafig.pdf", 
						transparent=True, 
						format='pdf')
			# Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

		elif show_samples == False:

			fig = plt.figure(figsize=(8, 6))

			ax2 = fig.add_subplot(111)
			l1, l2 = loadings[:,0], loadings[:,1]
			ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none')
			for x, y, l in zip(l1, l2, features):
				ax2.text(x, y, l, ha='center', va='center', color='black',
					fontdict={'family': 'Arial', 'size': 10})

			ax2.set_xlabel('PC1')
			ax2.set_ylabel('PC2')

			plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
			plt.axvline(x=0, ls="--", lw=0.5, c='0.75')

			plt.show()

			fig.savefig("/Users/jedgusse/compstyl/output/fig_output/pcafig.pdf", 
						transparent=True, 
						format='pdf')
			# Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

class GephiNetworks:

	def __init__(self, authors, titles, X, n_nbrs):
		self.authors = authors
		self.titles = titles
		self.X = X
		self.n_nbrs = n_nbrs
		self.distances = []
		self.indices = []

	def plot(self):

		model = NearestNeighbors(n_neighbors=self.n_nbrs,
								algorithm='ball_tree'
								).fit(self.X)

		distances, indices = model.kneighbors(self.X)

		# Here we make the tab-separated txt files which should go into GEPHI; we instantiate a file object.
		# Calculates Nearest Neighbours of each txt sample.
		# Outputs distances

		fob_nodes = open("/Users/jedgusse/compstyl/output/gephi_output/gephi_nodes.txt", "w")
		fob_edges = open("/Users/jedgusse/compstyl/output/gephi_output/gephi_edges.txt", "w")

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

		normalized_distances = (distances - lowest_value) / (highest_value - lowest_value)

		# Group our nearest texts:

		nearest_texts = []

		for dist_lst, ind_lst in zip(normalized_distances, indices):
			distances_persample = ()
			for num in range(1, self.n_nbrs):
				distances_persample = distances_persample + ((self.titles[ind_lst[num]],
															  self.authors[ind_lst[num]], 
															  dist_lst[num], ind_lst[num] + 1),)
			nearest_texts.append(distances_persample)

		for index, (author, title, nearest_text) in enumerate(zip(self.authors, 
																  self.titles, 
																  nearest_texts)):
			fob_nodes.write(str(index + 1) + "\t" + str(title) + "\t" + str(author) + "\n")
			for num in range(0, self.n_nbrs-1):
				fob_edges.write(str(index+1) + "\t" + str(nearest_text[num][3]) + "\t" + "Undirected" + "\t" + str(nearest_text[num][2]) + "\n")

		return distances, indices