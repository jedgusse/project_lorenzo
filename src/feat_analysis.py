#!/usr/bin/env

import argparse
from binascii import hexlify
import colorsys
from itertools import combinations
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (Normalizer,
                                   StandardScaler,
                                   FunctionTransformer)
from string import punctuation

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def align_yaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	_, y1 = ax1.transData.transform((0, v1))
	_, y2 = ax2.transData.transform((0, v2))
	inv = ax2.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
	miny, maxy = ax2.get_ylim()
	ax2.set_ylim(miny+dy, maxy+dy)

def align_xaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	x1, _ = ax1.transData.transform((v1, 0))
	x2, _ = ax2.transData.transform((v2, 0))
	inv = ax2.transData.inverted()
	dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0))
	minx, maxx = ax2.get_xlim()
	ax2.set_xlim(minx+dx, maxx+dx)

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
		
		pca = PCA(n_components=self.n_components)
		X_bar = pca.fit_transform(self.X)
		var_exp = pca.explained_variance_ratio_
		comps = pca.components_
		comps = comps.transpose()
		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(self.features, comps[:,0]), key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(self.features, comps[:,1]), key=lambda tup: tup[1], reverse=True)

		# Generate color dictionary
		color_dict = {author:index for index, author in enumerate(set(self.authors))}
		cmap = discrete_cmap(len(color_dict), base_cmap='rainbow')

		print("Explained variance: ", sum(pca.explained_variance_ratio_)*100)

		if show_samples == True:

			fig = plt.figure(figsize=(8,6))
			ax = fig.add_subplot(111)
			x1, x2 = X_bar[:,0], X_bar[:,1]

			# If anything needs to be invisible in plot, add to exclusion_list

			exclusion_list = []

			ax.scatter(x1, x2, 100, edgecolors='none', facecolors='none', cmap='rainbow')
			for index, (p1, p2, a, title) in enumerate(zip(x1, x2, self.authors, self.titles)):
				if a not in exclusion_list:
					ax.text(p1, p2, title[:2] + '_' + title.split("_")[0], ha='center',
				    va='center', color=cmap(color_dict[a]), fontdict={'size': 7})

			# Legend settings (code for making a legend)

			"""brev_patch = mpatches.Patch(color=colours['1'], label='Bernard\'s intra corpus (brevis)')
			#plt.legend(handles=[brev_patch, new_patch, pre_patch, mid_patch, post_patch, nic_patch], loc=2, prop={'size':9})"""

			if show_loadings == True:
				ax2 = ax.twinx().twiny()
				l1, l2 = loadings[:,0], loadings[:,1]
				ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
				for x, y, l in zip(l1, l2, self.features):
					ax2.text(x, y, l, ha='center', va="center", color="black",
					fontdict={'family': 'Arial', 'size': 6})

				# Align axes

				align_xaxis(ax, 0, ax2, 0)
				align_yaxis(ax, 0, ax2, 0)
			
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
					fontdict={'family': 'Arial', 'size': 6})

			ax2.set_xlabel('PC1')
			ax2.set_ylabel('PC2')

			plt.axhline(y=0, ls="--", lw=0.5, c='0.75')
			plt.axvline(x=0, ls="--", lw=0.5, c='0.75')

			plt.show()

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

		fob_nodes = open("", "w")
		fob_edges = open("", "w")

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

if __name__ == '__main__':
	# Instantiate parser
	parser = argparse.ArgumentParser()

	# Instantiate arguments (obligatory or optional (e.g. --n_components))
	parser.add_argument("path")
	parser.add_argument("--sample_size", type=int, default=2000, help='Please enter length of samples')
	parser.add_argument("--max_features", type=int, default=150)
	
	# Call arguments
	args = parser.parse_args()

	# Load corpus from path that is specified

	authors = []
	titles = []
	texts = []

	for filename in glob.glob(args.path + '/*'):
		author = filename.split("/")[-1].split(".")[0].split("_")[0]
		title = filename.split("/")[-1].split(".")[0].split("_")[1]

		bulk = []

		fob = open(filename)
		text = fob.read()
		for word in text.rstrip().split():
			for char in word:
				if char in punctuation:
					word = word.replace(char, "")
			word = word.lower()
			bulk.append(word)
		bulk = [bulk[i:i+args.sample_size] for i in range(0, len(bulk), args.sample_size)]

		for index, sample in enumerate(bulk):
			if len(sample) == args.sample_size:
				authors.append(author)
				titles.append(title + "_{}".format(str(index + 1)))
				texts.append(" ".join(sample))

	# Vectorize and normalize corpus as best-ranked in SVM
	
	model = TfidfVectorizer(max_features=args.max_features)
	tfidf_vectors = model.fit_transform(texts).toarray()
	features = model.get_feature_names()

	# Rank feature list and tfidf_vectors in order
	# corpus_vector sums up all tfidf_vectors and yield highest values for most occurring features
	corpus_vector = np.ravel(np.sum(tfidf_vectors, axis=0))
	mfwords = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]
	ordered_X = []
	for word in mfwords:
		for feature, freq in zip(features, tfidf_vectors.transpose()):
			if word == feature:
				ordered_X.append(freq)
	tfidf_vectors = np.array(ordered_X).transpose()

	# Normalize counts
	tfidf_normalized = StandardScaler().fit_transform(tfidf_vectors)

	PrinCompAnal(authors, titles, tfidf_normalized, mfwords, 2).plot(show_samples=True, show_loadings=True)


