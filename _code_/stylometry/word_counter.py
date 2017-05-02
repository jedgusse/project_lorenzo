#!/usr/bin/env

from collections import Counter
import numpy as np

def mfw_counter(texts, feat_amount, invalid_words):
	
	# Gather entire corpus into one large string to extract most common words in entire corpus
	temp_texts = " ".join(texts)
	
	# We will collect features in empty list
	features = []

	# Count all the words in the entire corpus
	general_counter = Counter(temp_texts.rstrip().split())

	# Choose features (takes into account the amount of features as specified in main.py, and ignores the invalid_words)
	for wordtuple in general_counter.most_common():
		if wordtuple[0] in invalid_words:
			pass
		else: 
			if len(features) < feat_amount:
				features.append(wordtuple[0])

	word_vectors = []
	for text_sample in texts:
		vector = []
		counter = Counter(text_sample.split())
		for feature in features:
			vector.append(counter[feature])
		word_vectors.append(vector)
	word_vectors = np.asarray(word_vectors)

	return word_vectors, features