#!/usr/bin/env

import numpy as np
from scipy import stats
from collections import Counter
import glob
from string import punctuation

def deviant_cwords(folder_location):

	# Look at every text in the corpus, and see which words conspicuously (dis)appear in comparison
	# This is calculated by the coefficient of variation; the division of the standard deviation (sigma) by the mean (mu)
	# The total_frequency parameter makes sure that only words with a measurable frequency are taken into account. For instance,
	# in total, the word should occur at least 10 times before we can speak of noticeable and measurable difference.
	# The folder should contain text A, text B, ... text n. 

	# Parameter:
	total_frequency = 10

	all_texts = ""

	for filename in glob.glob(folder_location + "/*"):
		fob = open(filename)
		text = fob.read()
		all_texts += text + " "
	
	general_counter = Counter(all_texts.rstrip().split())
	
	vocabulary = []
	for word in general_counter.keys():
		vocabulary.append(word)

	counted_texts = []
	text_labels = []

	for filename in glob.glob(folder_location + "/*"):
		text_labels.append(filename.split("/")[-1].split(".")[0])
		fob = open(filename)
		text = fob.read()
		individual_counter = Counter(text.rstrip().split())
		counted_texts.append(individual_counter)

	print("TEXT LABELS   ", text_labels)

	values = []
	for word in vocabulary:
		word_tuple = (word,)
		for counted_text in counted_texts:
			word_tuple = word_tuple + (counted_text[word],)
		values.append(word_tuple)

	data = []
	for each in values:
		word = each[0]
		a = np.array([each[1], each[2]])
		if np.sum(a) <= total_frequency:
			pass
		else:
			results = (word, np.std(a) / np.mean(a), text_labels[np.argmax(a)])
			data.append(results)
	
	sorted_by_score = sorted(data, key=lambda tup: tup[1], reverse=True)
	
	# Writing it out to output folder

	fb = open("/Users/jedgusse/stylofactory/output/text_output/vergil_words.txt", 'w')
	for tuplez in sorted_by_score:
		if tuplez[2] == 'ecloga_ecloga':
			fb.write(tuplez[0] + "\t" + "\t" + str(tuplez[1]) + "\t" + "\t" + str(tuplez[2]) +  "\n")


