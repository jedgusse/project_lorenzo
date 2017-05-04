#!/usr/bin/env

import glob

fob = open("/Users/jedgusse/stylofactory/corpora/_LEFTOVERS/div_no_roc_no_lec.txt")
text = fob.read()

def line_iterator(begin_value, end_value):

	focus_list = []
	index_list = []
	for index, word in enumerate(text.split()):
		if index in range(begin_value, end_value):
			focus_list.append(word)
			index_list.append(index)

	print()
	print("\"", " ".join(focus_list[:6]), "...", " ".join(focus_list[-7:]), "\"")
	print()

for index, i in enumerate(range(1500, 48*1500, 1500)):
	print("sm_{}".format(index+1)) 
	line_iterator(i-1500, i)