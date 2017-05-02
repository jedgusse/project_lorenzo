#!/usr/local/bin/python

import re
import glob

for filename in glob.glob("/Users/jedgusse/lorenzo_valla/data/output/patrologia/*"):

	fob = open(filename)
	text = fob.readlines()

	pattern = re.compile('18(.*)')

	begin_index = None
	end_index = None

	for index, line in enumerate(text):
		if line.rstrip() == "ELENCHUS" or line.rstrip() == "ELENCHUS OPERUM":
			if pattern.match(text[index-1].strip()):
				begin_index = index

		if line.rstrip() == "Main volume text":
			end_index = index

	fob.close()

	fob = open("/Users/jedgusse/lorenzo_valla/data/output/patrologia/PL_vol211.txt")
	text = fob.readlines()

	for index, line in enumerate(text):
		if index in range(begin_index, end_index):
			print(line)