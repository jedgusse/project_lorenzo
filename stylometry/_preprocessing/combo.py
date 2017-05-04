import glob
import re
import string

for filename in glob.glob("/Users/jedgusse/Desktop/new_texts/*"):
	
	# Delete single lines

	text = " "
	try:
		fob = open(filename, 'r')
		result_f = open("/".join(filename.split("/")[:-1]) + "/_edits/" + filename.split("/")[-1].split(".")[0] + "_ed.txt", 'w')

		for line in fob:
			line = line.rstrip()
			line = line.lower()
			text += line + " "

		# Delete brackets (if PL), punctuation and numbers

		brackets = re.compile(r"\((.*?)\)|\[(.*?)\]|1|2|3|4|5|6|7|8|9|0")
		text = re.sub(brackets, " ", text)

		for c in string.punctuation:
			text = text.replace(c, " ")

		# Separate compounded nouns with -que and -ve particles

		list_ve = ["ave", "sive", "cave"]
		fob_que = open("/Users/jedgusse/stylofactory/code/_preprocessing/que_words.txt", 'r')
		text_que = fob_que.read()
		list_que = text_que.split()

		text_list = text.split()

		for index, word in enumerate(text_list):
			if word[-3:] == "que":
				if word in list_que:
					pass
				else:
					if word[0:3] == "utr" or word[0:3] == "uter":
						pass
					elif word[0:2] == "qu":
						pass
					elif word[0:4] == "unic" or word[0:4] == "uniu":
						pass
					elif word[0:3] == "ubi":
						pass
					else:
						text_list[index] = word.replace(word[-3:], " que")
			if word == "despective":
				text_list[index] = word.replace(word[-2:], " ve")

		# Insert this into code as well!
		
		corr_dict = {'tanquam': 'tamquam', 'quoties': 'quotiens', 'eous': 'meus', 'aliquoties': 'aliquotiens', 'nunquid': 'numquid', 'nunquam': 'numquam',
			 'quanquam': 'quamquam', 'sui': 'sibi'}

		text = " ".join(text_list)

		result_f.write(text)

	except IsADirectoryError:
		pass

	# Other undesired things can be deleted by looking up the index and deleting from that index, 
	# such as chapter titles in upper case letters.