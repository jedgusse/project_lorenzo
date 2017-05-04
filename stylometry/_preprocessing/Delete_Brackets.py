import re

fob = open("/Users/jedgusse/Desktop/hugo_desacr.txt")
text = fob.read()

fob2 = open("/Users/jedgusse/Desktop/hugo_desacr2.txt", "w")

regex = re.compile(r"\((.*?)\)|\[(.*?)\]")
result = re.sub(regex, "", text)

# Other undesired things can be deleted by looking up the index and deleting from that index, such as chapter titles in upper case letters.

indices = []
for index, word in enumerate(result.split()):
	if word == "CAP.":
		indices.append(index)

indices_2 = [i+1 for i in indices]


for index, word in enumerate(result.split()):
	if index in indices or index in indices_2:
		pass
	else:
		fob2.write(word + " ")