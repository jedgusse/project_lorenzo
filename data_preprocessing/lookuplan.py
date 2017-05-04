import glob
import langid
import numpy as np

# Langid -> Used by Bamman for the 11K corpus too
# Marco Lui

sample_length = 200

counter = 0

for filename in glob.glob("/Users/jedgusse/lorenzo_valla/data/output/patrologia/*"):
    fob = open(filename)
    filename = filename.split("/")[-1].split(".")[0]
    text = fob.read().rstrip()
    text = text.split()
    bulk = [text[i:i+sample_length] for i in range(0, len(text), sample_length)]
    for i in bulk:
    sample = " ".join(i)
    if langid.classify(sample)[0] != 'la':
        counter += len(sample)
        print()
        print("... finding non-Latin text in {} ...".format(filename))
        print("count thusfar: ", counter)
        print()

print("TOTAL COUNT: ", counter)
