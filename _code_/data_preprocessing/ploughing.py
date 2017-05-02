#!/usr/local/bin/python

import glob
import os
from bs4 import BeautifulSoup
from collections import OrderedDict
from string import punctuation
import re
import urllib.request

# File containing the extraction code for the respective corpora of Latin texts which have been put to use.

""" 
# Perseus XML-parsing code
# Uses namespace prefixes 

# First few lines retrieves METADATA, constructs a dictionary
# Constructing a dictionary yielding the metadata
# This also contained translations

perseus_dict = {}

for author_folder in glob.glob("/Users/jedgusse/lorenzo_valla/data/input/perseus/*"):
	
	if os.path.isfile("{}/__cts__.xml".format(author_folder)) == True:
		pass
	else:
		print("File does not exist  ", author_folder.split("/")[-1])

	title_names = []
	for text_spec in glob.glob("{}/*".format(author_folder)):

		if text_spec[-3:] == "xml":
			fob_1 = open(text_spec)
			xml_1 = fob_1.read()
			author_doc = BeautifulSoup(xml_1, 'lxml')
			
			for author in author_doc.find_all('ti:groupname'):
				author = author.getText()

			for author in author_doc.find_all('cts:groupname'):
				author = author.getText()

			author = "".join([char for char in author if char not in punctuation])

		else:

			for title_spec in glob.glob("{}/*".format(text_spec)):
				
				# First retrieve metadata, specifications on the texts
				if title_spec.split('/')[-1] == "__cts__.xml":
					fob_2 = open(title_spec)
					xml_2 = fob_2.read()
					title_doc = BeautifulSoup(xml_2)

					for title in title_doc.find_all('ti:title'):
						title = title.getText()
						title_names.append(title)

					for title in title_doc.find_all('cts:title'):
						title = title.getText()
						title_names.append(title)

				elif os.path.isfile("{}/__cts__.xml".format(text_spec)) == False:
					if title_spec[-3:] == "xml":
						fob_3 = open(title_spec)
						xml_3 = fob_3.read()
						text_doc = BeautifulSoup(xml_3)

						title = text_doc.find('title').getText()
						title_names.append(title)

				# Retrieve texts themselves and write out to files

				if 'lat' in title_spec and title_spec[-3:] == "xml":
					fob_4 = open(title_spec)
					xml_4 = fob_4.read()
					txt = BeautifulSoup(xml_4)

					actual_text = txt.find('text').getText()
					unknown_fob = open("/Users/jedgusse/lorenzo_valla/data/output/perseus/{}".format("{}_{}.txt".format(author.split()[0], title.split()[0][:3])), 'w')
					unknown_fob.write(author + ", " + title + "\n" + "\n" + actual_text)

			perseus_dict[author] = title_names

perseus_sorted = OrderedDict(sorted(perseus_dict.items(), key=lambda x: x[0]))"""

"""camena_dict = {}

for corpus_folder in glob.glob("/Users/jedgusse/lorenzo_valla/data/input/camena/*"):
	folder_name = corpus_folder.split("/")[-1]
	for xml_file in glob.glob("/Users/jedgusse/lorenzo_valla/data/input/camena/{}/*".format(corpus_folder.split("/")[-1])):
		
		fob_name = xml_file.split("/")[-1].split(".")[0] + ".txt"

		fob = open(xml_file, 'r', encoding='utf-8', errors='ignore')
		xml = fob.read()
		soup = BeautifulSoup(xml, 'lxml')

		new_fob = open("/Users/jedgusse/lorenzo_valla/data/output/camena/{}/{}".format(folder_name, fob_name), 'w')

		text = soup.text
		new_fob.write(text)"""

"""csel_dict = {}

for author_folder in glob.glob("/Users/jedgusse/lorenzo_valla/data/input/csel/*"):
	author_name = author_folder.split("/")[-1]
	for title_folder in glob.glob("/Users/jedgusse/lorenzo_valla/data/input/csel/{}/*".format(author_name)):
		title_name = title_folder.split("/")[-1]
		if title_folder.split("/")[-1] == "__cts__.xml":
			fob_1 = open(title_folder)
			xml_1 = fob_1.read()
			author_doc = BeautifulSoup(xml_1, 'lxml')

			for author in author_doc.find_all('ti:groupname'):
				author = author.getText()

		else:
			for filename in glob.glob("/Users/jedgusse/lorenzo_valla/data/input/csel/{}/{}/*".format(author_name, title_name)):
				if filename.split("/")[-1] == "__cts__.xml":
					fob_2 = open(filename)
					xml_2 = fob_2.read()
					title_doc = BeautifulSoup(xml_2, 'lxml')

					for title in title_doc.find_all('ti:title'):
						title = title.getText()

				elif "lat" in filename:
					fob_3 = open(filename)
					xml_3 = fob_3.read()
					soup = BeautifulSoup(xml_3, 'lxml')

					pattern = re.compile('<note(.*)</note>', re.DOTALL)

					full_text = ""

					for div in soup.findAll('div'):
						full_code = str(div)
						full_code = re.sub(pattern, "", full_code)
						text = BeautifulSoup(full_code, "lxml")
						text = text.getText()
						full_text += text

					fob_4 = open("/Users/jedgusse/lorenzo_valla/data/output/csel/{}_{}.txt".format(author.split()[0], title.split()[0][:3]), 'w')
					fob_4.write(full_text)"""


pl_links = {1: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000000000', 2: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000089568', 3: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000062547', 4: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000054649', 5: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000046552', 6: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000038531', 7: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000031255', 8: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000019719', 9: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000008446', 10: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000180993', 11: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000171859', 12: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000165418', 13: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000159514', 14: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000152058', 15: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000141312', 16: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000132263', 17: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000118323', 18: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000107541', 19: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000099775', 20: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000088970', 21: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000080168', 22: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000069276', 23: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000067764', 24: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000067711', 25: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000067587', 26: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000067270', 27: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000065038', 28: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000064869', 29: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000063249', 30: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000062163', 31: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000061907', 32: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000060757', 33: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000059794', 34: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000058904', 35: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000058387', 36: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000058101', 37: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000057922', 38: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000055769', 39: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000055196', 40: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000052887', 41: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000052150', 42: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000050822', 43: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000050698', 44: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000049892', 45: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000049187', 46: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000049096', 47: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000048510', 48: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000047887', 49: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000046947', 50: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000046060', 51: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000044742', 52: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000044416', 53: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000043677', 54: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000041874', 55: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000041583', 56: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000041139', 57: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000040400', 58: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000039538', 59: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000039045', 60: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000038459', 61: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000037912', 62: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000037017', 63: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000035755', 64: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000035376', 65: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000034815', 66: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000034368', 67: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000033344', 68: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000032826', 69: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000031708', 70: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000030319', 71: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000029660', 72: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000028852', 73: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000028396', 74: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000027505', 75: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000026608', 76: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000025895', 77: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000024635', 78: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000023057', 79: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000020874', 80: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000019042', 81: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000018847', 82: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000018243', 83: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000017026', 84: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000014854', 85: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000014684', 86: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000012999', 87: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000011248', 88: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000009905', 89: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000008703', 90: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000007591', 91: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000007175', 92: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000006975', 93: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000005989', 94: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000005485', 95: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000004491', 96: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000002582', 97: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000002173', 98: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000001374', 99: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000000526', 100: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000180499', 101: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000179233', 102: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000178283', 103: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000176731', 104: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000176125', 105: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000174801', 106: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000173830', 107: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000173319', 108: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000172862', 109: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000172369', 110: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000171453', 111: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000170930', 112: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000170213', 113: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000169292', 114: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000168675', 115: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000167586', 116: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000167270', 117: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000167004', 118: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000166429', 119: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000165728', 120: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000164843', 121: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000163775', 122: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000163452', 123: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000162789', 124: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000162116', 125: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000161786', 126:'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000161047', 127: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000160838', 128: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000160745', 129: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000159918', 130: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000158934', 131: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000158351', 132: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000157848', 133: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000157384', 134: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000157123', 135: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000156344', 136: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000155771', 137: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000154902', 138: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000153802', 139: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000152534', 140: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000149779', 141: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000148702', 142: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000147606', 143: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000146765', 144: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000146325', 145: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000144875', 146: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000144111', 147: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000143566', 148: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000142747', 149: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000141587', 150: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000140129', 151: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000137915', 152: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000137915', 153: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000137315', 154: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000137206', 155: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000135788', 156: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000135215', 157: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000134503', 158: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000133680', 159: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000132897', 160: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000131733', 161: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000126550', 162: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000125057', 163: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000123613', 164: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000123176', 165: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000122444', 166: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000121221', 167: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000119648', 168: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000119543', 169: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000118890', 170: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000117327', 171: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000116693', 172: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000114976', 173: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000114517', 174: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000113878', 175: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000113386', 176: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000112235', 177: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000110650', 178: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000110056', 179: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000108725', 180: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000106421', 181: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000106083', 182: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000105243', 183: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000104836', 184: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000104203', 185: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000103142', 186: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000102211', 187: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000101871', 188: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000100970', 189: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000099995', 190: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000098464', 191: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000097952', 192: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000096472', 193: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000096004', 194: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000095570', 195: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000095119', 196: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000093949', 197: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000092800', 198: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000091284', 199: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000090462', 200: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000087201', 201: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000085847', 202: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000084946', 203: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000084144', 204: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000082924', 205: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000082398', 206: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000081972', 207: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000081303', 208: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000081189', 209: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000080663', 210: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000079522', 211: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000078867', 212: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000078136', 213: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000077392', 214: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000076247', 215: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000074745', 216: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000073303', 217: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000072339', 218: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000070634', 219: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000069647', 220: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000068664', 221: 'http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pld&rft_dat=xri:pld:ft:all:Z000068335'}

for vol in range(147, 222):
	pl_page = pl_links[vol]
	while True:
		try:
			page = urllib.request.urlopen(pl_page)
			soup = BeautifulSoup(page)
			text = soup.getText()
			fob = open("/Users/jedgusse/lorenzo_valla/data/output/patrologia/{}".format("PL_vol{}.txt".format(str(vol))), 'w')
			fob.write(text)
		except urllib.error.HTTPError: 
			continue
		break





