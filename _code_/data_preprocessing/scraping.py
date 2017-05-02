#!/usr/local/bin/python

import glob
import os
from bs4 import BeautifulSoup
from collections import OrderedDict
from string import punctuation
import urllib.request
import urllib.parse

url = "http://pld.chadwyck.co.uk/all/htxview?template=toc_hdft.htx&content=toc_az.htx"

values = {'name': '1'}

print(urllib.parse(url))