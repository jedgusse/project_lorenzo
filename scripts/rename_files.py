
import os
import json
from collections import defaultdict
from lxml import etree


ROOT_FOLDER = '../data/'


def normalize_name(s):
    s = s.replace('.', '-').replace(',', '').replace(';', '')
    return '_'.join(s.split())

packhum_subpath = 'packhum_post'
orig_packhum_path = os.path.join(ROOT_FOLDER, 'packhum/packhum.json')
works_by_author = defaultdict(lambda: defaultdict(list))
with open(orig_packhum_path, 'r+') as inf:
    for idx, line in enumerate(inf):
        obj = json.loads(line.strip())
        author, title = obj['author'], obj['work']
        author, title = normalize_name(author), normalize_name(title)
        works_by_author[author][title].append(obj)

for author in works_by_author:
    for title in works_by_author[author]:
        fname = '{author}.{title}.json'.format(author=author, title=title)
        fname = os.path.join(ROOT_FOLDER, packhum_subpath, fname)
        orig_pages = works_by_author[author][title]
        orig_author = orig_pages[0]['author']
        orig_title = orig_pages[0]['work']
        pages = sorted(orig_pages, key=lambda w: w['page'])
        work = {'author': orig_author,
                'title': orig_title,
                'pages': [{'page_num': p['page'],
                           'cit': p['cit'],
                           'text': p['text']}
                          for p in pages]}
        with open(fname, 'w') as outf:
            json.dump(work, outf)

pl_subpath = 'patrologia_rnr'
works_by_author = defaultdict(lambda: defaultdict(int))
for f in os.listdir(os.path.join(ROOT_FOLDER, pl_subpath)):
    with open(os.path.join(ROOT_FOLDER, pl_subpath, f), 'r+') as inf:
        s = inf.read()
        root = etree.fromstring(
            # get rid of rogue xml
            s.replace('<unknown>', 'unknown').encode('utf-8'))
        author, title = root.attrib['auteur'], root.attrib['titre']
        author, title = normalize_name(author), normalize_name(title)
        suffix = ''
    if author in works_by_author and title in works_by_author[author]:
        suffix = "." + str(works_by_author[author][title])
        works_by_author[author][title] += 1
        fname = '{author}.{title}{suffix}.xml'.format(
            author=author, title=title, suffix=suffix)
    with open(os.path.join(ROOT_FOLDER, 'pl', fname), 'w') as outf:
        outf.write(s)
