
import os
import json
from collections import defaultdict
from lxml import etree


def normalize_name(s):
    s = s.replace('.', '-').replace(',', '').replace(';', '')
    return '_'.join(s.split())


def process_packhum(works_by_author, root_folder, packhum_target):
    for author in works_by_author:
        for title in works_by_author[author]:
            fname = '{author}.{title}.json'.format(author=author, title=title)
            fname = os.path.join(root_folder, packhum_target, fname)
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


def process_pl(root_folder, orig_pl, pl_target):
    works_by_author = defaultdict(lambda: defaultdict(int))
    for f in os.listdir(orig_pl):
        with open(os.path.join(orig_pl, f), 'r+') as inf:
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
        with open(os.path.join(root_folder, pl_target, fname), 'w') as outf:
            outf.write(s)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--packhum_target', default='packhum/merged')
    parser.add_argument('--pl_target', default='pl')
    args = parser.parse_args()

    root_folder = args.data_path

    packhum_target = args.packhum_target
    orig_packhum = os.path.join(root_folder, 'packhum/packhum.json')
    works_by_author = defaultdict(lambda: defaultdict(list))
    with open(orig_packhum, 'r+') as inf:
        for idx, line in enumerate(inf):
            obj = json.loads(line.strip())
            author, title = obj['author'], obj['work']
            author, title = normalize_name(author), normalize_name(title)
            works_by_author[author][title].append(obj)

    if not os.path.isdir(os.path.join(root_folder, packhum_target)):
        os.mkdir(os.path.join(root_folder, packhum_target))
    process_packhum(works_by_author, root_folder, packhum_target)

    pl_target = args.pl_target
    orig_pl = os.path.join(root_folder, 'patrologia_rnr')
    if not os.path.isdir(os.path.join(root_folder, pl_target)):
        os.mkdir(os.path.join(root_folder, pl_target))
    process_pl(root_folder, orig_pl, pl_target)
