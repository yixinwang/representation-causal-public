import json
from pprint import pprint
import numpy as np
import pandas as pd
import string
# import matplotlib.pyplot as plt
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import PeerRead.bert.tokenization as tokenization
import os
from fnmatch import fnmatch
import argparse
from scipy import sparse
import ast



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--corpusname', \
        choices=['amazon', 'tripadvisor'], \
        default='tripadvisor')
    args, unknown = parser.parse_known_args()
    corpusname = args.corpusname

    rootdir = '/proj/sml/usr/yixinwang/datasets/reviews/'
    if args.corpusname == 'amazon':
        datadir = os.path.join(rootdir, 'Amazon_corpus')
    elif args.corpusname == 'tripadvisor':
        datadir = os.path.join(rootdir, 'TripAdvisor_corpus')

    outdir = os.path.join('../dat/reviews', corpusname)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    files = os.listdir(datadir)

    papers_dfs = []
    for i in range(len(files)):
        papers_df = pd.read_csv(os.path.join(datadir, files[i]), sep='\t', header=None)
        papers_dfs.append(papers_df)
    papers_df = pd.concat(papers_dfs)

    papers_df.columns = ['rawid', 'review_text', 'y']
    papers_df['id'] = np.arange(len(papers_df))

    papers_df[['id', 'review_text', 'y']].to_csv(\
        os.path.join(outdir, corpusname+"_meta.csv"), encoding='utf-8')

    np.save(os.path.join(outdir, "full_y.npy"), papers_df['y'])


    # process abstract

    punct = string.punctuation + '0123456789' 
    translator = str.maketrans('\n', ' ', punct)
    papers_df['review_text'] = [paper.translate(translator) for paper in papers_df['review_text']]

    vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(1,1), max_features=10000)
    X = vectorizer.fit_transform(papers_df['review_text'])

    sparse.save_npz(os.path.join(outdir, "full_X.npz"), X)


    with open(os.path.join(outdir, "full_vocab.txt"), 'w') as f:
        for item in vectorizer.get_feature_names():
            f.write("%s\n" % bytes(item, 'utf-8').decode('utf-8', 'ignore'))



