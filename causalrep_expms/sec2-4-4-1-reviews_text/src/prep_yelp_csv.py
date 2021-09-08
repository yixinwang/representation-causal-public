import json
from pprint import pprint
import numpy as np
import pandas as pd
import string
# import matplotlib.pyplot as plt
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# import PeerRead.bert.tokenization as tokenization
import os
from fnmatch import fnmatch
import argparse
from scipy import sparse
import ast



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # options are funny,useful,stars,cool
    args, unknown = parser.parse_known_args()

    datadir = '/proj/sml/usr/yixinwang/datasets/yelp_dataset/'



    review_df = pd.read_csv(os.path.join(datadir, 'review.csv'))
    business_df = pd.read_csv(os.path.join(datadir, 'business.csv'))
    papers_df = pd.merge(review_df, business_df, left_on="business_id", right_on="business_id")
    papers_df = papers_df.dropna(subset=['text', 'stars_x', 'funny', 'useful', 'stars_x', 'cool'])
    papers_df['id'] = np.arange(len(papers_df))
    
    outdir = '../dat/yelp'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    papers_df[['id', 'funny', 'useful', 'stars_x', 'cool',
        'review_id', 'text', 'business_id', 'date', 'user_id', \
        'categories', 'attributes.Open24Hours', 'postal_code', \
        'city', 'stars_y', \
        'latitude', 'name', 'is_open', 'longitude', \
        'review_count', 'attributes.NoiseLevel', 'state', \
        'hours']].to_csv(\
        os.path.join(outdir, "yelp_meta.csv"),
        encoding='utf-8')

    punct = string.punctuation + '0123456789' 
    translator = str.maketrans('\n', ' ', punct)
    papers_df['text'] = [paper.translate(translator) for paper in papers_df['text']]

    papers_df[['text', 'funny', 'useful', 'cool', 'stars_x']].to_csv('yelp_text.csv')

    papers_df[['text', 'funny', 'useful', 'cool', 'stars_x']].to_pickle('yelp_data.pkl')

    vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(1,1), max_features=10000)
    X = vectorizer.fit_transform(papers_df['text'])

    for ymeaning in ['funny', 'useful', 'cool', 'stars_x']:
        outdir = '../dat/yelp_'+ymeaning.split('_')[0]

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        sparse.save_npz(os.path.join(outdir, "full_X.npz"), X)
        np.save(os.path.join(outdir, "full_y.npy"), papers_df[ymeaning])

        with open(os.path.join(outdir, "full_vocab.txt"), 'w') as f:
            for item in vectorizer.get_feature_names():
                f.write("%s\n" % bytes(item, 'utf-8').decode('utf-8', 'ignore'))



