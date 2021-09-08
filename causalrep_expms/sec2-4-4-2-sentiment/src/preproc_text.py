import importlib
import copy
import io, time
from io import BytesIO
from itertools import combinations, cycle, product
import math
import numpy as np
import pandas as pd
import pickle
import tarfile
import random
import re
import requests
from scipy.sparse import hstack, lil_matrix

from tqdm import tqdm
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1) # change None to -1


from collections import Counter, defaultdict
import numpy as np
import re

import sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

# import torch
from transformers import * # here import bert

import warnings
warnings.filterwarnings("ignore")


from data_structure import Dataset #, get_IMDB, get_kindle
import argparse

import utils
importlib.reload(utils)
from utils import *

# randseed = 52744889
randseed = int(time.time()*1e7%1e8)
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
# torch.manual_seed(randseed)

parser = argparse.ArgumentParser(description='Preprocess datasets')
parser.add_argument('-d', '--dataset', type=str, default='kindle', required=False)
args, unk = parser.parse_known_args()



# load data

data_path0 = "/proj/sml/usr/yixinwang/representation-causal/src/causalrep_expms/aaai-2021-counterfactuals-main/data/Step0_data/"

data_path2 = "/proj/sml/usr/yixinwang/representation-causal/src/causalrep_expms/aaai-2021-counterfactuals-main/data/Step2_data/"

data_path3 = "/proj/sml/usr/yixinwang/representation-causal/src/causalrep_expms/aaai-2021-counterfactuals-main/data/Step3_data/"

data_out = "/proj/sml/usr/yixinwang/representation-causal/src/causalrep_expms/aaai-2021-counterfactuals-main/out/"


df_kindle = get_kindle(data_path0)
print("kindle dataset", df_kindle.shape, Counter(df_kindle.label))
# display(df_kindle.head())

df_imdb = get_IMDB(data_path0)
print("imdb dataset", df_imdb.shape, Counter(df_imdb.label))
# display(df_imdb.head())

df_large_imdb = get_large_IMDB_sentences(data_path0)
print("large imdb dataset", df_large_imdb.shape, Counter(df_large_imdb.label))
# display(df_large_imdb.head())

# start processing

datasets = []
get_data_df, moniker, coef_thresh, placebo_thresh = get_IMDB, 'imdb', 1.0, 0.1

df = get_data_df(data_path0)
X, y, vec, feats = simple_vectorize(df) # vectorize text
ds = Dataset(X, y, vec, df, moniker) # construct dataset object

print('%s dataset, %d instances' % (moniker,len(df)))
print('Label distribution: %s' % str(Counter(y).items()))
print('Feature matrix: %s' % str(X.shape))


# ds.top_feature_idx, ds.placebo_feature_idx, ds.coef = get_top_terms_preproc_preproc(ds, coef_thresh=coef_thresh, placebo_thresh=placebo_thresh, C=1)

# ds.top_features = feats[ds.top_feature_idx]

# ds.placebo_features = feats[ds.placebo_feature_idx]

# print('\n%d top terms: %d pos, %d neg\n' % (len(ds.top_features), len(np.where(ds.coef[ds.top_feature_idx]>0)[0]), len(np.where(ds.coef[ds.top_feature_idx]<0)[0])))

# print('\n%d placebo terms: %d pos, %d neg\n' % (len(ds.placebo_features), len(np.where(ds.coef[ds.placebo_feature_idx]>0)[0]), len(np.where(ds.coef[ds.placebo_feature_idx]<0)[0])))

# print('getting all sentences as control')
# ds.all_sentences = get_all_sentences(df)
# print('%d control sentences\n\n' % len(ds.all_sentences))
# embed_all_sentences(ds.all_sentences)    

# pickle.dump(ds, open(data_out + 'ds_' + moniker + '_emb.pkl','wb'))


# feats = np.array(ds.vec.get_feature_names())


# embeddings = np.array([sentence.context_embedding for sentence in ds.all_sentences])

# clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000, C=0.1)
# clf.fit(ds.X, ds.y)
# print(classification_report(ds.y, clf.predict(ds.X)))
# naive_coef = clf.coef_[0]

# bow_emb = np.column_stack([ds.X.toarray(), embeddings])
# clf.fit(bow_emb, ds.y)
# print(classification_report(ds.y, clf.predict(bow_emb)))
# causal_coef = clf.coef_[0]

# diff = naive_coef - causal_coef[:len(naive_coef)]

# print("top words for naive", feats[np.argsort(naive_coef* (-1))[:50]])

# print("top words for causal", feats[np.argsort(causal_coef[:len(naive_coef)] * (-1))[:50]])

# print("top words for pos diff", feats[np.argsort(diff * (-1))[:50]])

# print("top words for neg diff", feats[np.argsort(diff * (+1))[:50]])



# produce train/test datasets from step 2

df_kindle = pickle.load(open(data_path2+"kindle_data.pkl",'rb'))
df_test = df_kindle[df_kindle['flag']=='test']

df_test_select = select_sents(df_test, data_path2)
# display(df_test_select.head())

ds_imdb = run_experiment(moniker='imdb',coef_thresh=0.4,data_path=data_path2, data_out=data_out)
ds_imdb_sents = run_experiment(moniker='imdb_sents',coef_thresh=1.0,data_path=data_path2, data_out=data_out)

ds_kindle = run_experiment(moniker='kindle',coef_thresh=1.0,data_path=data_path2, data_out=data_out)



moniker = args.dataset
ds = load_data(moniker, data_out)
dir(ds)
train_data, test_data = organize_data(ds,limit='')
df_result = classification_performance(train_data, test_data)
df_result

train_data['all_original_sentences'] = get_all_sentences(pd.DataFrame(train_data['original']))
embed_all_sentences(train_data['all_original_sentences'])

pickle.dump(train_data, open(data_out + 'ds_' + moniker + 'train' + '_w_emb.pkl','wb'))

test_data['all_original_sentences'] = get_all_sentences(pd.DataFrame(test_data['Original']))
embed_all_sentences(test_data['all_original_sentences'])

test_data['all_counterfactual_sentences'] = get_all_sentences(pd.DataFrame(test_data['Counterfactual']))
embed_all_sentences(test_data['all_counterfactual_sentences'])

pickle.dump(test_data, open(data_out + 'ds_' + moniker + 'test' + '_w_emb.pkl','wb'))


# ds = load_data('imdb_sents')
# train_data, test_data = organize_data(ds,limit='')
# df_result = classification_performance(train_data, test_data)
# df_result

# ds = load_data('kindle')
# train_data, test_data = organize_data(ds,limit='')
# df_result = classification_performance(train_data, test_data)
# df_result



# train_data, test_data = organize_data(ds,limit='')
# train_data, ct_train_data, obs_test_data, ct_test_data = ds.train, ds.train_ct, ds.test, ds.test_ct

# # train_text, train_label = ct_train_data.text, ct_train_data.label
# train_text, train_label = train_data.text, train_data.label
# # test_text, test_label = ct_test_data.text, ct_test_data.label
# test_text, test_label = obs_test_data.text, obs_test_data.label

# train = 
# embed_all_sentences(get_all_sentences(pd.DataFrame({'text':train_text, 'label':train_label})))
# embed_all_sentences(get_all_sentences(pd.DataFrame({'text':test_text, 'label':test_label})))

# # pickle.dump(train_embedding, open(data_out + 'ds_' + moniker + '_ct_train_emb.pkl','wb'))

# vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
# X = vec.fit_transform(list(train_text) + list(test_text))
# X_train = vec.transform(train_text)
# X_test = vec.transform(test_text)

# clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000)
# clf.fit(X_train, train_label)
# result = classification_report(test_label, clf.predict(X_test), output_dict=True)
# print(result)