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


import torch
from transformers import * # here import bert

import warnings
warnings.filterwarnings("ignore")


from data_structure import Dataset #, get_IMDB, get_kindle

def get_kindle(data_path):
    """
    Only use selected_train data
    """
    df_kindle = pickle.load(open(data_path+'kindle_data.pkl','rb'))
    df_kindle = df_kindle[df_kindle['flag']=='selected_train']
    df_kindle.reset_index(drop=True,inplace=True)
    return df_kindle


class Counterfactual:
    def __init__(self, df_train, df_test, moniker):
        # display(df_train.head(1))
        self.moniker = moniker
        self.train = df_train
        self.test = df_test



def get_IMDB(data_path):
    """
    IMDB data split into sentences (len>1 and len<30)
    """
#     df_imdb = pickle.load(open(data_path+'imdb/imdb_sentences/imdb_sents.pkl','rb'))


    # ds_imdb_ct = pickle.load(open("/proj/sml/usr/yixinwang/representation-causal/src/causalrep_expms/aaai-2021-counterfactuals-main_full_data/Step1_data/ds_imdb.pkl", "rb")) # it is the same dataset as below

    ds_imdb_ct = pickle.load(open(data_path+"ds_imdb_sents.pkl", "rb"))
    df_imdb = ds_imdb_ct.train[['batch_id','text','label']]
    df_imdb.reset_index(drop=True,inplace=True)
    
    return df_imdb

def get_large_IMDB_sentences(data_path):
    """
    IMDB sentences from the original large dataset
    """
    df_imdb = pickle.load(open(data_path+'large_imdb_sents.pkl','rb'))
    return df_imdb


def simple_vectorize(df):
    """
    Vectorize text
    min_df = 10: agree with min_df for ite features
    """
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    X = vec.fit_transform(df.text)
    print(X.shape)
    y = df.label.values
    feats = np.array(vec.get_feature_names())
    
    return X, y, vec, feats

def get_top_terms(datasetX, datasety, coef_thresh, placebo_thresh, C=1):
    """
    Fit classifier, print top-200 terms;
    Top features: abs(coef) >= thresh
    Placebos: abs(coef) <= thresh
    """
    clf = LogisticRegression(class_weight='auto', C=C, solver='lbfgs', max_iter=1000)
    clf.fit(datasetX, datasety)
    
#     print_coef(clf, dataset.feats, n=100)
    #print('dummy coef= %.3f' % clf.coef_[0][dataset.vec.vocabulary_[DUMMY_TERM]])
    
    top_feature_idx = np.where(abs(clf.coef_[0]) >= coef_thresh)[0]
    placebo_feature_idx = np.where(abs(clf.coef_[0]) <= placebo_thresh)[0]
    
    return top_feature_idx, placebo_feature_idx, np.array([float("%.3f" % c) for c in clf.coef_[0]])


def get_top_terms_preproc(clf, vec, topn=0, min_coef=0.5, show_data=False):
    """
    - fit classifier
    - Select features by: topn or min_coef
    """
    df_vocab = pd.DataFrame({'term':vec.get_feature_names(),'coef':[float("%.3f" % c) for c in clf.coef_[0]]})
    
    if(topn == 0 and min_coef == 0):
        return df_vocab
    
    if(min_coef>0 and topn==0):
        df_top_terms = df_vocab[(df_vocab['coef']>= min_coef) | (df_vocab['coef'] < 0-min_coef)]
    elif(topn>0 and min_coef==0):
        df_vocab['coef_abs'] = df_vocab['coef'].apply(lambda x: abs(x))
        df_top_terms = df_vocab.sort_values(by=['coef_abs'], ascending=False).head(topn)
        df_top_terms.drop(columns=['coef_abs'],inplace=True)
    
    if(show_data):
        df_pos_terms = df_top_terms[df_top_terms['coef']>0]
        df_neg_terms = df_top_terms[df_top_terms['coef']<0]
        print("Features correlated with pos class: \n", [item['term']+'/'+str(item['coef']) for i, item in df_pos_terms.sort_values(by=['coef'], ascending=False).iterrows()])
        print("\nFeatures correlated with neg class: \n", [item['term']+'/'+str(item['coef']) for i, item in df_neg_terms.sort_values(by=['coef'], ascending=True).iterrows()])
    
    return df_top_terms

class SentenceEdit:
    def __init__(self, remove_wd, sentence_idx, left_context, right_context, context, label):
        
        self.sentence_idx = sentence_idx
        self.remove_wd = remove_wd
        self.context = context
        self.label = label
        self.left_context = left_context
        self.right_context = right_context
        
      
    def __repr__(self):
        " returns a printable string representation of an object"
        if(len(str(self.left_context).strip() + str(self.right_context).strip())==0):
            return '%s ||| %s \n' % (str(self.context), str(self.label))
        else:
            return '%s ||| %s ||| %s ||| %s \n' % (str(self.remove_wd), str(self.left_context), str(self.right_context), str(self.label))


def get_all_sentences(df):
    """
    Construct SentenceEdit object for all sentences
    """
    df['i_th'] = range(df.shape[0])
    all_sentences = []
    for ri, row in df.iterrows():
        sent_obj = SentenceEdit('', row['i_th'], '' , '',  row['text'], row['label'])
        all_sentences.append(sent_obj)
        
    return all_sentences


def load_bert():
    return (BertTokenizer.from_pretrained('bert-base-uncased'),
            BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True, output_attentions=True))

# bert representation of each sentence.
def embed_sentence(sentence, sentence_model, tokenizer):
    """
    # bert_tokenizer.vocab_size
    # bert_tokenizer.tokenize(sentence)
    # bert_tokenizer.convert_tokens_to_ids('on')
    # bert_tokenizer.convert_ids_to_tokens(102)
    # each sentence is encoded as a 3072 vec: 768 * 4 (concat last four layers)
    """
    with torch.no_grad():
        # sentence_model returns (logit output layer, pooler_output, hidden states, attentions)
        hidden_states = sentence_model(torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)]))[2]
        #last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        last_four_layers = [hidden_states[i] for i in (0,1,2,3)] # list of 4 element, each element is [1,16,768]
        # cast layers to a tuple and concatenate over the last dimension
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1) # [1,16,3072]
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze() # 3027
        #return(torch.mean(hidden_states[-1], dim=1).squeeze()) # average word embeddings in last layer
        return cat_sentence_embedding.numpy()                         # average last 4 layers
        
def embed_all_sentences(sentences, bert_tokenizer=None, sentence_model=None):
    """
    Each sentence is an object of SentenceEdit;
    Adding embedding attribute for the object;
    """
    if not bert_tokenizer:        
        bert_tokenizer, sentence_model = load_bert()
    for s in tqdm(sentences):
#         s.left_embedding = embed_sentence(s.left_context, sentence_model, bert_tokenizer)
#         s.right_embedding = embed_sentence(s.right_context, sentence_model, bert_tokenizer)
        s.context_embedding = embed_sentence(s.context, sentence_model, bert_tokenizer)
        s.word_embedding = embed_sentence(s.remove_wd, sentence_model, bert_tokenizer)

def load_data(moniker, data_path):
    if(moniker == 'kindle'):
        ds = pickle.load(open(data_path+"ds_kindle.pkl", "rb"))
    elif(moniker == 'imdb'):
        ds = pickle.load(open(data_path+"ds_imdb.pkl", "rb"))
    elif(moniker == 'imdb_sents'):
        ds = pickle.load(open(data_path+"ds_imdb_sents.pkl", "rb"))
    return ds

def organize_data(ds, limit=''):
    """
    Organize data for easy use in the evaluation
    'original','original+ctf_by_predicted_causal','original+ctf_by_annotated_causal','original+ctf_by_all_causal','original+ctf_by_human'
    
    """
    train_data = {}
    if(limit == 'ct'): # only those have counterfactuals as original data
        ds.train['len_ct_text_causal'] = ds.train['ct_text_causal'].apply(lambda x: len(x.strip()))
        df_select = ds.train[ds.train['len_ct_text_causal'] > 0]
        train_data['original'] = {'text':df_select.text.values, 'label':df_select.label.values}
    else:
        train_data['original'] = {'text':ds.train.text.values, 'label':ds.train.label.values}
    
    if(ds.moniker == 'imdb'):
        train_data['ctf_by_human'] = {'text':ds.train.ct_text_amt.values, 'label':ds.train.ct_label.values}
        train_data['original+ctf_by_human'] = {'text':list(train_data['original']['text'])+list(ds.train.ct_text_amt.values), 'label':list(train_data['original']['label'])+list(ds.train.ct_label.values)}
    
    elif(ds.moniker == 'imdb_sents'):
        train_data['ctf_by_human'] = {'text':ds.train_ct.text.values, 'label':ds.train_ct.label.values}
        train_data['original+ctf_by_human'] = {'text':list(train_data['original']['text'])+list(ds.train_ct.text.values), 'label':list(train_data['original']['label'])+list(ds.train_ct.label.values)}
        
    for flag, col in zip(['predicted_causal','annotated_causal','all_causal'],['identified_causal','causal','all_causal']):
        ds.train['len_ct_text_'+col] = ds.train['ct_text_'+col].apply(lambda x: len(x.strip()))
        df_train_ct_text_flag = ds.train[ds.train['len_ct_text_'+col] > 0]
        train_data['ctf_'+flag] = {'text':df_train_ct_text_flag['ct_text_'+col].values, 'label':df_train_ct_text_flag.ct_label.values}
        train_data['original+ctf_by_'+flag] = {'text':list(train_data['original']['text'])+list(df_train_ct_text_flag['ct_text_'+col].values),'label':list(train_data['original']['label'])+list(df_train_ct_text_flag.ct_label.values)}

    
    test_data = {}
    ds.test['len_ct_text_causal'] = ds.test['ct_text_causal'].apply(lambda x: len(x.strip()))
    df_test_ct_text_causal = ds.test[ds.test.len_ct_text_causal > 0]
    
    test_data['Original'] = {'text': ds.test.text.values, 'label': ds.test.label.values}
    test_data['ct_causal'] = {'text': df_test_ct_text_causal.ct_text_causal.values, 'label': df_test_ct_text_causal.ct_label.values}
    
    
    if(ds.moniker == 'imdb_sents'):
        test_data['Counterfactual'] = {'text': ds.test_ct.text.values, 'label': ds.test_ct.label.values}
    else:
        test_data['Counterfactual'] = {'text': ds.test.ct_text_amt.values, 'label': ds.test.ct_label.values}
        
    if(ds.moniker == 'kindle'):
        test_data['original_selected'] = {'text': ds.select_test.text.values, 'label': ds.select_test.label.values}
        test_data['ct_amt_selected'] = {'text': ds.select_test.ct_text_amt.values, 'label': ds.select_test.ct_label.values}
    
    return train_data, test_data

def fit_classifier(train_text, train_label, test_text, test_label, report=True, train='comb'):
    """
    Given training data and test data
    """
    if(len(train_text) == 0 or len(test_text) == 0): # not generating any counterfactual examples
        return 0.0
    
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    if(train == 'comb'):
        X = vec.fit_transform(list(train_text) + list(test_text))
        X_train = vec.transform(train_text)
        X_test = vec.transform(test_text)
    elif(train == 'train'):
        X_train = vec.fit_transform(list(train_text))
        X_test = vec.transform(test_text)
    
    clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, train_label)
    
    
    if(report):
        print(classification_report(test_label, clf.predict(X_test)))
        return clf, vec
    else:
        result = classification_report(test_label, clf.predict(X_test), output_dict=True)
        return float('%.3f' % result['accuracy'])


def classification_performance(train_data, test_data):
    """
    train: original, ct_auto, original+ct_auto
    test: original, ct_auto, ct_amt
    ct_text_auto: if no causal words or no antonym substitutions, then not use it;
    Not every text could generate a counterfactual text;
    """
    
    df_result = pd.DataFrame({'Sample_size':[0]*5,
                              'Original':[0]*5,
                              'Counterfactual':[0]*5})
    
    df_result.rename(index={i: f for i, f in enumerate(['original','original+ctf_by_predicted_causal','original+ctf_by_annotated_causal','original+ctf_by_all_causal','original+ctf_by_human'])}, inplace=True)

        
    for train_flag, test_flag in product(['original','original+ctf_by_predicted_causal','original+ctf_by_annotated_causal','original+ctf_by_all_causal','original+ctf_by_human'],['Original','Counterfactual']):
        try:
            df_result.loc[train_flag,'Sample_size'] = len(train_data[train_flag]['label'])
            df_result.loc[train_flag,test_flag] = fit_classifier(train_data[train_flag]['text'],  train_data[train_flag]['label'],test_data[test_flag]['text'], test_data[test_flag]['label'], report=False)
        except: # no human annotated counterfactual training data for kindle  
            df_result.loc[train_flag,'Sample_size'] = np.NaN
            df_result.loc[train_flag,test_flag] = np.NaN
    
    return df_result

def pre_process_imdb(data_file):
    """
    Pre-process to get original text and counterfactual text
    """
    
    df = pd.read_csv(data_file, sep='\t')

    
    combined_text = df.Text.values
    combined_batch_id = df.batch_id.values
    
    org_idx = [i for i in range(df.shape[0]) if(i % 2 == 0)]
    ct_idx = [i for i in range(df.shape[0]) if(i % 2 != 0)]
    
    org_batch_id = combined_batch_id[org_idx]
    ct_batch_id = combined_batch_id[ct_idx]
    if np.any(org_batch_id != ct_batch_id):
        print('Error: batch id not match!')
        return
    
    data = {}
    data['batch_id'] = org_batch_id
    data['text'] = combined_text[org_idx]
    data['ct_text_amt'] = combined_text[ct_idx]
    data['label'] = df.Sentiment.values[org_idx]
    data['ct_label'] = df.Sentiment.values[ct_idx]
    df_data = pd.DataFrame(data)
    
    map_lb = {'Positive':1, 'Negative':-1}
    df_data.replace({'label':map_lb, 'ct_label':map_lb}, inplace=True)

    return df_data


def select_sents(df, data_path):
    """
    - Select test sentences that contain one of the causal terms from full vocab
    """
    df_antonym_vocab = pd.read_csv(data_path+'kindle_vocab_antonym_causal.csv')
    keywords = list(df_antonym_vocab[df_antonym_vocab.causal == 1].term.values)
    
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    X = vec.fit_transform(df.text.values)
    y = df.label.values
    
    wd_sents = {}
    sent_idx = set()
    for wd in keywords:
        try:
            s_idx = np.nonzero(X[:,vec.vocabulary_[wd]])[0]
            wd_sents[wd] = s_idx
            sent_idx.update(s_idx)
        except:
            continue
    
    return df.iloc[list(sent_idx)]

def get_data(moniker, data_path):
    """
    - get kindle or imdb from different files
    """
    if(moniker == 'kindle'):
        df_kindle = pickle.load(open(data_path+"kindle_data.pkl",'rb'))
        df_train = df_kindle[df_kindle['flag']=='selected_train']
        df_test = df_kindle[df_kindle['flag']=='test']
        df_antonym_vocab = pd.read_csv(data_path+'kindle_vocab_antonym_causal.csv')
        df_identified_causal = pd.read_csv(data_path+'kindle_identified_causal.csv')
    elif(moniker == 'imdb'):
        df_train = pre_process_imdb(data_file = data_path + "train_paired.tsv")
        df_test = pre_process_imdb(data_file = data_path + "test_paired.tsv")
        df_antonym_vocab = pd.read_csv(data_path+'imdb_vocab_antonym_causal.csv')
        df_identified_causal = pd.read_csv(data_path+'imdb_identified_causal.csv')
    elif(moniker == 'imdb_sents'):
        df_train = pickle.load(open(data_path+"train_paired_sents.pkl", 'rb'))
        df_test = pickle.load(open(data_path+"test_paired_sents.pkl", 'rb'))
        df_antonym_vocab = pd.read_csv(data_path+'imdb_vocab_antonym_causal.csv')
        df_identified_causal = pd.read_csv(data_path+'imdb_identified_causal.csv')
        
    return df_train, df_test, df_antonym_vocab, df_identified_causal

def get_antonyms(vocab, causal_words):
    """
    - antonyms: top term with opposite coefficient;
    - get antonyms for all words in the vocab
    - Help provide more options for manually edit counterfactual examples
    - # 90 min for imdb vocab
    """
    term_antonyms = {}
    for ti, term in enumerate(causal_words):
        try:
            term_coef = vocab[term]

            ant_terms = {} # antonym and its coef
            for ant in dictionary.antonym(term):
                if (ant in vocab) and (term_coef * vocab[ant] < 0): # opposite coef, 
                    ant_terms[ant] = vocab[ant]

            if(len(ant_terms) == 0):
                for syn in dictionary.synonym(term):
                    if(len(re.findall('\w+', syn)) == 1):
                        for ant in dictionary.antonym(syn):
                            if (ant in vocab) and (ant != term) and (term_coef * vocab[ant] < 0): # 
                                ant_terms[ant] = vocab[ant]
        except:
            continue
    
        term_antonyms[term] = ant_terms
        
    return term_antonyms


def fit_classifier(train_text, train_label, test_text, test_label, report=True, train='comb'):
    """
    Fit a basic binary classifier
    """
    
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    if(train == 'comb'):
        X = vec.fit_transform(list(train_text) + list(test_text))
        X_train = vec.transform(train_text)
        X_test = vec.transform(test_text)
    elif(train == 'train'):
        X_train = vec.fit_transform(list(train_text))
        X_test = vec.transform(test_text)
        
    clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, train_label)
    
    if(report):
        print(classification_report(test_label, clf.predict(X_test)))
        return clf, vec
    else:
        result = classification_report(test_label, clf.predict(X_test), output_dict=True)
        return float('%.3f' % result['accuracy'])

# def get_top_terms(clf, vec, topn=0, min_coef=0.5, show_data=False):
#     """
#     - fit classifier
#     - Select features by: topn or min_coef
#     """
#     df_vocab = pd.DataFrame({'term':vec.get_feature_names(),'coef':[float("%.3f" % c) for c in clf.coef_[0]]})
    
#     if(topn == 0 and min_coef == 0):
#         return df_vocab
    
#     if(min_coef>0 and topn==0):
#         df_top_terms = df_vocab[(df_vocab['coef']>= min_coef) | (df_vocab['coef'] < 0-min_coef)]
#     elif(topn>0 and min_coef==0):
#         df_vocab['coef_abs'] = df_vocab['coef'].apply(lambda x: abs(x))
#         df_top_terms = df_vocab.sort_values(by=['coef_abs'], ascending=False).head(topn)
#         df_top_terms.drop(columns=['coef_abs'],inplace=True)
    
#     if(show_data):
#         df_pos_terms = df_top_terms[df_top_terms['coef']>0]
#         df_neg_terms = df_top_terms[df_top_terms['coef']<0]
#         print("Features correlated with pos class: \n", [item['term']+'/'+str(item['coef']) for i, item in df_pos_terms.sort_values(by=['coef'], ascending=False).iterrows()])
#         print("\nFeatures correlated with neg class: \n", [item['term']+'/'+str(item['coef']) for i, item in df_neg_terms.sort_values(by=['coef'], ascending=True).iterrows()])
    
#     return df_top_terms

def identify_causal_words(df, df_causal_terms, flag='causal', show_data=True):
    """
    Identify causal words in each sentence
    - Use CSR matrix from CountVectorizer instead of regular expression
    - flag = 'causal' or flag = 'bad' or flag='top'
    """
    df[flag+'_wds'] = df['text'].apply(lambda x: [wd for wd in re.findall('\w+', x.lower()) if wd in df_causal_terms.term.values])
    df['n_'+flag+'_wds'] = df[flag+'_wds'].apply(lambda x: len(x))
    
    if(show_data):
        print("%d out of %d sentences include %d %s words" % (df[df['n_'+flag+'_wds']>0].shape[0], df.shape[0], df_causal_terms.shape[0], flag))

def generate_ct_sentences(df, df_causal_terms, flag='causal'):
    """
    Generate counterfactual sentences for those contain causal words:
        - substitute all the causal words to antonyms;
        - antonyms: top term with opposite coefficient;
        - If no antonyms, keep the original causal word;
    """
    random.seed(42)
    
    all_ct_wds = []
    for ri, row in df.iterrows():
        if row['n_'+flag+'_wds'] > 0:
            words = re.findall('\w+', row.text.lower())
            new_wds = []
            ct_wds = []
            for wd in words:
                if(wd in df_causal_terms.term.values):
                    # randomly select antonym that has equal coef with current word
                    sub_w = list(df_causal_terms[df_causal_terms['term'] == wd].antonyms.values[0].keys())

                    if(len(sub_w) == 1):
                        ct_wd = str(sub_w[0])
                    elif(len(sub_w) > 1):
                        ct_wd = str(random.sample(sub_w,1)[0])
                    else: # if no antonyms then remove current word
                        ct_wd = wd
                        
                    new_wds.append(ct_wd)
                    ct_wds.append(ct_wd)
                else:
                    new_wds.append(wd)
                
            if(new_wds == words): # no antonym for the causal word
                all_ct_wds.append([])
                df.loc[ri, 'ct_text_'+flag] = ' '
            else:    
                all_ct_wds.append(ct_wds)
                df.loc[ri, 'ct_text_'+flag] = ' '.join(new_wds)
        else:
            all_ct_wds.append([])
            df.loc[ri, 'ct_text_'+flag] = ' '
        
        
    df['ct_'+flag+'_wds'] = all_ct_wds

def run_experiment(moniker,coef_thresh, data_path, data_out):
    """
    1. Get train and test data from file and construct Counterfactual object
    2. Get top words
    3. Annotate/predict causal words
    4. Generate antonyms for causal words
    5. Automatically generate counterfactual samples for both training and testing data
    """
    random.seed(42)
    
    print("Experiments for %s" % moniker)

    # 1. Get train and test data from file and construct Counterfactual object
    if(moniker == 'imdb_sents'):
        df_train_comb, df_test_comb, df_antonym_vocab, df_identified_causal = get_data(moniker,data_path)
        df_train = df_train_comb[df_train_comb['flag']=='original'][['batch_id','text','label']]
        df_test = df_test_comb[df_test_comb['flag']=='original'][['batch_id','text','label']]
        ds = Counterfactual(df_train, df_test, moniker)

        ds.train_ct = df_train_comb[df_train_comb['flag']=='counterfactual'][['batch_id','text','label']]
        ds.test_ct = df_test_comb[df_test_comb['flag']=='counterfactual'][['batch_id','text','label']]
    else:
        df_train, df_test, df_antonym_vocab, df_identified_causal = get_data(moniker,data_path)
        ds = Counterfactual(df_train, df_test, moniker)
        
        if(moniker == 'kindle'):
            ds.select_test = select_sents(df_test, data_path)
    
    ds.identified_causal_terms = df_identified_causal[df_identified_causal.identified_causal == 1]
        
    print('Train: %s' % str(Counter(df_train.label).items()))
    print('Test: %s' % str(Counter(df_test.label).items()))

    # 2. Get true causal terms from pre-annotated file
    clf, vec = fit_classifier(train_text = df_train.text.values, train_label = df_train.label.values,
                              test_text = df_test.text.values, test_label=df_test.label.values, 
                              report=True, train='train')
    
    vocab = get_top_terms_preproc(clf, vec, topn=0, min_coef=0, show_data=False)
    
    ds.antonym_vocab = df_antonym_vocab
    ds.all_causal_terms = ds.antonym_vocab[(ds.antonym_vocab.causal == 1) & (ds.antonym_vocab.term.isin(vocab.term.values))]
    ds.all_causal_terms['antonyms'] = ds.all_causal_terms['antonyms'].apply(lambda x: eval(x))
    
    # 3. Get top words
    clf, vec = fit_classifier(train_text = df_train.text.values, train_label = df_train.label.values,
                                   test_text = df_test.text.values, test_label=df_test.label.values, 
                              report=True, train='train')

    ds.top_terms = get_top_terms_preproc(clf, vec, topn=0, min_coef=coef_thresh, show_data=True)

    # Number of top terms not covered in the full vocab
    missing_terms = [term for term in ds.top_terms.term if term not in ds.antonym_vocab.term.values]
    print('\n%d top terms: %d pos, %d neg, %d missing from full vocab\n' % (ds.top_terms.shape[0], ds.top_terms[ds.top_terms.coef>0].shape[0], ds.top_terms[ds.top_terms.coef<0].shape[0], len(missing_terms)))
    print('Missing terms:', missing_terms)

    # 3. Assign causal label to top words (load from pre-annotated file)
    ds.top_terms['causal'] = [ds.antonym_vocab[ds.antonym_vocab['term'] == item.term].causal.values[0] if item.term in ds.antonym_vocab.term.values else 0 for i, item in ds.top_terms.iterrows()]
    
    # 4. Get antonyms for causal words    
    ds.top_terms['antonyms'] = [eval(ds.antonym_vocab[ds.antonym_vocab['term'] == item.term].antonyms.values[0]) if item.term in ds.antonym_vocab.term.values else {} for i, item in ds.top_terms.iterrows()]
    ds.top_terms['n_antonyms'] = ds.top_terms['antonyms'].apply(lambda x: len(x))
    df_causal_terms = ds.top_terms[ds.top_terms['causal'] == 1]
    df_bad_terms = ds.top_terms[ds.top_terms['causal'] == 0]
    ds.identified_causal_terms['antonyms'] = [eval(ds.antonym_vocab[ds.antonym_vocab['term'] == item.term].antonyms.values[0]) if item.term in ds.antonym_vocab.term.values else {} for i, item in ds.identified_causal_terms.iterrows()]

    print('\nGet antonyms for %d out of %d causal terms' % (df_causal_terms[df_causal_terms['n_antonyms'] > 0].shape[0], ds.top_terms[ds.top_terms['causal'] == 1].shape[0]))
    print('Closest opposite match identified causal terms: %d out of %d\n' % (ds.identified_causal_terms[ds.identified_causal_terms.causal==1].shape[0],ds.identified_causal_terms.shape[0]))

    # 5. Automatically generate counterfactual samples for both training and testing data
    for flag, df_ct_terms in zip(['causal','bad','top','identified_causal','all_causal'],[df_causal_terms, df_bad_terms, ds.top_terms, ds.identified_causal_terms, ds.all_causal_terms]):
        identify_causal_words(ds.train, df_ct_terms, flag, show_data=True)
        generate_ct_sentences(ds.train, df_ct_terms, flag)

        identify_causal_words(ds.test, df_ct_terms, flag, show_data=True)
        generate_ct_sentences(ds.test, df_ct_terms,flag)
    
    
    if(moniker == 'kindle'):
        df_annotate_ct = pd.read_csv(data_path+'kindle_ct_edit_500.csv')
        ds.test['ct_text_amt'] = [df_annotate_ct[df_annotate_ct['id']==idx]['ct_text_amt'].values[0] for idx in ds.test.index.values]
        ds.select_test['ct_text_amt'] = ds.test.loc[list(ds.select_test.index.values)]['ct_text_amt'].values
        ds.select_test['ct_label'] = ds.select_test['label'].apply(lambda x: 0-x)
    if(moniker == 'kindle' or moniker == 'imdb_sents'):
        ds.train['ct_label'] = ds.train['label'].apply(lambda x: 0-x)
        ds.test['ct_label'] = ds.test['label'].apply(lambda x: 0-x)

    # display(ds.test.head(2))


    if(moniker == 'kindle'):
        pickle.dump(ds, open(data_out+"ds_kindle.pkl", "wb"))
    elif(moniker == 'imdb'):
        pickle.dump(ds, open(data_out+"ds_imdb.pkl", "wb"))
    elif(moniker == 'imdb_sents'):
        pickle.dump(ds, open(data_out+"ds_imdb_sents.pkl", "wb"))

    return ds




def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()