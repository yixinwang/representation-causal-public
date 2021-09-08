import requests, tarfile
from io import BytesIO
import pandas as pd
import pickle
import numpy as np

class Dataset:
    def __init__(self, X, y, vec, df, moniker):
        print('new dataset with %d records' % len(df))
        # display(df.head(1))
        self.X = X
        self.y = y
        self.vec = vec
        self.df = df
        self.feats = np.array(vec.get_feature_names())
        self.moniker = moniker 

        
class SentenceEdit:
    def __init__(self, remove_wd, sentence_idx, left_context, right_context, label):
        
        self.sentence_idx = sentence_idx
        self.remove_wd = remove_wd
        
        self.left_context = left_context
        self.right_context = right_context
        
        self.label = label
        
      
    def __repr__(self):
        " returns a printable string representation of an object"
        return '%s ||| %s ||| %s ||| %s \n' % (str(self.remove_wd), str(self.left_context), str(self.right_context), str(self.label))
    
    
def get_IMDB():
    """
    Get IMDB movie reviews from the link;
    Sentences labeled as pos and neg
    
    """
    data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

    web_response = requests.get(data_url, stream=True)
    gz_file = web_response.content # Content in bytes from requests.get, see comments below why this is used.
    zipfile = tarfile.open(fileobj=BytesIO(gz_file))
    neg_file = zipfile.extractfile('rt-polaritydata/rt-polarity.neg')
    pos_file = zipfile.extractfile('rt-polaritydata/rt-polarity.pos')
    df = pd.DataFrame([{'label': -1, 'text': t.decode("cp437").strip() } for t in neg_file] +
                      [{'label': 1, 'text': t.decode("cp437").strip()} for t in pos_file])

    return df


def get_kindle():
    """
    Kindle book reviews with sentiment labels;
    Pre-processed by:
        Split comments into sentences;
        Filter by keywords;
        Limit sentence length [5,50];
    """
#     df = pickle.load(open("/data/zwang/2020_S/EMNLP/V_5_doubleCheck_sentiment/kindle_sentiment_5_50.pickle",'rb'))
#     df = pickle.load(open("/data/zwang/2020_S/EMNLP/V_6_shortSents/kindle_short_sents.pickle",'rb'))
    df = pickle.load(open("/data/zwang/2020_S/Attention/matches/kindle/kindle_unique_sents.pkl",'rb'))
#     df = df.sample(frac=1)
    df.reset_index(drop=True,inplace=True)
        
    return df


def get_toxic_comment():
    df = pickle.load(open("/data/zwang/2020_S/Toxic/Concat_last4_emb/V_6_shortSents/toxic_short_sents.pickle",'rb'))
#     df = df.sample(frac=1)
    df.reset_index(drop=True,inplace=True)
    
    return toxic_df


def get_toxic_tw():
    """
    Toxic tweets from paper: Characterizing Variation in Toxic Language by Social Context
    """
    random.seed(42)
    df = pd.read_csv(open("/data/zwang/2020_S/Toxic/Data/TW/toxic_tweets.csv"))
    df['label'] = df['hostile'].apply(lambda x: 1 if x==1 else -1)
    
    return df[['id','text','label']]




