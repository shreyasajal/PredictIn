import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords') 
nltk.download('punkt')


from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from torch.nn import CosineSimilarity
import torch

cs = CosineSimilarity()
emb=SentenceTransformerDocumentEmbeddings('stsb-roberta-base')
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
def pre_process(corpus):
    # convert input corpus to lower case.
    corpus = corpus.lower()
    # collecting a list of stop words from nltk and punctuation form
    # string class and create single array.
    stopset = stopwords.words('english') + list(string.punctuation)
    # remove stop words and punctuations from string.
    corpus=remove_emoji(corpus)
    # word_tokenize is used to tokenize the input corpus in word tokens.
    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    corpus = unidecode(corpus)
    return corpus

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B/glove.6B.50d.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'word2vec.txt'
word_emb_model = KeyedVectors.load_word2vec_format(filename, binary=False)


from collections import Counter
import itertools

def map_word_frequency(document):
    return Counter(itertools.chain(*document))
    
def get_sif_feature_vectors(sentence1, sentence2, word_emb_model=word_emb_model):
    sentence1 = [token for token in sentence1.split() if token in word_emb_model.wv.vocab]
    sentence2 = [token for token in sentence2.split() if token in word_emb_model.wv.vocab]
    word_counts = map_word_frequency((sentence1 + sentence2))
    embedding_size = 50 # size of vectore in word embeddings
    a = 0.001
    sentence_set=[]
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word]) # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word_emb_model.wv[word])) # vs += sif * word_vector
        vs = np.divide(vs, sentence_length) # weighted average
        sentence_set.append(vs)
    return sentence_set

def get_feature_vectors(sentence1, sentence2, word_emb_model=word_emb_model):
    sentence1 = [token for token in sentence1.split() if token in word_emb_model.wv.vocab]
    sentence2 = [token for token in sentence2.split() if token in word_emb_model.wv.vocab]
    word_counts = map_word_frequency((sentence1 + sentence2))
    embedding_size = 50 # size of vectore in word embeddings
    
    sentence_set=[]
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = 1 # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word_emb_model.wv[word])) # vs += sif * word_vector
        vs = np.divide(vs, sentence_length) # weighted average
        sentence_set.append(vs)
    return sentence_set
    
from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]
    
    
    
    
    
    
    
    
def input_variables(post_content,about,headline):
    text=post_content.lower()
    
    numb_hashtags=len([tag.strip("#") for tag in text.split() if tag.startswith("#")])
    #classification
    labels = {"achievement":["achievement"],"info":["knowledge and facts","updates and announcements"],"insights":["insightful experiences and life lessons"],"job opening":["recruiting"],"call to action":["share with us in comments what are your opinions preferences and suggestions"]}
    label_emb = {}
    for lab in labels:
        label_emb[lab] = []
        for labi in labels[lab]:
            sen = Sentence(labi)
            emb.embed(sen)
            label_emb[lab].append(sen.embedding.reshape(1,-1))
    x=post_content
    sen = Sentence(x)
    emb.embed(sen)
    sen_emb = sen.embedding.reshape(1,-1)
    
    label_sims = {}
    
    for lab in label_emb:
        simi = 0
        for embd in label_emb[lab]:
            simi+=cs(embd,sen_emb)
        simi = simi/len(label_emb[lab])
        label_sims[lab] = simi
    
    max_lab = "other"
    max_sim = 0
    
    for lab in label_emb:
        if(label_sims[lab]>max_sim):
            max_sim = label_sims[lab].item()
            max_lab = lab
    post_type=max_lab  
    confidence=max_sim   
    #convert tweets to lower case
    
    
#url removes
    text=text.replace(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b','')
    text=text.replace(r'www\.\S+\.com','')

#removes retweets & cc
    text=text.replace(r'rt|cc', '')

#hashtags removes
    text=text.replace(r'#', '')

#user mention removes
    text=text.replace(r'@\S+', '')

#emoji 
    text=text.replace(r'[^\x00-\x7F]+', '')

#html tags
    text=text.replace(r'<.*?>', '')

#punctuation
    text=text.replace('[{}]'.format(string.punctuation), '')

#removes extra spaces
    text=text.replace(r'( )+', ' ').strip()
    
    
    
#relevance score
    info=about+headline
    sentence_1=info
    sentence_2=text
   
    sentence_1= pre_process(sentence_1)
    sentence_2= pre_process(sentence_2)
     
    sentence_set=get_feature_vectors(sentence_1, sentence_2, word_emb_model=word_emb_model)
    if np.isnan(sentence_set[0]).any():
        similarity=0
    elif np.isnan(sentence_set[1]).any():
        similarity=0   

    else:
        similarity=get_cosine_similarity(sentence_set[0], sentence_set[1])
    relevance_score=similarity
    content_len=len(text)
    
    return numb_hashtags,content_len,relevance_score,post_type,confidence    
    
    
