#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import string


# In[6]:


pip install sklearn


# In[3]:


df=pd.read_csv('pd.csv',low_memory=False)


# In[18]:


d=df[['about','headline','location','content']][:15]


# In[19]:


d['about'][0]


# In[115]:


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



# In[15]:


wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text


def tokenization(text):
    text = re.split('\W+', text)
    return text


stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


# In[56]:


import nltk
nltk.download('wordnet')


# In[79]:


d['content'].fillna('',inplace=True)


# In[76]:


d.info()


# In[88]:


d['about']= d['about'].apply(lambda x: remove_punct(x.lower()))
d['about']= d['about'].apply(lambda x: remove_emoji(x.lower()))
d['about']= d['about'].apply(lambda x: remove_html(x.lower()))
d['about']= d['about'].apply(lambda x: remove_URL(x.lower()))

d['tokenized'] = d['about'].apply(lambda x: tokenization(x.lower()))
d['No_stopwords'] = d['tokenized'].apply(lambda x: remove_stopwords(x))
d['lemmatized'] = d['No_stopwords'].apply(lambda x: lemmatizer(x))


# In[70]:


d['lemmatized'][0]


# In[86]:


d['content']= d['content'].apply(lambda x: remove_punct(x.lower()))
d['content']= d['content'].apply(lambda x: remove_emoji(x.lower()))
d['content']= d['content'].apply(lambda x: remove_html(x.lower()))
d['content']= d['content'].apply(lambda x: remove_URL(x.lower()))
d['tokenized'] = d['content'].apply(lambda x: tokenization(x.lower()))
d['No_stopwords'] = d['tokenized'].apply(lambda x: remove_stopwords(x))
d['lemmatized_'] = d['No_stopwords'].apply(lambda x: lemmatizer(x))


# In[16]:


corpus1 = ["A girl is styling her hair.", "A girl is brushing her hair."]
lemmatizer(x for x in corpus1)


# In[85]:


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


# In[17]:


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
for x in corpus1:
    
    words = word_tokenize(x)
    lemmatizer.lemmatize(w for w in words)


# In[100]:


jaccard_similarity(d['lemmatized'][0], d['lemmatized_'][8])


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
# sentence pair

for c in range(len(corpus)):
    corpus[c] = pre_process(corpus[c])
# creating vocabulary using uni-gram and bi-gram
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf_vectorizer.fit(corpus)
feature_vectors = tfidf_vectorizer.transform(corpus)


# In[63]:


df['content'].fillna('',inplace=True)


# In[49]:


corpus[0]


# In[27]:


corpus[-1]


# In[116]:


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string
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


# In[117]:


corpus[2]


# In[40]:


from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B/glove.6B.50d.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'word2vec.txt'
word_emb_model = KeyedVectors.load_word2vec_format(filename, binary=False)


# In[44]:


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


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


# In[90]:


ss=get_sif_feature_vectors(corpus[2], corpus[-1], word_emb_model=word_emb_model)


# In[106]:


corpus[2]==''


# In[50]:


df['about'].fillna('',inplace=True)


# In[54]:


df['headline'].fillna('',inplace=True)
info=[]
for i in range(df.shape[0]):
    a=df['about'][i]+df['headline'][i]
    info.append(a)


# In[59]:


df['info']=pd.Series(np.array(info))


# In[133]:


relevance=[]
for i in range(df.shape[0]):
    sentence_1=df['info'][i]
    sentence_2=df['content'][i]
   
    sentence_1= pre_process(sentence_1)
    sentence_2= pre_process(sentence_2)
     
    sentence_set=get_sif_feature_vectors(sentence_1, sentence_2, word_emb_model=word_emb_model)
    if np.isnan(sentence_set[0]).any():
        similarity=0
    elif np.isnan(sentence_set[1]).any():
        similarity=0   

    else:
        similarity=get_cosine_similarity(sentence_set[0], sentence_set[1])
    
    relevance.append(similarity)
df['relevance_score']=pd.Series(np.array(relevance))


# In[134]:


df[['info','content','relevance_score']]

