# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 01:02:25 2019

@author: basil.p.sony
"""

###################### Importing all required packages ######################
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

####### Reading the data set
data = pd.read_csv('C:/Users/basil.p.sony/OneDrive - Accenture/Python Stuff/Python/Text Analytics/train_E6oV3lV.csv')


########################## Basic Feature extraction operations ###################

### Number of Words
data['word_count'] = data['tweet'].apply(lambda x: len(str(x).split(" ")))
data[['tweet','word_count']].head()

###Number of characters
data['char_count'] = data['tweet'].str.len() ## includes spaces
data[['tweet','char_count']].head()

### Average length of words
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['tweet'].apply(lambda x: avg_word(x))
data[['tweet','avg_word']].head()

### Number of stop words
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
data['stopwords'] = data['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['tweet','stopwords']].head()

### Number of special characters or hash tags
data['hastags'] = data['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['tweet','hastags']].head()

### Number of numerics
data['numerics'] = data['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['tweet','numerics']].head()

### Number of uppercase
data['upper'] = data['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['tweet','upper']].head()


########################## Basic Text pre processing operations ###################

### Converting to lower case
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['tweet'].head()

### Removing punctuations
data['tweet'] = data['tweet'].str.replace('[^\w\s]','')
data['tweet'].head()

### Removing stopwords
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['tweet'].head()

### Removing top 10 frequent/common words
freq_high = pd.Series(' '.join(data['tweet']).split()).value_counts()[:10]
freq_high
freq_high = list(freq_high.index)
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_high))
data['tweet'].head()

### Removing Bottom 10 rare words 
freq_low = pd.Series(' '.join(data['tweet']).split()).value_counts()[-10:]
freq_low
freq_low = list(freq_low.index)
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_low))
data['tweet'].head()

### Correctng spellings
from textblob import TextBlob
data['tweet'][:10].apply(lambda x: str(TextBlob(x).correct()))

### Tokenization
nltk.download('punkt')
TextBlob(data['tweet'][3]).words

### Steming
from nltk.stem import PorterStemmer
st = PorterStemmer()
data['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

### Lemmatization
from textblob import Word
nltk.download('wordnet')
data['tweet'] = data['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['tweet'].head()


########################## Advanced Text pre processing operations ###################

### N-grams
TextBlob(data['tweet'][0]).ngrams(2)
TextBlob(data['tweet'][0]).ngrams(3)

### Term Frequency
tf1 = (data['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1

### Inverse document frequency
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['tweet'].str.contains(word)])))
  
tf1

### Term Frequence - Inverse Document Frequency
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1
# Alternate sckit learn package available
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(data['tweet'])  
train_vect

### Bag of words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(data['tweet'])
train_bow

### Sentiment Analysis
data['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)
data['sentiment'] = data['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['tweet','sentiment']].head()


###### Topic Modelling ######
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."
# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]
# Cleaning and Pre processing
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]        
# Importing Gensim
import gensim
from gensim import corpora
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
# Results
print(ldamodel.print_topics(num_topics=3, num_words=3))
