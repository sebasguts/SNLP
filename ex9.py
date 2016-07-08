import json
import numpy as np

#LOAD data

with open("kickstarter.json") as f:
    documents = [json.loads(d)["text"] for d in f.readlines()]
    

print("Loaded {} documents".format(len(documents)))
print("Here is one of them:")
print(documents[0])

#CREATEterm-document matrix

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .8, min_df=50)
term_doc_M = vectorizer.fit_transform([doc for doc in documents]).transpose()

#DECOMPSITION with SVD

from scipy.sparse.linalg import svds
u, s, v_trans = svds(term_doc_M, k = 140)

#TO DO: plot here
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

#TO DO decompose and compute reduced term space and document space

words_compressed = 0
docs_compressed = 0

#row normalize
from sklearn.preprocessing import normalize
words_compressed = normalize(words_compressed, axis = 1)

word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.iteritems()}

def closest_words(word_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]/sims[asort[0]]) for i in asort[1:]]

#TO DO: change input analyze output

closest_words("record")

#WORD2VEC ----------------------------------------------------

from nltk import word_tokenize
tok_docs=[word_tokenize(doc) for doc in documents]

#TO DO try different window sizes
from gensim import models
W2V_model = models.Word2Vec(tok_docs, window=5, min_count=5, workers=4, size=100)

#TO DO change input analyze output
W2V_model.most_similar(positive=["record"])

W2V_model.most_similar(positive=['daughter', 'man'], negative=['woman'], topn=1)



