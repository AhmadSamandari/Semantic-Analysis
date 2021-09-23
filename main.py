import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

lemmatize = WordNetLemmatizer()

# have words in aaray

titles = [w.rstrip() for w in open('book_title.txt')]

#again we need to remove stopword

stopword = set(wo.rstrip() for wo in open('stopwords.txt'))


def tokenization(lin):
    lin.lower()  # to make them all lowercase
    token = nltk.tokenize.word_tokenize(lin)
    token = [x for x in token if x not in stopwords]
    token = [x for x in token if len(x)>2]
    return token

for string in titles:
    title_words = tokenization(string.text)

print(title_words)
