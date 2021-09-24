import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

my_lemmatizer = WordNetLemmatizer()

# have words in array
titles = [w.rstrip() for w in open('book_title.txt')]

#again we need to remove stopword

stopword = set(wo.rstrip() for wo in open('stopwords.txt'))
stopword = stopword.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guid', 'essential', 'printed',
    'third', 'second', 'fourth'
}) 

def my_tokenization(s):
    s = s.lower()     # to make them all lowercase
    toky = nltk.tokenize.word_tokenize(s)
    print(toky)
    toky = [x for x in toky if len(x)>2]
    toky = [my_lemmatizer.lemmatize(t) for t in toky]
    toky = [x for x in toky if x not in stopwords]
    toky = [x for x in toky if not any(c.isdigit()) for c in x]  #remove the edition like first, second, ...
    return toky


vocab_dict_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_to_word_map = []


for string in titles:
    try:
        string = string.encode('ascii', 'ignore')  #some of them are not valid in ascii
        all_titles.append(string)
        title_words = my_tokenization(string)
        all_tokens.append(title_words)
        for tok in title_words:
            if tok not in vocab_dict_map:
                vocab_dict_map[tok] = current_index
                current_index +=1
                index_to_word_map.append(tok)

    except:
        pass
    
def token_to_vector(tittle):
    array = np.zeros(len(vocab_dict_map))
    for x in tittle:
        i = vocab_dict_map[x]
        array[i] = 1  #binary clustring
    return array

D = len(vocab_dict_map)
N = len(all_tokens)
X = np.zeros((D,N))
i = 0

for item in all_tokens:
    X[:, i] = token_to_vector(item)    #it is term dicument matrix
    i += 1

#print(all_tokens)
#SVD = TruncatedSVD()
#Z = SVD.fit_transform(X)

