import os
import nltk
import re
import math
import numpy as np
import collections
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#-----STOP_WORRDS AND STEMMING INITIALIZING-----#
port = PorterStemmer()
stop_words = set(stopwords.words('english'))

def pw(word_index):
    for i in word_index:
        print(i, ": ", word_index[i])

#-----OPTIMISING WORDS-----#
def optimise_word(word):
    word = word.lower()
    word = re.sub(r"[^a-zA-Z0-9]", "", word)
    if word.isalnum:
        #print(word)
        word = port.stem(word)
        return word

#-----CREATING WORD_INDEX FROM FILES-----#
def read_text_file(file_path, word_index, doc_id):
    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            for word in line.split():
                #if word not in stop_words:
                word = optimise_word(word)
                if word not in word_index.keys():
                    word_index[word] = {}
                    word_index[word][doc_id] = 1
                else:
                    if doc_id in word_index[word].keys():
                        word_index[word][doc_id] += 1
                    else:
                        word_index[word][doc_id] = 1
    return word_index

#-----PATH-----#
path = "D:\\4th year\IR\IR_assignment_2"
os.chdir(path)

word_index = {}
N = 0

#-----EXTRACTING FILES FROM PATH-----#
for file in os.listdir():
    if file.endswith('.txt'):
        file_path = f"{path}/{file}"
        file = file.split('.')
        N += 1
        file = file[0].split('t')
        file = int(file[1])
        read_text_file(file_path, word_index, file)

#-----SORTING WORD_INDEX-----#
word_index = collections.OrderedDict(sorted(word_index.items()))
pw(word_index)

#-----ENTER THE QUERY-----#
query = input("Enter a query here: ")
#query = "gold silver truck"

qry_index = {}
doc_mag = [0] * (N+1)

#------WORD FREQ IN QUERY------#
for word in query.split():
    word = optimise_word(word)
    if word not in qry_index.keys():
        qry_index[word] = 1
    else:
        qry_index[word] += 1

#-----CALCULATING TF-IDF FOR ALL WORDS-----#
for word in word_index:
    idtf = len(word_index[word].keys())
    if word in qry_index.keys():
        qry_index[word] *= (math.log10(N / idtf))    
    for doc_id in word_index[word]:
        word_index[word][doc_id] = (math.log10(N / idtf)) * (word_index[word][doc_id])

#-----MAGNITUDE OF DOC-----#
for i in range(1, N+1):
    for word in word_index:
       if i in word_index[word]:
           doc_mag[i] += (word_index[word][i] ** 2)
    doc_mag[i] = math.sqrt(doc_mag[i])
    
print("\nvector magnitude of docs: ", doc_mag)

#-----MAGNITUDE OF QUERY-----#
qry_mag = 0
for word in qry_index:
    qry_mag += qry_index[word] **2
qry_mag = math.sqrt(qry_mag)

print("query magnitude: ", qry_mag)

#-----DOT PRODUCT OF Q AND D-----#
dot = [0] * (N+1)
for i in range(1, N+1):
    for word in qry_index:
        if word in word_index.keys():
            if i in word_index[word]:
                dot[i] += word_index[word][i]*qry_index[word]
        else:
            continue

print("dot: ", dot)

#-----CONSINE SIMILARITY-----#
cos = []
for i in range(1, N+1):
    cos.append(dot[i] / (qry_mag * doc_mag[i]))
print("cos: ", cos)

#-----RANKING THE DOCUMENTS-----#
rank = np.flip(np.argsort(cos))

print("\nDoc no. Ranked to relevance: ")
for i in range(10):
    print(i+1,".) doc ", rank[i] + 1)