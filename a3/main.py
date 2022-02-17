# -*- coding: utf-8 -*-

import argparse
import sys
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords') #Needed for first time use

#Retrieve command line argument
print(sys.argv)
filePath = sys.argv[1]

neg = filePath + '/neg.txt'
pos = filePath + '/pos.txt'

#Combine pos.txt and neg.txt that are saved to my google drive
filenames = [neg, pos]
with open(filePath + '/negpos.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

with open(filePath + '/filePath.txt', 'w') as f:
  f.write(filePath)

reviews = []

tokenizer = RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))


with open (filePath + '/negpos.txt') as fin:
  for line in fin:
    wordsFiltered = []
    reviewList = []
    tokens = tokenizer.tokenize(line)
    for t in tokens:
      if t not in stopWords:
        reviewList.append(t)
    reviews.append(reviewList)

word_2_vec_model = Word2Vec(sentences=reviews, window=5, min_count=1, workers=4)

pickle.dump(word_2_vec_model, open(filePath + '/word2Vec.pkl', 'wb'))

print('word2Vec.pkl Created')