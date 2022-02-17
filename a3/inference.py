# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pickle
from gensim.models import Word2Vec

#Retrieve command line argument
print(sys.argv)
txt_file = sys.argv[1]

filePath = open('data/filePath.txt', "r").read()

with open(filePath + '/word2Vec.pkl', 'rb') as f:
  word_2_vec_model = pickle.load(f)

reviewsList = []

with open(txt_file) as infile:
  for line in infile:
    reviewsList.append(line.strip())
        
for i in range(len(reviewsList)):
  similar_words = word_2_vec_model.wv.most_similar(reviewsList[i], topn=20)

  print('\n')
  print('Given Word: ' + reviewsList[i])
  print('\n')

  for j in range(len(similar_words)):
    ranked_word = j + 1
    number = str(ranked_word)
    print('Similar Word ' + number)
    print(similar_words[j])
    print('\n')