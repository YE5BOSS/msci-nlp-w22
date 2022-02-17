# -*- coding: utf-8 -*-
import argparse
import os
import sys
import pickle

def model_run(model_type):

  if(model_type == 'mnb_uni'):
    with open(filePath + '/mnb_uni.pkl', 'rb') as f:
      model = pickle.load(f)
    return model

  elif(model_type == 'mnb_bi'):
    with open(filePath + '/mnb_bi.pkl', 'rb') as f:
      model = pickle.load(f)
    return model

  elif(model_type == "mnb_uni_bi"):
    with open(filePath + '/mnb_uni_bi.pkl', 'rb') as f:
      model = pickle.load(f)
    return model

  elif(model_type == 'mnb_bi_ns'):
    with open(filePath + '/mnb_bi_ns.pkl', 'rb') as f:
      model = pickle.load(f)
    return model

  elif(model_type == 'mnb_uni_ns'):
    with open(filePath + '/mnb_uni_ns.pkl', 'rb') as f:
      model = pickle.load(f)
    return model

  elif(model_type == 'mnb_uni_bi_ns'):
    with open(filePath + '/mnb_uni_bi.pkl', 'rb') as f:
      model = pickle.load(f)
    return model

  else: 
    print('Invalid Model Chosen')

#Inputs to call the function
print(sys.argv)
txt_file = sys.argv[1]
model_type = sys.argv[2]
filePath = open('data/filePath.txt', "r").read()
model = model_run(model_type)
reviews = []

with open(txt_file) as infile:
    for line in infile:
      reviews.append(line.strip())
for i in range(len(reviews)):
  if(model.predict([reviews[i]]) == 0):
    print("Negative Sentient")
  if(model.predict([reviews[i]]) == 1):
    print("Positive Sentient")