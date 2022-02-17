# -*- coding: utf-8 -*-
import argparse
import os
import sys
import pandas
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

#command line arguments
print(sys.argv)
filePath = sys.argv[1]

#keep file path as txt file for inference
print(filePath)
with open(filePath + '/filePath.txt', 'w') as f:
    f.write(filePath)
    

# Negative Training Reviews w stopwords and assigning sentient
training_neg = filePath + '/train_neg.csv'
training_neg_df = pandas.read_csv(training_neg, delimiter = "\t", names=['reviews'])
training_neg_df['sentient'] = 0
training_neg_df['reviews'] = training_neg_df['reviews'].str.replace('\d+', '')
training_neg_df['reviews'] = training_neg_df['reviews'].apply(eval).apply(' '.join)

# Negative Training Reviews wo stopwords and assigning sentient
training_neg_ns = filePath + '/train_ns_neg.csv'
training_neg_ns_df = pandas.read_csv(training_neg_ns, delimiter = "\t", names=['reviews'])
training_neg_ns_df['sentient'] = 0
training_neg_ns_df['reviews'] = training_neg_ns_df['reviews'].str.replace('\d+', '')
training_neg_ns_df['reviews'] = training_neg_ns_df['reviews'].apply(eval).apply(' '.join)

# Positive Training Reviews w stopwords and assigning sentient
training_pos = filePath + '/train_pos.csv'
training_pos_df = pandas.read_csv(training_pos, delimiter = "\t", names=['reviews'])
training_pos_df['sentient'] = 1
training_pos_df['reviews'] = training_pos_df['reviews'].str.replace('\d+', '')
training_pos_df['reviews'] = training_pos_df['reviews'].apply(eval).apply(' '.join)

# Positive Training Reviews wo stopwords and assigning sentient
training_pos_ns = filePath + '/train_ns_pos.csv'
training_pos_ns_df = pandas.read_csv(training_pos_ns, delimiter = "\t", names=['reviews'])
training_pos_ns_df['sentient'] = 1
training_pos_ns_df['reviews'] = training_pos_ns_df['reviews'].str.replace('\d+', '')
training_pos_ns_df['reviews'] = training_pos_ns_df['reviews'].apply(eval).apply(' '.join)

#combine files for neg pos training to pass into models
reviews_joined = [training_pos_df, training_neg_df]
traing_reviews = pandas.concat(reviews_joined)
reviews_joined_ns = [training_pos_ns_df, training_neg_ns_df]
training_reviews_ns = pandas.concat(reviews_joined_ns)

#create pipeline model
pipeline_model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

#models for unigrams with stopwords
mnb_uni_param = {'vect__ngram_range': [(1, 1)],'clf__alpha': [1]}
mnb_uni = GridSearchCV(pipeline_model, mnb_uni_param, cv=10)
mnb_uni.fit(traing_reviews['reviews'], traing_reviews['sentient'])

#models for bigrams with stopwords
mnb_bi_param = {'vect__ngram_range': [(2, 2)],'clf__alpha': [1]}
mnb_bi = GridSearchCV(pipeline_model, mnb_bi_param, cv=10).fit(traing_reviews['reviews'], traing_reviews['sentient'])

#models for bigrams and unigrams with stopwords
mnb_uni_bi_param = {'vect__ngram_range': [(1, 2)],'clf__alpha': [1]}
mnb_uni_bi = GridSearchCV(pipeline_model, mnb_uni_bi_param, cv=10).fit(traing_reviews['reviews'], traing_reviews['sentient'])

#models for unigrams without stopwords
mnb_uni_ns_param = {'vect__ngram_range': [(1, 1)],'clf__alpha': [1]}
mnb_uni_ns = GridSearchCV(pipeline_model, mnb_uni_ns_param, cv=10).fit(training_reviews_ns['reviews'], training_reviews_ns['sentient'])

#models for bigrams without stopwords
mnb_bi_ns_param = {'vect__ngram_range': [(2, 2)],'clf__alpha': [1]}
mnb_bi_ns = GridSearchCV(pipeline_model, mnb_bi_ns_param, cv=10).fit(training_reviews_ns['reviews'], training_reviews_ns['sentient'])

#models for bigrams and unigrams without stopwords
mnb_uni_bi_ns_param = {'vect__ngram_range': [(1, 2)],'clf__alpha': [1]}
mnb_uni_bs_ns = GridSearchCV(pipeline_model, mnb_uni_bi_ns_param, cv=10).fit(training_reviews_ns['reviews'], training_reviews_ns['sentient'])

#putting the models into pickle files
pickle.dump(mnb_uni, open(filePath + '/mnb_uni.pkl', 'wb'))
pickle.dump(mnb_bi, open(filePath + '/mnb_bi.pkl', 'wb'))
pickle.dump(mnb_uni_bi, open(filePath + '/mnb_uni_bi.pkl', 'wb'))
pickle.dump(mnb_uni_ns, open(filePath + '/mnb_uni_ns.pkl', 'wb'))
pickle.dump(mnb_bi_ns, open(filePath + '/mnb_bi_ns.pkl', 'wb'))
pickle.dump(mnb_uni_bs_ns, open(filePath + '/mnb_uni_bi_ns.pkl', 'wb'))


print('Models are saved')