#!/usr/bin/env python
# -*- coding: utf-8 -*-
    
from bs4 import BeautifulSoup 
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list

import logging
logging.basicConfig(level=logging.DEBUG)

#Do some very minor text preprocessing
def clean_text(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


#Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
#We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
#a dummy index of the review.
def labelizeReviews(reviews, label_type):
    import gensim
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


#Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def main():
    # Import the pandas package, then use the "read_csv" function to read
    # the labeled training data
    import pandas as pd       
    import gensim
    from sklearn.cross_validation import train_test_split
    import numpy as np

    logging.info('Loading data ...')
    train = pd.read_csv("/home/amir/kaggle/labeledTrainData.tsv", 
                        header=0,
                        delimiter="\t", 
                        quoting=3
                        )
    test = pd.read_csv("/home/amir/kaggle/testData.tsv", 
                        header=0,
                        delimiter="\t", 
                        quoting=3
                        )

    unsup_reviews = pd.read_csv("/home/amir/kaggle/unlabeledTrainData.tsv", 
                            header=0,
                            delimiter="\t", 
                            quoting=3
                            )

    x_train = [train["review"][i] for i in xrange(0, len(train['review']))]
    y_train = [train["sentiment"][i] for i in xrange(0, len(train['sentiment']))]
    x_test = [test["review"][i] for i in xrange(0, len(test['review']))]
    x_unsup_reviews = [unsup_reviews["review"][i] for i in xrange(0, len(unsup_reviews['review']))]

    logging.info('Labeling data ...')

    x_train = clean_text(x_train)
    x_test = clean_text(x_test)
    x_unsup_reviews = clean_text(x_unsup_reviews)

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    x_unsup_reviews = labelizeReviews(x_unsup_reviews, 'UNSUP')

    import random
    
    size = 300
    
    #instantiate our DM and DBOW models
    logging.info('instantiate our DM and DBOW models')
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
    
    logging.info('build vocab over all reviews')
    #build vocab over all reviews
    model_dm.build_vocab(np.concatenate((x_train, x_test, x_unsup_reviews)))
    model_dbow.build_vocab(np.concatenate((x_train, x_test, x_unsup_reviews)))
    
    #We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    logging.info('Training ...')
    all_train_reviews = np.concatenate((x_train, x_unsup_reviews))
    for epoch in range(5):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm])
        model_dbow.train(all_train_reviews[perm])

    
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

    #train over test set
    x_test = np.array(x_test)
    
    for epoch in range(5):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])
        
    #Construct vectors for test reviews
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    from sklearn.linear_model import SGDClassifier
    
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    x_test_predictions = lr.predict(x_test)
     
    logging.info('Saving results ...\n')
#     # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":x_test_predictions} )
#      
     # Use pandas to write the comma-separated output file
    output.to_csv("Doc2Vec_model_results.csv", index=False, quoting=3)


    

if __name__ == '__main__':
    main()
