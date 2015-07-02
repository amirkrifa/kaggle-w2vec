#!/usr/bin/env python

import re
import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

def review_to_wordlist( review, remove_stopwords=False ):
    # Remove HTML
    review_text = BeautifulSoup(review).get_text()
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # Return a list of words
    return(words)

def main():
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
                   quoting=3 )
    y = train["sentiment"]  
    logging.info('Cleaning and parsing movie reviews...')      
    traindata = []
    for i in xrange( 0, len(train["review"])):
        traindata.append(" ".join(review_to_wordlist(train["review"][i], True)))
    testdata = []
    for i in xrange(0,len(test["review"])):
        testdata.append(" ".join(review_to_wordlist(test["review"][i], True)))
    
    logging.info('vectorizing... ') 
    tfv = TfidfVectorizer(min_df=10,  
                          max_features=None, 
                          strip_accents='unicode', 
                          analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 4), 
                          use_idf=True,
                          smooth_idf=True,
                          sublinear_tf=True,
                          stop_words = 'english')
    
    tfv.fit(traindata)
    X_train = tfv.transform(traindata)
    X_test = tfv.transform(testdata)

    logging.info('Training ...')
    
    model = LogisticRegression(penalty='l2', 
                               dual=True, 
                               tol=0.0001, 
                               C=1, 
                               fit_intercept=True, 
                               intercept_scaling=1.0, 
                               class_weight=None, 
                               random_state=None)

    model.fit(X_train, y)

    logging.info('20 Fold CV Score: %f'%np.mean(cross_validation.cross_val_score(model, X_train, y, cv=20, scoring='roc_auc')))
    logging.info('Predicting test data ...')
    result = model.predict_proba(X_test)[:,1]
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'result.csv'), index=False, quoting=3)

if __name__ == '__main__':

    main()