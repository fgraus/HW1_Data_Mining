import re
from typing import Text
#import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

from apyori import apriori
from mlxtend.frequent_patterns import apriori

import numpy as np

def getDataToWork():
    trainText = getText('Resources/train.txt', training = False)
    testText = getText('Resources/test.txt',training = False, train = False)
    return trainText, testText

def getDataFromFile():
    return getText('Resources/train.txt')


def getText(path2File, training = True, train= True, chuncks = 0):

    content = ''
    with open(path2File, encoding='UTF-8') as text:
        content = text.read()
        text.close()

    content = re.sub('\'n','',content)

    lines = np.array(re.compile(r'#EOF').split(content))[:-1]

    # DELETE THIS LATER IT IS JUST FOR PICKING ONLY THE FIRST 5000 REVIEWS
    if training:
        lines = lines[:5000]

    evaluation = []
    reviews = []

    separationLine = re.compile(r'\t')
    for line in lines:
        try:
            if train:
                evaluation.append(int(separationLine.split(line)[0]))
                reviews.append(deleteUselessWords(separationLine.split(line)[1]))
            else:
                reviews.append(deleteUselessWords(line))
                evaluation.append(0.0)
        except IndexError:
            print('Error Index')
        except ValueError:
            print('Value error')
    
    return translate2Diccionary(reviews, evaluation, train= train)


def deleteUselessWords(text):

    cleaner = re.compile('<.*?>')

    text = re.sub('-', ' ', text)
    text = re.sub(cleaner, '', text)
    text = re.sub('\'s','',text)
    text = re.sub('\'t','',text)
    text = re.sub('\\s', ' ', text)
    text = re.sub(r'[^\w\s]','', text)
    text = re.sub(r'\n','', text)
    text = text.lower()

    lemmatizer = WordNetLemmatizer()
    newText = ''
    words = re.split(' ', text)
    for word in words:
        newText = newText + ' ' + lemmatizer.lemmatize(word)

    return newText

def translate2Diccionary(listReview, evaluation, train = True):

    global diccionary
    #aprioriRule(listReview, evaluation)
    if train:
        # delete words that appear 1 time
        diccionary = CountVectorizer(stop_words= ENGLISH_STOP_WORDS, max_features=10000)
        reviews = diccionary.fit_transform(listReview)
    else:
        reviews = diccionary.transform(listReview)
        del diccionary

    listWordTranslated = []
    # this should be mejorable
    for i in range(len(evaluation)):
        a = reviews[i].toarray()[0]
        # unitare the vector
        listWordTranslated.append([evaluation[i],a / np.sqrt(np.dot(a,a))])

    a = np.array(listWordTranslated, dtype=object)
    if train:
        np.random.shuffle(a)

    return a

##def aprioriRule(listReview, evaluation):

    listaTotal =[]
    for i in range(len(listReview)):
        palabra = re.split(' ',listReview[i])
        #palabra.append(evaluation[i])
        listaTotal.append(palabra)

    a = list(apriori(listaTotal,min_support=0.0045, min_confidence=0.8, min_lift=4.0))
    return 0



#getDataFromFile()