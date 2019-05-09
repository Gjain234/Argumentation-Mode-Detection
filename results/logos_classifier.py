import numpy as np
from preprocessing import *
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

logos_X = np.zeros((len(sentences), 2))
logos_Y = np.array(logos_labels)

#from https://blog.udemy.com/list-of-transition-words/
transitions_set = {'first', 'next', 'now', 'later', 'previously', 'afterward', 'whenever', 'meanwhile', 'during', 'one', 'first', 'second', 'next', 'moreover', 'finally', 'example', 'instance', 'illustrate', 'illustrates', 'demonstrate', 'demonstrates', 'because', 'since', 'effect', 'effects', 'affect', 'cause', 'caused', 'so', 'thus', 'therefore', 'hence', 'accordingly', 'reason', 'result', 'results', 'resulted', 'affected', 'affects'}

##########features for ethos

for i in range(len(sentences)):
    tokens = re.findall('[a-zA-Z]+', sentences[i].lower())
    for token in tokens:
        if token in transitions_set:
            logos_X[i,0] +=1
    doc = nlp(sentences[i])
    logos_X[i,1] = len(doc.ents)
#print(np.sum(ethos_X[:,0]))
#print(np.sum(ethos_X[:,1]))




train_samples = int(0.8*len(sentences))
X_train = logos_X[:train_samples,:]
X_test = logos_X[train_samples:,:]
y_train = logos_Y[:train_samples]
y_test = logos_Y[train_samples:]


x1true = np.take(X_train[:,0],np.where(y_train))
x2true = np.take(X_train[:,1],np.where(y_train))
x1false = np.take(X_train[:,0],np.where(y_train!=1))
x2false = np.take(X_train[:,1],np.where(y_train!=1))
plt.plot(x1false, x2false, 'r^', x1true,x2true, 'bo')
lr = LogisticRegression().fit(X_train,y_train)
plt.figtext(.6,.03, lr.score(X_test,y_test))
print(lr.score(X_test,y_test))
plt.show()
