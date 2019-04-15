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

ethos_X = np.zeros((len(sentences), 2))
ethos_Y = np.array(ethos_labels)

pronoun_set = {'i', 'me'}

##########features for ethos

for i in range(len(sentences)):
    tokens = re.findall('[a-zA-Z]+', sentences[i])
    doc = nlp(sentences[i])
    for token in tokens:
        if token in pronoun_set:
            ethos_X[i,0] += 1
    for entity in doc.ents:
        if entity.label_ == 'ORG':
            ethos_X[i,1] += 1
print(np.sum(ethos_X[:,0]))
print(np.sum(ethos_X[:,1]))


'''

train_samples = int(0.8*len(sentences))
X_train = ethos_X[:train_samples,:]
X_test = ethos_X[train_samples:,:]
y_train = ethos_Y[:train_samples]
y_test = ethos_Y[train_samples:]


x1true = np.take(X_train[:,0],np.where(y_train))
x2true = np.take(X_train[:,1],np.where(y_train))
x1false = np.take(X_train[:,0],np.where(y_train!=1))
x2false = np.take(X_train[:,1],np.where(y_train!=1))
plt.plot(x1false, x2false, 'r^', x1true,x2true, 'go')
lr = LogisticRegression().fit(X_train,y_train)
print(lr.score(X_test,y_test))
plt.show()
'''