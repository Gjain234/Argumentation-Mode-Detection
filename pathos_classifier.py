import numpy as np
from preprocessing import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression


pathos_X = np.zeros((len(sentences), 2))
pathos_Y = np.array(pathos_labels)
#Y = Y.reshape(len(sentences),1)
sid = SentimentIntensityAnalyzer()

pronoun_set = {'you', 'your', 'yourself', 'yours'}

##########features for pathos

#pronoun counts
for i in range(len(sentences)):
    tokens = re.findall('[a-zA-Z]+', sentences[i].lower())
    for token in tokens:
        if token in pronoun_set:
            pathos_X[i,0] +=1
    pathos_X[i,1] = abs(sid.polarity_scores(sentences[i])['compound'])


train_samples = int(0.8*len(sentences))
X_train = pathos_X[:train_samples,:]
X_test = pathos_X[train_samples:,:]
y_train = pathos_Y[:train_samples]
y_test = pathos_Y[train_samples:]


x1true = np.take(X_train[:,0],np.where(y_train))
x2true = np.take(X_train[:,1],np.where(y_train))
x1false = np.take(X_train[:,0],np.where(y_train!=1))
x2false = np.take(X_train[:,1],np.where(y_train!=1))
plt.plot(x1false, x2false, 'r^', x1true,x2true, 'go')
lr = LogisticRegression().fit(X_train,y_train)
plt.figtext(.6,.03, lr.score(X_test,y_test))
print(lr.score(X_test,y_test))
plt.show()
