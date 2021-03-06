# -*- coding: utf-8 -*-
"""Assignment_MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iV8Q7DwwLhE7vufcC8JR3DLwBUTNniO8
"""

# Commented out IPython magic to ensure Python compatibility.
# %pylab
# %matplotlib inline
import numpy as np 
import pandas as pd
import scipy 
import sklearn

from sklearn.datasets import load_digits
dig = load_digits()

dig.images.shape, dig.data.shape, dig.target.shape, dig.target_names

from scipy.special import expit
expit(-1000)

plt.imshow(dig.images[2], cmap=plt.cm.gray_r)

from sklearn.model_selection import train_test_split
dig_data = dig.data / 16
X_train, X_test, y_train, y_test = train_test_split(dig_data, dig.target, test_size=0.1, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# encode the data using onehotencoder

from sklearn.preprocessing import OneHotEncoder
encoding = OneHotEncoder()
y_train_encod = encoding.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_encod = encoding.fit_transform(y_test.reshape(-1,1)).toarray()
#y_train_encod[:3], y_train[:3]
#y_test_encod[:3],y_test[:3]

class Network(object):

     def __init__(self):
        self.w1 = None 
        self.w2 = None 
        self.b1 = None 
        self.b2 = None  
        self.dw1 = 0
        self.dw2 = 0
        self.db1 = 0 
        self.db2 = 0
        self.x = None       
        self.a1 = None
        self.a2 =None
        
     def set_parameters(self, weights, bias):
       
        self.w1, self.w2 = weights[0], weights[1]
        self.b1, self.b2 = bias[0], bias[1]

     def forward_propagation(self, x):
        
        self.x = x
        self.a1 = np.dot(self.x, self.w1) + self.b1
        self.a2 = self.sigmoid(self.a1)
        a3 = np.dot(self.a2, self.w2) + self.b2
        return self.softmax(a3)

     def backward_propagation(self, err1, l_r=0.1):
        
        self.dw2 = l_r * np.dot(self.a2.T, err1) / self.x.shape[0]
        self.db2 = l_r * np.average(err1, axis=0)
        err2 = np.dot(err1, self.w2.T)
        err2 *= self.a2*(1 - self.a2)
        self.dw1 = l_r * np.dot(self.x.T, err2) / self.x.shape[0]
        self.db1 = l_r * np.average(err2, axis=0)

     def update_parameters(self, l2=0):
        
        w1 = self.w1 + self.dw1 - l2 * self.w1
        w2 = self.w2 + self.dw2 - l2 * self.w2
        b1 = self.b1 - l2 * self.db1
        b2 = self.b2 - l2 * self.db2
        self.set_parameters([w1, w2], [b1, b2])

     def model(self, X, y, epochs=10, l_r=0.1, convert=False, l2=0):
        
        cost = []
        for i in range(epochs):
            y_hat = self.forward_propagation(X)
            error = y - y_hat
            self.backward_propagation(error, l_r)
            self.update_parameters(l2/y.shape[0])
            if convert:
                y_hat = np.clip(y_hat, 0.00001, 0.99999)
                cost.append(-np.sum(y * np.log(y_hat))/y.shape[0])
            #print ("Cost on training data: {}".format(cost))
        return cost
                       
          
     def predict(self, X):
        y_hat = self.forward_propagation(X)
        return np.argmax(y_hat, axis=1)
    
     def sigmoid(self, y_hat):
        return expit(y_hat)
    
     def softmax(self, y_hat):
        tmp = y_hat - y_hat.max(axis=1).reshape(-1, 1)
        exp_tmp = np.exp(tmp)
        return exp_tmp / exp_tmp.sum(axis=1).reshape(-1, 1)

n4 = Network()

b = np.sqrt(1./64)
w1 = np.random.uniform(-b, b, (64, 100))
b1 = np.random.uniform(-b, b, 100)
b = np.sqrt(1./100)
w2 = np.random.uniform(-b, b, (100, 10))
b2 = np.random.uniform(-b, b, 10)
n4.set_parameters([w1, w2], [b1, b2])

train_costs = n4.model(X_train, y_train_enc, 2000, 0.1, convert=True)

predicted_test = n4.predict(X_test)

predicted_train = n4.predict(X_train)

# calculating accuracy of test data

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_test)
print(accuracy, "accuracy of test data")

# cost of test data
test_costs = n4.model(X_test, y_test_enc, 2000, 0.1, convert=True)

# calculate accuracy of train data

accuracy1 = accuracy_score(y_train, predicted_train)
print(accuracy1, "accuracy of train data")

#caluculate the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_test)
print(cm , "value of confusion matrix")

#calculating Sensitivity, Specificity of data

FP = cm.sum(axis=0)-np.diag(cm)
FN = cm.sum(axis=1)-np.diag(cm)
TP = np.diag(cm)
TN = cm.sum()-(FP+FN+TP)

print('False Positives\n {}'.format(FP))
print('False Negetives\n {}'.format(FN))
print('True Positives\n {}'.format(TP))
print('True Negetives\n {}'.format(TN))

TPR = TP/(TP+FN)
print('Sensitivity \n {}'.format(TPR))
TNR = TN/(TN+FP)
print('Specificity \n {}'.format(TNR))

#calculate precision and recall

Precision = TP/(TP+FP)
print('Precision \n {}'.format(Precision))

Recall = TP/(TP+FN)
print('Recall \n {}'.format(Recall))

#calculate the accuracy of a model
Acc = (TP+TN)/(TP+TN+FP+FN)
print('??ccuracy \n{}'.format(Acc))

# Calculate F1-score 
from sklearn.metrics import cohen_kappa_score
F1_score = 2*(Precision*Recall)/(Precision+Recall)
print('F1_Score \n{}'.format(F1_score))

plt.plot(test_costs)

plt.plot(train_costs)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='sgd', learning_rate_init=0.1, alpha=0, batch_size=1617,
                    activation='logistic', random_state=10, max_iter=2000,
                    hidden_layer_sizes=100, momentum=0)

mlp.fit(X_train, y_train)

mlp.score(X_test, y_test)