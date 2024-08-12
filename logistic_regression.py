#%% Logistic Regression Model


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import math

SEED = 5

class LogisticRegression:
    
    def __init__(self, n_features, n_classes):
        self.weights = np.ones((n_features, n_classes))
        self.biases = np.zeros((1, n_classes))
        return
    
    def __sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))

    def train(self, x, y):
        #start, compute the dot product and add bias term
        z = np.dot(x, self.weights)
        print(z)
        b = np.tile(self.biases, (x.shape[0], 1))
        print(b)
        z = np.add(z, b)
        print()
        print(z)
        preds = self.__sigmoid(z)
        print(preds)
        #now compute error and new weight terms
        
        return preds
    
    def loss(self, sigmoid_preds, y):
        p1 = np.dot()
    def predict(self, x, y):
        return
    

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=SEED)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

m = LogisticRegression(4, 3)
m.train(X_train,y_train)

predictions = m.predict(X_test, y_test)

#print out classification results table 
cm = confusion_matrix(y_test, predictions)
print(cm)
