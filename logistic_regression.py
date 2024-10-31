#%% Logistic Regression Model


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import math

SEED = 5

class LogisticRegressionFromScratch:
    
    def __init__(self, n_features, n_classes):
        self.n = n_classes
        self.weights = np.zeros((n_classes, n_features))
        self.biases = np.zeros((n_classes, 1))
        
        self.learning_rate = 0.001
        return
    

    def fit(self, x, y, iterations=10):
        #start, compute the dot product and add bias term
        for i in range(iterations):
            preds = self.__pred(x)
            
            #now compute error and new weight terms
            loss = self.__loss(self.__class_pred(preds), y)
            print("Loss: ", round(loss, 4))
            
            #learn per class weights
            for c in range(self.n):
                #compute the derviative of loss for each of these
                print(y.shape, preds[:, c].shape, x.shape)
                dW = -(preds[:, c] - y)
                print(x.T)
                dW = dW.T*x
                print(dW)
                print(self.weights[c].shape)
                self.weights[c] = self.weights[c] - (self.learning_rate*dW)
                self.biases[c] = self.biases[c] - (self.learning_rate*dW)
        return preds
    
    
    def __loss(self, preds, y):
        return -np.sum(y*np.log(preds))
        
    def __pred(self, x):
        preds = []
        for c in range(self.n): 
            wc = self.weights[c, :]
            bc = self.biases[c, :]
            z = np.dot(x, wc) + bc
            preds.append(np.exp(z))

        predsum = np.sum(np.array(preds))

        p = []
        for c in range(self.n):
            p.append(preds[c]/predsum)
            
        p = np.array(p)
        p = np.swapaxes(p,0,1)
        
        return p
    
    def __class_pred(self, preds):
        return np.argmax(preds, axis=1) 
    
    def predict(self, x):
        return self.__class_pred(x)
    

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=SEED)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


m = LogisticRegressionFromScratch(4, 3)
m.fit(X_train,y_train)

predictions = m.predict(X_test)

#print out classification results table 
cm = confusion_matrix(y_test, predictions)
print(cm)

#%%
# create this model to test out our results
m = LogisticRegression()
m.fit(X_train,y_train)

predictions = m.predict(X_test)
cm = confusion_matrix(y_test, predictions, normalize='true')
print(cm)

