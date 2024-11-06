#
# Implements SVM
#

#%%
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
SEED = 1

class SVMClassifier:
    def __init__(self, lr=0.001, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.lam = 0.01
        self.w = None
        self.b = None
        #find hyperplane defined by wx - b = 0
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        y_labels = np.where(y <=0, 0, 1)
        #condition and if x else y where np.where(con, x, y)
        
        self.w = np.zeros(n_features) #weights per feature
        self.b = 0
        
        for i in range(self.iterations):
            for idx, x_i in enumerate(x): 
                #loop each sample in dataset
                fn = np.dot(x_i, self.w - self.b)
                if y_labels[idx] * fn >= 1:
                    self.w -= self.lr * (2.0*self.lam*self.w)
                else:
                    self.w -= self.lr * (2.0*self.lam*self.w - np.dot(x_i, y_labels[idx]))
                    self.b -= self.lr*y_labels[idx]
            
    def predict(self, x):
        approx = np.dot(x, self.w) - self.b
        print(approx)
        return np.sign(approx)
    
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=SEED)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

m = SVMClassifier(iterations=1000)
m.fit(X_train,y_train)
predictions = m.predict(X_test)
cm = confusion_matrix(y_test, predictions, normalize='true')
print(cm)

#%% Verification with SKlearn implementation

m = svm.SVC()
m.fit(X_train,y_train)
predictions = m.predict(X_test)
cm = confusion_matrix(y_test, predictions, normalize='true')
print(cm)
