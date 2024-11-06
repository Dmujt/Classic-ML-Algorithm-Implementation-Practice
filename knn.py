#
# File runs k nearest neighbors algo
#

#%%
import scipy.spatial
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy
SEED = 1
N_NEIGHBORS = 3

class KNNClassifier():
    def __init__(self, n):
        self.n = n #n neighbors
        self.x = None
        self.y = None
        
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.n_classes = np.unique(self.y).shape[0]
        print("Class count: ", self.n_classes)
    
    def predict(self, x_test):
        distances = self.__distance_between_points(x_test, self.x)
        nearest_neighbor_classes = self.__select_nearest_k(distances)
        voted = self.__vote(nearest_neighbor_classes)
        return voted
    
    def __select_nearest_k(self,distances):
        # np.argpartition gets the k smallest
        k_closest = [] #return indexes of k closest points
        for d in distances:
            closest = np.argpartition(d, self.n)
            k_closest.append(closest[:self.n])
        return self.y[k_closest]
    
    def __distance_between_points(self, a, b):
        return scipy.spatial.distance.cdist(a, b)
    
    def __vote(self, predictions):
        preds = []
        for p in predictions:
            classVote = [0]*self.n_classes
            for y in p:
                classVote[y] += 1
            preds.append(np.argmax(classVote))
            
        return np.array(preds)
    
        
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=SEED)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

my_model = KNNClassifier(N_NEIGHBORS)
my_model.fit(X_train,y_train)
predictions = my_model.predict(X_test)
cm = confusion_matrix(y_test, predictions, normalize='true')
print(cm)

#%% create this model to test out our results
m = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
m.fit(X_train,y_train)
predictions = m.predict(X_test)
cm = confusion_matrix(y_test, predictions, normalize='true')
print(cm)


