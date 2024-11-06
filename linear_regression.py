#%% Linear regression shows the linear relationship between 
# the independent (predictor) variable and the dependent output
# variable.

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
SEED = 1

class LinearRegressionModel:
    
    def __init__(self, lr=0.01, iterations=100):
        print("Creating Model")
        self.learning_rate = lr
        self.iterations = iterations
        
    def predict(self, x):
        return x.dot(self.w) + self.b
    
    def fit(self,x, y):
        training_samples_count, n_features = x.shape
        
        #init
        self.w = np.random.rand(n_features)
        self.b = 0.1
        
        #loop
        for i in range(self.iterations):
            preds = self.predict(x)
            err = np.sum(y - preds)
            
            #gradient
            dw = -(2*(x.T).dot(err) )/training_samples_count
            db = -2*np.sum(err)/training_samples_count
            print(dw, db)
            
            #update weights
            self.w -= self.learning_rate*dw
            self.b -= self.learning_rate*db 
            
        return self.w, self.b

X,y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=SEED)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

m = LinearRegressionModel()
w, b = m.fit(X_train,y_train)
print(w,b)
y_pred = m.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f"% r2_score(y_test, y_pred))

#%% sklearn comparison

m = LinearRegression()
m.fit(X_train,y_train)
y_pred = m.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f"% r2_score(y_test, y_pred))