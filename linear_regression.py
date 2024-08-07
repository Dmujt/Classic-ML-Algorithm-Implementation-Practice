#%% Linear regression shows the linear relationship between 
# the independent (predictor) variable and the dependent output
# variable.


class LinearRegressionModel:
    
    def __init__(self):
        print("Creating Model")
        
    def predict(self, x):
        return
    
    def train(self,x, y):
        return
    

xTrain = []
yTrain = []

m = LinearRegressionModel()

m.train(xTrain,yTrain)

from sklearn.datasets import load_iris
data = load_iris()
data.target[[10, 25, 50]]
list(data.target_names)