import numpy as np
import utils
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle

class LogisticModel(object):
    def __init__(self):
        self.W = None
        self.b = 0.0
    
    def fit(self, X, Y, learning_rate = 10e-7, reg=0, epochs=120000, show_fig=False):
        X, Y = shuffle(X,Y)

        #split train validation
        Xvalid, Yvalid = X[-40:], Y[-40:]
        X,Y = X[:-40], Y[:-40]
        N,D = X.shape
        self.W = np.random.randn(D) / np.sqrt(D)
        self.b = 0
        
        costs = []
        best_validation_error = 1.
        
        for i in xrange(epochs):
            pY = self.forward(X)
            #gradient descent step
            self.W -= learning_rate * X.T.dot(pY-Y) + reg*self.W
            self.b -= learning_rate*((pY-Y).sum() + reg*self.b)
            
            if i % int(epochs / 10) == 0:
                pYvalid = self.forward(Xvalid)
                c = utils.sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = utils.error_rate(Yvalid, np.round(pYvalid))
                print "i:", i, "costs:", c, "error:", e
                if e < best_validation_error:
                    best_validation_error = e

        print "Best validation error = ", best_validation_error
        if show_fig:
            plt.plot(costs)
            plt.show()
   
    def forward(self, X):
        return utils.sigmoid(X.dot(self.W) + self.b)
    
    def predict(self, X):
        pY = self.forward(X)
        return np.round(pY)
    
    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - utils.error_rate(Y, prediction)

    
#    def random_test(self, num_tests=10):
#        for i in range(num_tests):
#            index = random.randrange(0, X.shape[0])
#            print index
                    
                    
        
