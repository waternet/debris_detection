import pickle
import data
import math
import random

def main():
    X, y = data.getData(as_grayscale=True, balance_ones=True)
    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))

    lr_model = pickle.load(open('1stmodel.pickle', 'rb'))

    print "Model score: ", lr_model.score(X,y)

    y_pred = lr_model.predict(X)
    #om de afbeelding weer te geven moeten we de matrix weer 200x200 maken!
    size = int(math.sqrt(X.shape[1]))
    X = X.reshape((X.shape[0],size,size))
    for i in range(0, 10):
        index = random.randrange(0, X.shape[0])
        data.showImage(X, y_pred, index)

if __name__=="__main__":
    main()
    
