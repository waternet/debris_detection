import data
import lr_model
import sys
import pickle
import numpy as np

def main():
    #we gaan gezien het aantal features in kleur eerst voor de oplossing met grayscale waarden
    X, y = data.getData(as_grayscale=True, balance_ones=True)

    #eerst even testen of reshape werkt zoals ik denk dat het werkt..
    #Xtest = np.arange(30).reshape((5,3,2))
    #print Xtest
    #Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1]*Xtest.shape[2]))
    #print Xtest
    #yep.. werkt!
    
    #we maken gebruik van het logistic model waarbij de feature matrix platgeslagen moet zijn
    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    
    #data.showRandomImage(X, y)
    model = lr_model.LogisticModel()
    model.fit(X, y, epochs=10000, show_fig=True)
    print "Model score: ", model.score(X,y)

    with open('1stmodel.pickle', 'wb') as fp:
        pickle.dump(model, fp)

if __name__=="__main__":
    main()
    
