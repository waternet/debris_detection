# DATA.PY
#
# Rob van Putten - Waternet - GPLv3 - 2017
#
# Dit script genereert de drijfvuildata voor de toepassing in machine learning algoritmen.
# Op basis van vastgestelde mappen zijn deze functies in staat om afbeeldingen in te lezen,
# te converteren naar matrici, classes te balanceren en andere functionaliteit.

import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import misc
from skimage import color


PATH_TO_DEBRIS_IMAGES = "data/labeled/debris/"
PATH_TO_NODEBRIS_IMAGES = "data/labeled/no_debris/"

def getData(as_grayscale=False, balance_ones=True):
    """Haal de data op uit de mappen waar de afbeeldingen staan en maak een matrix X (features) 
en y (labels). 

Input:  as_grayscale (default False), converteert de afbeeldingen naar grijswaarden
        balance_ones (default True), zorgt ervoor dat het aantal 1 labels ongeveer gelijk 
        ligt met het aantal 0 labels 

Output: X, features matrix  y, label matrix

TODO: De balancering van de classes vindt plaats door simpelweg dezelfde fotos te kopieren.
Dit kan later verbeterd worden door andere technieken (bv rotatie of shifting) toe te passen."""

    X = []
    y = []

    #zoek de bestandsnamen in de opgegeven mappen
    img_debris = glob.glob(PATH_TO_DEBRIS_IMAGES + '*.png')
    img_no_debris = glob.glob(PATH_TO_NODEBRIS_IMAGES + '*.png')

    #maak de originele matrix van de 1 labels (drijfvuil)
    Xdebris = []
    for img in img_debris:
        Ximg = misc.imread(img)
        if as_grayscale: #converteer naar grijsschaal (let op, meteen waarden tussen 0.0 en 1.0)
            Ximg = color.rgb2gray(Ximg)
        Xdebris.append(Ximg)
        
    #maak de originele matrix van de 0 labels (geen drijfvuil)
    Xnodebris = []
    for img in img_no_debris:
        Ximg = misc.imread(img)
        if as_grayscale:
            Ximg = color.rgb2gray(Ximg)
        Xnodebris.append(Ximg)

    #indien we de classes willen balanceren maken we een kopie van de de 1 classes
    if balance_ones:
        factor = round(len(img_no_debris) / len(img_debris))
        Xdebris = np.repeat(Xdebris, factor, axis=0)

    #maak de uiteindelijke matrix
    X = np.vstack([Xdebris, Xnodebris])
    y = np.array([1]*Xdebris.shape[0] + [0]*len(Xnodebris))

    return X, y

def showImage(X, y, i):
    Ximg = X[i]
    label = y[i]
    fig = plt.figure()
    if len(Ximg.shape) == 2:
        plt.imshow(Ximg, cmap='gray')
    elif len(Ximg.shape) == 3:
        plt.imshow(Ximg)
    else:
        print "Onbekende matrix vorm!", Ximg.shape

    if label == 0:
        fig.suptitle("foto %d, label=geen drijfvuil" % i)
    else:
        fig.suptitle("foto %d, label=drijfvuil" % i)
        
    plt.show()

def showRandomImage(X, y):
    showImage(X,y,random.randrange(0, X.shape[0]))

if __name__=="__main__":
    X, y = getData(as_grayscale=True, balance_ones=True)
    print X.shape
    for i in range(10):
        showRandomImage(X, y)
