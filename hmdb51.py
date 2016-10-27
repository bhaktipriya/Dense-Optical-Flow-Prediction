import pickle
import numpy as np
import sys
from sklearn.cross_validation import train_test_split

def load_data():

    #load data
    fr = open('frames.data','rb')
    fl = open('flow.data','rb')
    reload(sys)  
    sys.setdefaultencoding('utf8')
    #X is the list of all frames
    X=[]
    while 1:
        try:
            X.append(pickle.load(fr))
        except(EOFError):
            break
    #Y is the list of all flows
    Y=[]
    while 1:
        try:
            Y.append(pickle.load(fl))
        except(EOFError):
            break

    X=np.array(X)
    Y=np.array(Y)		
    #Take 80% of the total data as train data
    #Take the rest as test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return (X_train, y_train), (X_test, y_test)
    
