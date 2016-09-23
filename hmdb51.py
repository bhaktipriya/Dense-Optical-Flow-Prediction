import pickle
import numpy as np


def load_data():

    #load data
    fr = open('frames.data','rb')
    fl = open('flow.data','rb')
    
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

    #Take 80% of the total data as train data
    #Take the rest as test data
    
    #For Frames
    X_train = X[ :int(len(X)*0.8)]
    X_train = np.array(X_train)
    X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1], X_train.shape[2])
    X_train = X_train.astype('float32')
    X_train /= 255

    X_test = X[int(len(X)*0.8):]
    X_test = np.array(X_test)
    X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1], X_test.shape[2])
    X_test = X_test.astype('float32')
    X_test /= 255
    
    #For Flows
    y_train = Y[ :int(len(Y)*0.8) ]
    y_train = np.array(y_train)
    y_train = y_train - np.amin(y_train)
    y_train = y_train/ (np.amax(y_train) - np.amin(y_train) );
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[2]))

    y_test = Y[int(len(Y)*0.8):]
    y_test = np.array(y_test)
    y_test = y_test - np.amin(y_test)
    y_test = y_test/(np.amax(y_test) - np.amin(y_test) );
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[2]))
   
    return (X_train, y_train), (X_test, y_test)
    
