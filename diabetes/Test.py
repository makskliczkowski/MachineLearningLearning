# from tensorflow import keras
from keras.models import Sequential
from numpy import loadtxt
from keras.layers import Dense
import pandas as pd
import numpy as np
from numpy import loadtxt
from keras.models import load_model
from keras import backend as K



def createModel(epo : int, batch : int):
    dt = loadtxt("pima-indians-diabetes.data.csv", delimiter=',')
    # split into input (X) and output (y) variables
    X = np.array(dt[:, 0:8])
    y = np.array(dt[:, 8])

    dtPD = pd.read_csv("pima-indians-diabetes.data.csv", header=None, index_col=None)

    # print(dtPD.iloc[1,:])
    sizeTeach = int(4*len(dtPD.index)/5)

    # CREATE MODEL
    model = Sequential()
    model.add(Dense(16, input_dim=8, activation='relu'))  # DENSE TYPE OF LAYER - EACH NEURON IS CONNECTED
    # TO EACH IN NEXT
    # SUCH ARCHITECTURE GIVES THE HIGHEST CHANCE OF TEACHING WELL BUT ALSO TO OVERDUE THINGS
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation='relu'))
    # we compile  model now
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=epo, batch_size=batch)
    # epochs are how many times the waves will be recalculated
    # batch size are how many imput sets will be used in each epoch
    #
    # save model and architecture to single file
    model.save("model.h5")
    print("Saved model to disk")

def testModel():
    # load model
    model = load_model('model.h5')
    # summarize model.
    model.summary()
    # load dataset
    dtPD = pd.read_csv("pima-indians-diabetes.data.csv", header=None, index_col=None)
    # print(dtPD.iloc[1,:])
    sizeTeach = int(1*len(dtPD.index)/5)
    # evaluate the model
    score = model.evaluate(dtPD.iloc[sizeTeach:-1, 0:8], dtPD.iloc[sizeTeach:-1, 8])
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    #_, accuracy = model.evaluate(X, y)
    test = np.array(dtPD.iloc[sizeTeach:-1,0:8])
    #print(test)
    # remeber that it always needs to know how many batch
    predictions = model.predict_classes(test)

    print(predictions,dtPD.iloc[:,8])
#createModel(150, 10)
testModel()