os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
from random import random
from keras.models import Sequential  
from keras.layers.core import Dense, Activation  , Dropout
from keras.layers.recurrent import LSTM
import time

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


### Load the data                                                      ###
# Assuming we have a seperate training and test data set #

ts_train = []
ts_test = []
with open('train.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    try:
        for row in reader:
            ts_train.append(float(row[0]))
    except csv.Error as e:
        sys.exit('train.csv, line %d: %s' % (reader.line_num, e))
        
with open('test.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    try:
        for row in reader:
            ts_test.append(float(row[0]))
    except csv.Error as e:
        sys.exit('test.csv, line %d: %s' % (reader.line_num, e))

# Plot of the data
plt.plot(range(len(ts_train)), ts_train, 'ro');
plt.show()

# put it in a numpy array
ts_test = np.asarray(ts_test)
ts_train = np.asarray(ts_train)

# shifts the data in l by n steps (with replacement from the other end)
def shift(l, n):
    return np.concatenate([l[n:], l[:n]])

# Make data available in an infinite ring-like structure with a function
# that generates chunks of all possible sliding window positions
# to train the network with.
# Returns a chunk in a matrix together with time-delayed columns.
def prepare_train_set (data, past = 40, future = 1):
    data_mat = [0] * (past+future)
    
    for i in range(-past+1,0):
        data_mat[i+past-1] = shift(data,i)
    data_mat[past-1] = data
    for i in range(1,future+1):
        data_mat[i+past-1] = shift(data,i)
    
    df = pd.DataFrame(data=data_mat)
    columns = [df]
    
    return df.values.transpose()


### Initial deep learning model ###

batch_size = 72
epochs  = 50
y_len = 1
x_len = 50
neurons = [60, 1]

# Make data ready for input into model
train = prepare_train_set(ts_train, x_len, y_len)
X, y = train[:, 0:-y_len], train[:, -y_len:]

# Samples/TimeSteps/Features
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the model
model = Sequential()
model.add(LSTM(units=neurons[0], input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=neurons[1]))
model.compile(loss='mae', optimizer='adam')

# Fitting
h = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)
plt.plot(h.history['loss'], label='loss')
plt.legend()
plt.show()

# To predict one future time step of the data, using a sliding window (of length 50), 
# one hidden long-short-term-memory layer with 60 neurons was used, followed by a simple
# fully connected layer with one neuron.
# To predict more than one time step into the future, the predicted one will be shifted
# into the input time series before repeating the prediction.


### Evaluate the results ###

# Predict n time-steps from a window.
def predict_from_win(model, window, n):
    X = window.reshape(1, window.shape[0], 1)

    for i in range(n):
        p = model.predict(X)

        X = X[:,1:,:]
        X = np.concatenate((X, p.reshape(1,1,1)), axis=1)
        
    return X[:,-10:,:]

# Returns a window of with a given start point and length.
def get_window(data, start, length):
    len_data = len(data)
    end = start + length
    if end > len_data:
        end = end-len_data
        return np.concatenate((data[start:], data[:end]))
    return data[start:end]
    
# Test the model by predicting 10 timepoints at a time until
# the whole test data set is reproduced.
def test_model(model, ts_test, x_len):
    y_len = 10
    predicted = np.asarray([])

    # windows starting at end of sequence
    for i in np.arange(len(ts_test)-x_len, len(ts_test)-y_len+1, y_len):
        window = get_window(ts_test, i, x_len)
        y = predict_from_win(model,window,y_len)
        y = np.reshape(y,y.size)
        predicted = np.append(predicted,y)

    for i in np.arange(0, len(ts_test) - x_len, y_len):
        window = get_window(ts_test, i, x_len)
        y = predict_from_win(model,window,y_len)
        y = np.reshape(y,y.size)
        predicted = np.append(predicted,y)

    mse = ((ts_test - predicted) ** 2).mean(axis=0)
    print('MSE =', mse)

    plt.plot(range(len(predicted)),predicted, label='predicted')
    plt.plot(range(len(ts_test)), ts_test, label='truth')
    plt.legend()
    plt.show()
    
test_model(model, ts_test, x_len)

# The produced prediction is compared to the true test set and the quality of the prediction is
# measured by the mean squared error (MSE).


### Improvement through hyperparameter optimization ###
# WARNING
# Running this next one might take some time.

def shift(l, n):
    return np.concatenate([l[n:], l[:n]])

def get_data():
    x_len = 50
    y_len = 1
    
    ts_train = []
    with open('train.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ts_train.append(float(row[0]))
            
    ts_train = np.asarray(ts_train)

    past = 50
    future = 1
    data_mat = [0] * (past+future)
    
    for i in range(-past+1,0):
        data_mat[i+past-1] = np.concatenate([ts_train[i:], ts_train[:i]])
    data_mat[past-1] = ts_train
    for i in range(1,future+1):
        data_mat[i+past-1] = np.concatenate([ts_train[i:], ts_train[:i]])
    
    df = pd.DataFrame(data=data_mat)
    columns = [df]
    
    train = df.values.transpose()
    
    X, y = train[:, 0:-y_len], train[:, -y_len:]

    #Samples/TimeSteps/Features
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Seperate validation set
    val_set = np.random.randint(0, high=len(X), size=int(round(len(X)/10)), dtype='l')
    train_set = np.arange(X.shape[0])
    train_set = np.delete(train_set,val_set)
    
    x_train = X[train_set,:,:]
    y_train = y[train_set]
    x_test = X[val_set,:,:]
    y_test = y[val_set]

    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(LSTM(units={{choice([16,32,64,128,256])}}, 
                   input_shape=(x_train.shape[1], x_train.shape[2])))
    
    model.add(Dropout({{uniform(0, 1)}}))

    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(units={{choice([2,4,8,16,32,64,128,256])}}))
        model.add(Dropout({{uniform(0,1)}}))

    model.add(Dense(1))
   
    model.compile(loss='mae', # because mse gives me NaN's some times
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(x_train, y_train,
              batch_size={{choice([8,16,32,64, 128])}},
              epochs=50,
              verbose=0,
              shuffle=False,
              validation_data=(x_test, y_test))
    mae = model.evaluate(x_test, y_test, verbose=0)
    print('MAE:', mae)
    return {'loss': mae, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
        data=get_data,
        algo=tpe.suggest,
        max_evals=200,
        trials=Trials(),
        eval_space=True,
        #return_space=True,
        notebook_name='Deep Learning Test')
    X_train, Y_train, X_test, Y_test = get_data()
    print('Evalutation of best performing model:')
    print(best_model.evaluate(X_test, Y_test))
    print('Best performing model chosen hyper-parameters:')
    print(best_run)

# Predict n time-steps from a window.
def predict_from_win(model, window, n):
    X = window.reshape(1, window.shape[0], 1)

    for i in range(n):
        p = model.predict(X)

        X = X[:,1:,:]
        X = np.concatenate((X, p.reshape(1,1,1)), axis=1)
        
    return X[:,-10:,:]

# Returns a window of with a given start point and length.
def get_window(data, start, length):
    len_data = len(data)
    end = start + length
    if end > len_data:
        end = end-len_data
        return np.concatenate((data[start:], data[:end]))
    return data[start:end]
    
# Test the model by predicting 10 timepoints at a time until
# the whole test data set is reproduced.
def test_model(model, ts_test, x_len):
    y_len = 10
    predicted = np.asarray([])

    # windows starting at end of sequence
    for i in np.arange(len(ts_test)-x_len, len(ts_test)-y_len+1, y_len):
        window = get_window(ts_test, i, x_len)
        y = predict_from_win(model,window,y_len)
        y = np.reshape(y,y.size)
        predicted = np.append(predicted,y)

    for i in np.arange(0, len(ts_test) - x_len, y_len):
        window = get_window(ts_test, i, x_len)
        y = predict_from_win(model,window,y_len)
        y = np.reshape(y,y.size)
        predicted = np.append(predicted,y)

    mse = ((ts_test - predicted) ** 2).mean(axis=0)
    print('MSE =', mse)

    plt.plot(range(len(predicted)),predicted, label='predicted')
    plt.plot(range(len(ts_test)), ts_test, label='truth')
    plt.legend()
    plt.show()
    
# Load test data
ts_test = []
        
with open('test.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    try:
        for row in reader:
            ts_test.append(float(row[0]))
    except csv.Error as e:
        sys.exit('test.csv, line %d: %s' % (reader.line_num, e))
ts_test = np.asarray(ts_test)
        

# ...and finally test the model
test_model(best_model, ts_test, 50)

# To improve the model, the following hyperparameters are chosen using hyperopt's
# Tree-of-Parzen-Estimators algorithm for mean absolute error minimization:
# - Number of neurons in the LSTM layer
# - Number of Dense layers below (2 or 3)
# - Number of neurons in the optional layer
# - Dropout percentage in highest 2 layers
# - Optimizer used for compilation
# - Batch size for fitting
