import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
#from lstm import lstm_model

#https://github.com/zhuojimmy/tensorflow-lstm-regression

def x_sin(x):
    return x * np.sin(x)

def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)



#funcao geradora

def getPairfromCSV(arquivo):
    data = pd.read_csv('./data/'+arquivo, sep=";")
    data.columns = ["ind","X", "Y"]
    data=pd.DataFrame(data,columns=['X','Y'])
    return (data,getPair(data['X'],data['Y']))

def setPairCtoCSV(X,Y,arquivo):
    data = X.join(Y,lsuffix='X',rsuffix='Y',sort=False)
    data.to_csv('data/'+arquivo, sep=';')
    return data

def setPairCtoCSV2(X,Y,arquivo):
    data = pd.DataFrame(X)
    data.columns = ['A0X', 'A1X']
    data = data.join(Y,sort=False)
    data.columns = ['A0X', 'A1X','A0Y', 'A1Y']
    data.to_csv('data/'+arquivo, sep=';')
               
def getPairfromCSV2(arquivo):
    data = pd.read_csv('./data/'+arquivo, sep=";")
    data.columns = ['ind','A0X', 'A1X','A0Y', 'A1Y']
    dataX=pd.DataFrame(data,columns=['A0X', 'A1X'])
    dataY=pd.DataFrame(data,columns=['A0Y', 'A1Y'])
    #return (data['A0trainX'], data['A1trainX'],data['A0trainY'], data['A1trainY'])
    return (dataX,dataY)

def getPair(X,Y):
    #formatar dados de acordo com timesteps
    return(rnn_data(X,TIMESTEPS),
           rnn_data(Y,TIMESTEPS,labels=True))

