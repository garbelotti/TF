from lstm import lstm_model as lstm
import numpy as np
import pandas as pd
import tensorflow as tf
import rnnInput as inp
import bioinfoInput as bioinp
import time as t
from tensorflow.contrib import learn

def iniciaRede(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS,LOG_DIR):
#inicializacao do regressor com os dados da rede
    with tf.device('/gpu:0'):
        regressor = learn.Estimator(model_fn=lstm(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), model_dir=LOG_DIR)
    return regressor

def treinaRede(valid_x,valid_y,train_x,train_y,regressor,passos=1000,TIMESTEPS=10,BATCH_SIZE=100):
    validaY=inp.rnn_data(valid_y,TIMESTEPS,labels=True)
    validaX=inp.rnn_data(valid_x,TIMESTEPS)
    #o monitor de validacao que é responsavel por "reforcar" o aprendizado, ele usa o data_val para
    validation_monitor = learn.monitors.ValidationMonitor(validaX, validaY, every_n_steps=1000, early_stopping_rounds=1000)
    orig_trainX = inp.rnn_data(train_x,TIMESTEPS)
    orig_trainY = inp.rnn_data(train_y,TIMESTEPS,labels=True)

    tempoInicial = t.time()
    tempot = tempoInicial
    with tf.device('/gpu:0'):
        regressor.fit(orig_trainX, orig_trainY, monitors=[validation_monitor],            batch_size=BATCH_SIZE, steps=passos)
    tempoTotal = t.time() - tempoInicial
    print("tempo total do treinamento: ",tempoTotal)
