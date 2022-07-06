from keras.callbacks import History
import time
import datetime
from tensorflow.keras import initializers
import keras
import pylab
from tensorflow import keras
from keras.layers import Input, Conv2D, Dense, concatenate
import random
import numpy as np
from keras import Sequential
from collections import deque
from tensorflow.keras.layers import Bidirectional, LSTM
from keras.layers import Dense, LayerNormalization, Permute, LeakyReLU, DepthwiseConv2D, Embedding,  Lambda,  Add, Average, TimeDistributed, Conv1D, Conv2D, Subtract, Activation, LocallyConnected2D, LocallyConnected1D, Reshape, concatenate, Concatenate, Flatten, Input, Dropout, MaxPooling1D,  MaxPooling2D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM, Input, concatenate
from keras.optimizers import Adagrad, RMSprop
from keras.metrics import Mean
from keras import backend as K
import pathlib
import tensorflow as tf
import pandas as pd
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as FF
from icecream import ic
from keras.layers.advanced_activations import PReLU, LeakyReLU
 

def build_simple_model(action_space=2,state_space=1):
    
    X_input = Input(shape=(state_space,))
    X = Dense(1, activation="relu")(X_input)
    X1 = Dense(action_space, activation="relu")(X) 
    model = Model(inputs=X_input, outputs=X1, name='build_model1') 
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00025,epsilon=0.01), metrics=["accuracy"])
    return model, model.get_weights() 
if __name__ == '__main__':
    build_simple_model()
