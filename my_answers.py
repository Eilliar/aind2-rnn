import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    X = np.array([series[i : i + window_size] for i in range(0, len(series) - (window_size))])
    y = np.array(series[window_size : ])
    y = np.reshape(y, (len(y),1))
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units = 5, input_shape = (window_size, 1), activation='tanh'))
    model.add(Dense(1, activation='tanh') )
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import re
    # punctuation = ['!', ',', '.', ':', ';', '?']
    text = re.sub('[^a-z!,.:;?]+', ' ', text)
    text = text.replace('  ',' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # This implementation below is failing in the udacity submit test... still don't know why... since it's producing the same output that tha uncommented one
    #inputs = np.array([text[i : i + window_size] for i in range(0, len(text) - (window_size), step_size)])
    #outputs = np.array([text[i] for i in range(window_size, len(text), step_size)])
    
    # 2nd Try:
    last_element = window_size
    while(last_element < len(text)):
        inputs.append(text[last_element - window_size:last_element])
        outputs.append(text[last_element])
        last_element += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    # layer 1 should be an LSTM module with 200 hidden units
    model.add(LSTM(units = 200, input_shape = (window_size, num_chars), activation = 'tanh'))
    # layer 2 should be a linear module, fully connected
    model.add(Dense(units = num_chars, activation = 'linear'))
    # layer 3 should be a softmax activation
    model.add(Activation('softmax'))

    return model