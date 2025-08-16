import numpy as np
import pandas as pd
import math



def label_convertion(label_matrix): # Converting string labels into binary format
    for i in range(len(label_matrix)):
        if label_matrix[i]=="rain": label_matrix[i] = 1
        else: label_matrix[i] = 0
    return label_matrix

y_train = label_convertion(y_train)

y_train = y_train.astype(float)
x_train = x_train.astype(float) 
weights = weights.astype(float)


def sigmoid(X,w,b):
    z = np.dot(X, w) + b
    s_val = 1/(1+np.exp(-z))
    return s_val


for x in range(iteration): # Training Loop
    difference = sigmoid(x_train, weights,bias) - y_train
    dw = np.dot(x_train.T, difference) * (1/x_train.shape[0])
    db = np.sum(difference) * (1/x_train.shape[0])

    weights = weights - lr * dw
    bias = bias - lr * db

def prediction(s_val): # Predicting sigmoid outputs for 0.5 threshold
    return 1 if s_val>0.5 else 0

class LogisticRegression:
    def __init__(self, iteration=1000, bias=1, lr=0.01):
        self.iteration = iteration
        self.bias = bias
        self.lr = lr
    
    def loadData(self):
        df = pd.read_csv('weather_forecast_data.csv') # Load Dataset
        df = df.to_numpy()

        self.x_train, self.y_train = df[:,:4], df[:,5] # Splitting training features and training labels
        self.weights = np.zeros(x_train.shape[1]) # Creating weight vector

        for i in range(len(self.y_train)):
            if self.y_train[i]=="rain": self.y_train[i] = 1
            else: self.y_traini] = 0
    def