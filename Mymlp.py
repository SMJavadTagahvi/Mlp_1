import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import Preprossecing as psc
import Split_data as sd
class Mymlp:
    def __init__(self, x, y, input_net, out) -> None:
        self.x = x
        self.y = y
        self.input_net = input_net
        self.out = out
    
    def neural_wighet(self):
        n0 = 9 # input layer
        n1 = 8 # first hidden layer
        n2 = 4 # second hidden layer
        n3 = 1 # output layer

        w1 = np.random.uniform(low=-10,high= +10,size=(n1,n0))
        w2 = np.random.uniform(low=-10,high= +10,size=(n2,n1))
        w3 = np.random.uniform(low=-10,high= +10,size=(n3,n2))     
    
        return n0, n1, n2, n3, w1, w2, w3
        
    def activation(self, x):
        y = 1/(1 + np.exp(-1 * x))
        return y
    
    def feedforward(self, input_net, w1, w2, w3, activation):
        x1 = np.dot(input_net , w1.T)
        y1 = activation(x1)
        x2 = np.dot(y1 , w2.T)
        y2 = activation(x2)
        x3 = np.dot(y2 , w3.T)
        y3 = activation(x3)

        return y1 , y2 , y3
    
    def d_activation(out):
        d_y = out * ( 1 - out)
        return d_y