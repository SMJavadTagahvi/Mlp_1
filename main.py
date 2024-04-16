import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import Preprossecing as psc
import Split_data as sd
import Mymlp as mlp

epochs = 500
lr = 0.01

List_AccTrain = []
List_MseTrain = []
List_AccValid = []
List_MseValid = []
for i in range(epochs):
    for j in range(len(X_train)):
        input = mlp.X_train[j] 
        input = np.reshape(input , newshape=(1,n0)) 
        target = mlp.Y_train[j]

        y1 , y2 , y3 = mlp.feedforward(input)
        error = target - y3


        d_f3 = mlp.d_activation(y3)

        d_f2 = mlp.d_activation(y2)
        diag_d_f2 = np.diagflat(d_f2)

        d_f1 = mlp.d_activation(y1)
        diag_d_f1 = np.diagflat(d_f1)

        temp1 = -2 * error * d_f3
        temp2 = np.dot(temp1 , w3)
        temp3 = np.dot(temp2 , diag_d_f2)
        temp4 = np.dot(temp3 , w2)
        temp5 = np.dot(temp4 , diag_d_f1)
        temp5 = temp5.T
        temp6 = np.dot(temp5 , input)

        w1 = w1 - lr * temp6

        w2 = w2 - lr * np.dot(temp3.T , y1)

        w3 = w3 - lr * np.dot(temp1.T , y2)

#calculating MSE and accuracy for Train

    Netoutput_Train = []
    target_Train = []
    rnd_Netoutput_Train = []
    for idx in range(len(mlp.X_train)):
      input = mlp.X_train[idx]
      target = mlp.Y_train[idx]
      target_Train.append(target)

      _ , _ , pred = mlp.feedforward(input)
      Netoutput_Train.append(pred)
      rnd_Netoutput_Train.append(np.round(pred))

    mse_Train = mse(target_Train , Netoutput_Train)
    List_MseTrain.append(mse_Train)
    acc_Train = acc(target_Train , rnd_Netoutput_Train)
    List_AccTrain.append(acc_Train)
    print('epoch ' , i , ' : MSE_Train = '  , mse_Train, '\tAcc_val = ' , acc_Train)
    
#calculating MSE and accuracy for validatin

    Netoutput_val = []
    target_val = []
    rnd_Netoutput_val = []
    for idx in range(len(mlp.X_val)):
      input = mlp.X_val[idx]
      target = mlp.Y_val[idx]
      target_val.append(target)

      _ , _ , pred = mlp.feedforward(input)
      Netoutput_val.append(pred)
      rnd_Netoutput_val.append(np.round(pred))

    mse_val = mse(target_val , Netoutput_val)
    List_MseValid.append(mse_val)
    acc_val = acc(target_val , rnd_Netoutput_val)
    List_AccValid.append(acc_val)
    print('epoch ' , i , ' : MSE_val = '  , mse_val, '\tAcc_val = ' , acc_val)
    print('---------------------------------------------------------------------')