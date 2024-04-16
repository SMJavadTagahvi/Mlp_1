import numpy as np
import pandas as pd
import Normalliz as nrm
class Split_data:
    def __init__(self, inputs, outputs, per_list, per_idx, tmp_input, tmp_output, X_train, Y_train, X_val, Y_val) -> None:
        nrm.__init__(inputs, outputs, per_list, per_idx, tmp_input, tmp_output)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        
    def spliting_data(self, inputs_sh, outputs_sh):
        trn_test_split = int(0.75 * len(inputs_sh))
        X_train = inputs_sh[0:trn_test_split , : ]
        Y_train = outputs_sh[0 : trn_test_split]

        X_val = inputs_sh[trn_test_split : , :] 
        Y_val = outputs_sh[trn_test_split : ,]