import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
class Preprossecing:
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs
        # self.per_list = per_list
        # self.per_idx = per_idx
        # self.tmp_input = tmp_input
        # self.tmp_output = tmp_output
        
    def shufel(self, data):
        inputs = data[:, :9]
        outputs = data[:, 9]
        per_list = np.random.permutation(len(data))
        inputs_sh = []
        outputs_sh = []
        for i in range(len(data)):
            per_idx = per_list[i]
            tmp_input = inputs[per_idx]
            tmp_output = outputs[per_idx]
            inputs_sh.append(tmp_input)
            outputs_sh.append(tmp_output)
        inputs_sh = np.array(inputs_sh)
        outputs_sh = np.array(outputs_sh)
        
        return inputs_sh, outputs_sh
    def norm_min_max(self, data):
        min_vec = inputs_sh.min(axis = 0)
        max_vec = inputs_sh.max(axis=0)
        inputs_sh = (inputs_sh - min_vec) / (max_vec - min_vec)

        return inputs_sh


        