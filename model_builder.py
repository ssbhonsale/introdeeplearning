import numpy as np
import torch 
from torch import nn
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def Normalize(X_train,X_test):
    se = StandardScaler()
    X_train = se.fit_transform(X_train)
    X_test = se.transform(X_test)
    return X_train, X_test

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

class NeuralNetModel(nn.Module):
    def __init__(self,input_dim,output_dim,n_layer,layer_size,activation) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.layer_size = layer_size
        self.activation = activation
        match self.activation:
            case "relu":
                self.act = nn.ReLU()
            case "sigmoid":
                self.act = nn.Sigmoid()
            case "tanh":
                self.act = nn.Tanh() 
        layers = []
        for i in range(self.n_layer):
            in_size = self.input_dim if i == 0 else self.layer_size
            out_size = self.output_dim if i == n_layer-1 else self.layer_size
            layers.append(nn.Linear(in_features=in_size,out_features=out_size))
            layers.append(self.act)
        self.model = nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x)
    

