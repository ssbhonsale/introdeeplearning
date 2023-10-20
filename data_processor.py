import numpy as np
import torch 
from torch import nn
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader, Dataset
def loadData():
    df = pd.read_csv("winequality-red.csv")
    X = df.iloc[:,:11]
    y = df["quality"]
    return X,y

def splitData(X,y,ratio):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = ratio)
    return X_train, X_test, y_train, y_test


def Normalize(X_train,X_test,Type=None):
    match Type:
        case "standard":
           se = StandardScaler()
        case "minmax":
            se = MinMaxScaler()
    if Type is not None:
        X_train = se.fit_transform(X_train)
        X_test = se.transform(X_test)
    return X_train, X_test

class WineData(Dataset):
    def __init__(self,X,y):
        self.x = torch.from_numpy(X).type(torch.float)
        self.y = torch.from_numpy(y.to_numpy()).type(torch.LongTensor)
        self.n_sample = self.x.shape[0]
        self.n_features = self.x.shape[1]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_sample
    

def generateDataset(ratio,Type,droplist):
    X,y = loadData()
    X = X.drop(droplist,axis=1)
    X = X.to_numpy()
    X_train, X_test, y_train, y_test = splitData(X,y,ratio)
    print(type(X_train))
    print(X.shape)
    X_train, X_test = Normalize(X_train, X_test, Type)
    print
    trainingData = WineData(X_train,y_train)
    testData = WineData(X_test,y_test)
    return trainingData, testData