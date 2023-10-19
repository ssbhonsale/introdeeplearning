import numpy as np
import torch 
from torch import nn
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def loadData():
    df = pd.read_csv("winequality-red.csv")
    X = df.iloc[:,:11]
    y = df["quality"]
    return X,y

def splitData(X,y,ratio):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = ratio,random_state=5)
    return X_train, X_test, y_train, y_test


def Normalize(X_train,X_test):
    se = StandardScaler()
    X_train = se.fit_transform(X_train)
    X_test = se.transform(X_test)
    return X_train, X_test