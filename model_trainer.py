import numpy as np
import torch 
from torch import nn
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from model_builder import *
from data_processor import *
from torch.utils.data import DataLoader, Dataset

def run_training(args):
    trainingData, testData = generateDataset(args.RATIO,args.NORM_TYPE)
    loadTrainingData = DataLoader(dataset=trainingData,batch_size=args.BATCH_SIZE, shuffle=True)
    loadTestData = DataLoader(dataset=testData,batch_size=testData.n_sample, shuffle=False)
    neural_net = NeuralNetModel(11,10,args.N_LAYER,args.LAYER_SIZE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=neural_net.parameters(),lr=args.LR)
    loss_ep = []
    acc_ep = []
    for ep in range(args.EP):
        neural_net.train()
        loss_iter = []
        acc_iter = []
        for i, (x,y) in enumerate(loadTrainingData ):
            yp = neural_net(x)
            loss = loss_func(yp,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred = torch.softmax(yp,dim=1).argmax(dim=1)
            loss_iter.append(loss.detach().numpy())
            acc_iter.append(accuracy_fn(y,y_pred))
        
        loss_ep.append(loss_iter)
        acc_ep.append(acc_iter)

        neural_net.eval()
        with torch.inference_mode():
            for i, (xt,yt) in enumerate(loadTestData):
                yp_t = neural_net(xt)
                ytest = torch.softmax(yp_t,dim=1).argmax(dim=1)
                loss_test = loss_func(yp_t,yt)
                acc_t = accuracy_fn(ytest,yt)

        if ep%10 == 0:
            print(f"Epoch: {ep}|Training Loss:{np.mean(loss_iter)}| Training Accuracy:{np.mean(acc_iter)}| Validation Loss:{loss_test}| Validation Accuracy{acc_t}")