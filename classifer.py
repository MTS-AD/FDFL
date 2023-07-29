import numpy as np
import torch
import torchvision
import h5py
import os
import random
from torch.utils.data import TensorDataset,DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
os.environ['CUDA_VISIBLE_DEVICES']='3'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_data(DATA_PATH):
    f = h5py.File(DATA_PATH, 'r')
    domain = f.keys()
    client_num = len(domain)
    # restore data into the list, data[i] represents the MTS data of i-th client, label[i] denotes the label information.
    data = [[] for _ in range(client_num)]
    label = [[] for _ in range(client_num)]
    for index, i in enumerate(domain):
        data[index] = f[i]['data'][:]
        label[index] = f[i]['labels'][:]

    label_list = np.unique(label[0])
    return data,label,label_list

class LSTM_Net(nn.Module):
    def __init__(self,hidden_dim,input_dim,num_layer,num_class):
        super(LSTM_Net, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.rep=nn.LSTM(self.input_dim,self.hidden_dim,num_layers=num_layer,
                         bidirectional=False,batch_first=True,dropout=0.1)
        self.mlp=nn.Linear(hidden_dim,num_class)

    def forward(self, x):
        x,_=self.rep(x)
        x=self.mlp(x[:,-1,:])
        return x

def inference(epoch,testloader,Net):
    correct=0
    total=0
    with torch.no_grad():
        for item in testloader:
            seq, target = item
            seq, target = seq.float().to(device), target.to(device)
            output = Net(seq)
            tempred=torch.argmax(output,1)
            correct+=(tempred==target[:,0]).sum().float()
            total+=len(target)
    print('EPOCH: {} ----Accuracy: {} '.format(epoch,((correct/total).cpu().detach().data.numpy())))

if __name__=='__main__':
    from DataTransform import desample
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataname = 'SEDFx'

    data, label, label_list = load_data('/remote-home/heyf/FDAD_data/' + dataname + '.h5')

    if dataname != 'HAR':
        data = desample(data, 5)

    hidden_dim=128
    input_dim=4
    num_class=len(label_list)
    num_layer=2

    Loss_func=nn.CrossEntropyLoss()
    Net=LSTM_Net(hidden_dim,input_dim,num_layer,num_class).to(device)
    #load train data
    data_tensor=torch.tensor(data[1])
    target_tensor=torch.tensor(label[1])
    # np.random.shuffle(data[1])
    # np.random.shuffle(label[1])
    # datath=0
    # index=np.array(range(len(data[datath])))
    # np.random.shuffle(index)
    # num=round(len(index)*0.8)
    # data_tensor = torch.tensor(data[datath][index[:num],:])
    # target_tensor = torch.tensor(label[datath][index[:num],:])
    dataset=TensorDataset(data_tensor,target_tensor)
    dataloader=DataLoader(dataset,batch_size=64,shuffle=True)
    #load test data

    test_data=torch.tensor(data[1])
    test_target=torch.tensor(label[1])
    testdataset=TensorDataset(test_data,test_target)
    testloader=DataLoader(testdataset,batch_size=64)

    #Train the model
    optimizer=optim.Adam(Net.parameters(),lr=0.001,weight_decay=0.001)
    max_epochs=1000
    for epoch in range(max_epochs):
        for item in dataloader:
            seq,target=item
            seq,target=seq.float().to(device),target.to(device)
            optimizer.zero_grad()
            output=Net(seq)

            loss=Loss_func(output,target[:,0])
            loss.backward()
            optimizer.step()
            #print(loss.item())
        if epoch%1 == 0:
            inference(epoch,testloader,Net)


            




