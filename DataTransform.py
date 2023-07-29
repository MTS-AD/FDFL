import numpy as np
import torch
import torchvision
import h5py
import os
import random
from torch.utils.data import TensorDataset,DataLoader,Dataset

def desample(data,scale):
    L=np.shape(data[0])[1]
    index=np.array(range(0,L,scale))
    for i in range(len(data)):
        data[i]=data[i][:,index,:]
    return data


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

    return data,label

def conv_label(normal_label,label):
    convert_label = [[] for _ in range(len(label))]
    for j in range(len(label)):
        convert_label[j] = np.where(label[j] == normal_label, 0, 1)
    return convert_label

def load_data_with_outlier(data,convert_label,p):
    Polluted_data=[[] for _ in range(len(data))]
    Polluted_label=[[] for _ in range(len(convert_label))]
    for i in range(len(data)):
        local_data=data[i]
        local_label=convert_label[i]
        normal_index=np.where(local_label==0)[0]
        anomaly_index=np.where(local_label==1)[0]
        anomaly_num=round(len(normal_index)*p)
        np.random.seed(2023)
        anomaly_index=np.random.choice(anomaly_index,anomaly_num,replace=False)
        #print(anomaly_index)
        Polluted_data[i]=np.concatenate([local_data[normal_index,:],local_data[anomaly_index,:]])
        Polluted_label[i]=np.concatenate([local_label[normal_index,:],local_label[anomaly_index,:]])
    return Polluted_data,Polluted_label

def MinMaxNorm(A):
    for i in range(np.size(A,2)):
        MAX=np.max(A[:,:,i])
        MIN=np.min(A[:,:,i])
        A[:,:,i]=(A[:,:,i]-MIN)/(MAX-MIN)
    return A

def Load_Polluted_Data(data,label,normal_label,p=0.1,normalize=True):

    convert_label = conv_label(normal_label, label)
    Polluted_data, Polluted_label = load_data_with_outlier(data, convert_label, p)

    if normalize==True:
        for j in range(len(Polluted_data)):
            Polluted_data[j]=MinMaxNorm(Polluted_data[j])

    return Polluted_data,Polluted_label

if __name__=='__main__':
    dataname='SEDFx'
    data, label = load_data('/remote-home/heyf/FDAD_data/'+dataname+'.h5')
    num=0
    print(len(data))
    for i in data:
        num+=len(i)
    print(num,np.shape(i))
    print(len(np.unique(label[0])))


    if dataname!='HAR':
        data=desample(data,10)
    if dataname=='SEDFx':

        for i in range(len(data)):
            L = np.shape(data[i])[0]
            index = np.array(range(0, L, 5))
            data[i] = data[i][index,:, :]
            label[i]=label[i][index,:]

    data=np.array(data)
    label=np.array(label)

    np.savez(os.getcwd()+'/DesampleData/'+dataname,data=data,label=label)



