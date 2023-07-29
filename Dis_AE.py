import abc
from torch import nn
import torch
from torch.autograd import Variable
from DataTransform import Load_Polluted_Data
import os
import random
import numpy as np
import time
from torch.utils.data import TensorDataset,DataLoader,Dataset
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES']='6,7'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_time(t):
    hours = int(t / 3600)
    minutes = int(t / 60 % 60)
    seconds = t - 3600 * hours - 60 * minutes
    hours_str = str(hours)
    if len(hours_str) < 2:
        hours_str = '0' * (2 - len(hours_str)) + hours_str
    minutes_str = str(minutes)
    if len(minutes_str) < 2:
        minutes_str = '0' * (2 - len(minutes_str)) + minutes_str
    seconds_left = str(seconds).split('.')[0]
    seconds_str = str(seconds)
    if len(seconds_left) < 2:
        seconds_str = '0' * (2 - len(seconds_left)) + seconds_str
    return '' + hours_str + ':' + minutes_str + ':' + seconds_str

class global_AE(nn.Module):
    def __init__(self,n_features,hidden_size,n_layers,dropout):
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_layers = n_layers
        self.dropout = dropout
        super(global_AE, self).__init__()
   
        self.global_encoder=nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], dropout=self.dropout[0])

        self.global_decoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True,
                                      num_layers=self.n_layers[1], dropout=self.dropout[1])

    def _init_hidden(self, batch_size):
        return (torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(device),
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(device))

    def forward(self,ts_batch):
        batch_size = ts_batch.shape[0]
        # global AE
        gl_enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        gl_enc_features, gl_enc_hidden = self.global_encoder(ts_batch.float(),
                                                             gl_enc_hidden)  # .float() here or .double() for the model
        gl_dec_features, gl_dec_hidden = self.global_decoder(gl_enc_features.float())

        return (gl_enc_features,gl_enc_hidden),(gl_dec_features,gl_dec_hidden)

class local_AE(nn.Module):
    def __init__(self,n_features,hidden_size,n_layers,dropout):
        super(local_AE, self).__init__()

        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_layers = n_layers
        self.dropout = dropout
        self.local_encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                                     num_layers=self.n_layers[0], dropout=self.dropout[0])

        self.local_decoder = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True,
                                     num_layers=self.n_layers[1], dropout=self.dropout[1])
    def _init_hidden(self, batch_size):
        return (torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(device),
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(device))
    def forward(self,ts_batch):
        batch_size = ts_batch.shape[0]
        # local_AE
        lo_enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        lo_enc_features, lo_enc_hidden = self.local_encoder(ts_batch.float(),
                                                            lo_enc_hidden)  # .float() here or .double() for the model
        lo_dec_features, lo_dec_hidden = self.local_decoder(lo_enc_features.float())

        return (lo_enc_features,lo_enc_hidden),(lo_dec_features,lo_dec_hidden)

class DIS_LSTMAE(nn.Module):
    def __init__(self,n_features,hidden_size,n_layers,dropout):
        super(DIS_LSTMAE, self).__init__()
        self.hidden_size=hidden_size
        self.n_features=n_features
        self.n_layers=n_layers
        self.dropout=dropout

        self.gl_AE=global_AE(self.n_features,self.hidden_size,self.n_layers,self.dropout)
        self.lo_AE=local_AE(self.n_features,self.hidden_size,self.n_layers,self.dropout)

        self.hidden2output  = nn.Linear(self.hidden_size, self.n_features)

    def _init_hidden(self, batch_size):
        return (torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(device),
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_().to(device))

    def forward(self, ts_batch):

        gl_enc,gl_dec = self.gl_AE(ts_batch.float())  # .float() here or .double() for the model


        #local_AE

        lo_enc,lo_dec = self.lo_AE(ts_batch.float()) # .float() here or .double() for the model


        output=self.hidden2output(gl_dec[0])+self.hidden2output(lo_dec[0])

        output_flatten = output.reshape((output.shape[0], output.shape[1] * output.shape[2]))
        ts_batch_flatten = ts_batch.reshape((ts_batch.shape[0], ts_batch.shape[1] * ts_batch.shape[2]))
        rec_err = torch.abs(output_flatten ** 2 - ts_batch_flatten ** 2)
        rec_err = torch.sum(rec_err, dim=1)

        return (gl_enc[0],gl_dec[0]),(lo_enc[0],lo_dec[0]),rec_err,output


def DIS_LSTMAE_Perf(Input,Target,dataname,normal_label,p):
    args = {'epochs': 100, 'batch_size': 128, 'lr': 0.001, 'hidden_size': 128, 'n_layers': (1, 1),
            'use_bias': (True, True), 'dropout': (0, 0), 'criterion': nn.MSELoss(), 'random_seed': 42}


    random_seed =args['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)


    model = DIS_LSTMAE(n_features=np.size(Input,2),hidden_size=args['hidden_size'],n_layers=args['n_layers'],dropout=args['dropout']).to(device)

    optimizer=torch.optim.Adam(model.parameters(),lr=args['lr'])



    best_auc_roc = 0
    best_ap = 0

    model_save_path = os.path.abspath(os.path.join(os.getcwd())) + '/mypaths/DISlstmae_{}_{}_{}.pth'.format(dataname,
                                                                                                           normal_label,
                                                                                                           p)
    score_save_path = os.path.abspath(os.path.join(os.getcwd())) + '/myscores/DISlstmae_{}_{}_{}.npy'.format(dataname,
                                                                                                             normal_label,
                                                                                                             p)
    # construct training data
    train_data = torch.tensor(Input)
    train_target = torch.tensor(Target)
    train_dataset = TensorDataset(train_data, train_target)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    test_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)

    for e in range(args['epochs']):

        time_start = time.time()
        model.train()
        epoch_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x=x.to(device)
            #y.to(device)
            # print(x.detach().cpu().numpy().max(), x.detach().cpu().numpy().min())
            optimizer.zero_grad()
            gl_features,lo_features, logits, others = model(x)
            loss = args['criterion'](others,x)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        #print('epoch', e + 1, 'loss', epoch_loss, end=' ')

        scores = []
        ys = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                #
                optimizer.zero_grad()
                gl_features,lo_features, logits, others = model(x)
                scores.append(logits.detach().cpu().numpy())
                ys.append(y.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)
        ys = np.concatenate(ys, axis=0)
        # print(scores.shape, ys.shape)
        if len(scores.shape) == 2:
            scores = np.squeeze(scores, axis=1)
        if len(ys.shape) == 2:
            ys = np.squeeze(ys, axis=1)
        auc_roc = roc_auc_score(ys, scores)
        ap = average_precision_score(ys, scores)
        print('auc-roc: ' + str(auc_roc) + ' auc_pr: ' + str(ap))

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, scores)
        #     print(' update')
        # else:
        #     print('\n', end='')

    time_end = time.time()

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))

if __name__=='__main__':
    args = {'epochs': 30, 'batch_size': 128, 'lr': 0.001, 'hidden_size': 128, 'n_layers': (1, 1),
            'use_bias': (True, True), 'dropout': (0.1, 0.1), 'criterion': nn.MSELoss(), 'random_seed': 42}

    dataname='HAR'
    A = np.load('/tmp/FDAD/DesampleData/' + dataname + '.npz', allow_pickle=True)
    A_data, A_label = A['data'], A['label']
    l_list = len(np.unique(A_label[0]))
    #for i in range(l_list):
    i=5
    p = 0.1
    print("Normal_label:--{} P--{}".format(i, p))
    data, label = Load_Polluted_Data(A_data, A_label, i, p=p)
    Input, Target = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
    DIS_LSTMAE_Perf(Input, Target, dataname, i, p)






