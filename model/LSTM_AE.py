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
#
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class PyTorchUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0
        self.torch_save = True

    def to_var(self, t, **kwargs):
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)


class LSTMAE(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple, device: str):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.device = device

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = True):
        batch_size = ts_batch.shape[0]

        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        dec_hidden = enc_hidden

        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        output_flatten = output.reshape((output.shape[0], output.shape[1] * output.shape[2]))
        ts_batch_flatten = ts_batch.reshape((ts_batch.shape[0], ts_batch.shape[1] * ts_batch.shape[2]))
        rec_err = torch.abs(output_flatten ** 2 - ts_batch_flatten ** 2)
        rec_err = torch.sum(rec_err, dim=1)
        #output = output[:, -1, :]
        others = {}
        others['output'] = output
        return enc_hidden[1][-1], rec_err, others if return_latent else output



def LSTMAE_Perf(Input,Target,dataname,normal_label,p):
    args = {'epochs': 30, 'batch_size': 128, 'lr': 0.001, 'hidden_size': 128, 'n_layers': (1, 1),
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


    model = LSTMAE(n_features=np.size(Input,2),hidden_size=args['hidden_size'],n_layers=args['n_layers'],use_bias=args['use_bias'],dropout=args['dropout'],device=device).to(device)

    optimizer=torch.optim.Adam(model.parameters(),lr=args['lr'])



    best_auc_roc = 0
    best_ap = 0

    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/pths/lstmae_{}_{}_{}.pth'.format(dataname,
                                                                                                           normal_label,
                                                                                                           p)
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/scores/lstmae_{}_{}_{}.npy'.format(dataname,
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
            feature, logits, others = model(x)
            if 'output' in others.keys():
                pred_x= others['output']
            loss = args['criterion'](pred_x,x)
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
                feature, logits, others = model(x)
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

def main1(dataname):
    A = np.load('/tmp/FDAD/DesampleData/' + dataname + '.npz', allow_pickle=True)
    A_data, A_label = A['data'], A['label']
    l_list = len(np.unique(A_label[0]))
    print('dataname: {}, number of categories: {}'.format(dataname, l_list))
    for i in range(l_list):
        p = 0.1
        print("Normal_label:--{} P--{}".format(i, p))
        data, label = Load_Polluted_Data(A_data, A_label, i, p=p)

        Input, Target = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
        LSTMAE_Perf(Input, Target, dataname, i, p)


if __name__=='__main__':
    device=torch.device('cuda:7')
    main1('HAR')

