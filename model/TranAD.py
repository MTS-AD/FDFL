import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerDecoder
from DataTransform import Load_Polluted_Data
import os
import random
import numpy as np
import time
from torch.utils.data import TensorDataset,DataLoader,Dataset
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
os.environ['CUDA_VISIBLE_DEVICES']='6'
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats,n_window):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = 0.001
        self.batch = 128
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        hidden1 = self.encode(src, c, tgt)
        x1 = self.fcn(self.transformer_decoder1(*hidden1))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        hidden2 = self.encode(src, c, tgt)
        x2 = self.fcn(self.transformer_decoder2(*hidden2))
        features1 = torch.squeeze(hidden1[0], dim=0)
        features2 = torch.squeeze(hidden2[0], dim=0)
        features = torch.cat((features1, features2), dim=2)
        features=features.permute(1,0,2)
        # print(x2.shape, tgt.shape, src.shape)
        logits = (x2 - tgt) ** 2
        # print(logits.shape)
        others = {'x1': x1, 'x2': x2}
        return x2, logits, others

def TranAD_Perf(Input,Target,dataname,normal_label,p):
    random_seed=2023
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    if dataname=='HAR':
        n_window=500
    else:
        n_window=300
    feats=Input.shape[-1]
    model=TranAD(feats,n_window).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),model.lr,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    epochs=100
    l1s,l2s=[],[]
    l=nn.MSELoss(reduction='none')
    best_auc_roc=0
    best_ap=0

    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/pths/tranad_{}_{}_{}.pth'.format(dataname,normal_label,p)
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/scores/tranad_{}_{}_{}.npy'.format(dataname,normal_label,p)
    # construct training data
    train_data = torch.tensor(Input)
    train_target = torch.tensor(Target)
    train_dataset = TensorDataset(train_data, train_target)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    test_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)
    time_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_items = 0
        n = epoch + 1
        for d, _ in train_loader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2).to(device)
            elem = window[-1, :, :].view(1, local_bs, feats).to(device)
            features, logits, others = model(window, elem)
            z = (others['x1'], others['x2'])
            l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.item() * local_bs
            epoch_items += local_bs
        scheduler.step()
       # print('epoch', n, 'loss:', (epoch_loss / epoch_items), end='')

        model.eval()
        test_losses = []
        for d, _ in test_loader:
            window = d.permute(1, 0, 2).to(device)
            elem = window[-1, :, :].view(1, d.shape[0], feats).to(device)
            features, logits, others = model(window, elem)
            z = (others['x1'], others['x2'])
            if isinstance(z, tuple): z = z[1]
            test_losses.append(z[0].detach().cpu().numpy())
        test_losses = np.concatenate(test_losses, axis=0)
        test_losses = np.mean(test_losses, axis=1)

        labels =Target
        auc_roc = roc_auc_score(labels, test_losses)
        ap = average_precision_score(labels, test_losses)
        #print(' auc_roc:', auc_roc, 'auc_pr:', ap)

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, test_losses)

    time_end = time.time()

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))
    return 0

if __name__=='__main__':
    dataname='SEDFx'
    A=np.load('/tmp/FDAD/DesampleData/'+dataname+'.npz',allow_pickle=True)
    A_data,A_label=A['data'],A['label']
    l_list = len(np.unique(A_label[0]))
    print('dataname: {}, number of categories: {}'.format(dataname, l_list))
    for i in range(l_list):
        #for j in range(5):

            p=0.1
            print("Normal_label:--{} P--{}".format(i,p))
            data,label=Load_Polluted_Data(A_data,A_label,i,p=p)

            Input,Target=np.concatenate(data,axis=0),np.concatenate(label,axis=0)
            TranAD_Perf(Input,Target,dataname,i,p)