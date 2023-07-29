import torch
import torch.nn as nn
from DataTransform import Load_Polluted_Data
import os
import random
import numpy as np
import time
from torch.utils.data import TensorDataset,DataLoader,Dataset
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
os.environ['CUDA_VISIBLE_DEVICES']='4'
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


def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class USAD_Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class USAD_Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w

class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = USAD_Encoder(w_size, z_size)
        self.decoder1 = USAD_Decoder(z_size, w_size)
        self.decoder2 = USAD_Decoder(z_size, w_size)

    def forward(self, x, alpha=.5, beta=.5):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        score = alpha * torch.mean((x - w1) ** 2, axis=1) + beta * torch.mean((x - w2) ** 2, axis=1)

        return z, score, None

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        score = 0.5 * torch.mean((batch - w1) ** 2, axis=1) + 0.5 * torch.mean((batch - w2) ** 2, axis=1)
        others = {'loss1': loss1, 'loss2': loss2}
        return z, score, others

    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))

def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for x, y in train_loader:
            x = to_device(x, device)
            y = to_device(y, device)

            x = x.view([x.shape[0], x.shape[1] * x.shape[2]])

            # Train AE1
            feature, logits, others = model.training_step(x, epoch + 1)
            loss1, loss2 = others['loss1'], others['loss2']
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            feature, logits, others = model.training_step(x, epoch + 1)
            loss1, loss2 = others['loss1'], others['loss2']
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def testing(model, test_loader, alpha=.5, beta=.5):
    results = []
    for [batch] in test_loader:
        batch = to_device(batch, device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        results.append(alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1))
    return results

def USAD_Perf(Input,Target,dataname,normal_label,p):
    n_epochs = 50
    hidden_size = 100
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    w_size = np.size(Input, 1) * np.size(Input, 2)
    model = UsadModel(w_size, hidden_size)
    model = to_device(model, device)

    optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=0.001)
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=0.001)

    best_auc_roc = 0
    best_ap = 0

    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/pths/usad_{}_{}_{}.pth'.format(dataname,
                                                                                                           normal_label,
                                                                                                           p)
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/scores/usad_{}_{}_{}.npy'.format(dataname,
                                                                                                             normal_label,
                                                                                                             p)
    # construct training data
    train_data = torch.tensor(Input)
    train_target = torch.tensor(Target)
    train_dataset = TensorDataset(train_data, train_target)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    test_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)

    for epoch in range(n_epochs):

        time_start = time.time()
        model.train()
        for x, y in train_loader:
            x = to_device(x, device)
            # y = to_device(y, device)

            x = x.view([x.shape[0], x.shape[1] * x.shape[2]])

            # Train AE1
            z, score, others = model.training_step(x, epoch + 1)
            loss1, loss2 = others['loss1'], others['loss2']
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            z, score, others = model.training_step(x, epoch + 1)
            loss1, loss2 = others['loss1'], others['loss2']
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        time_end = time.time()
        # print('Epoch: {}  Train Time: {}'.format(epoch,time_end-time_start))

        results = []
        alpha = 0.5
        beta = 0.5

        model.eval()
        for x, y in test_loader:
            batch = x
            batch = to_device(batch, device)
            batch = batch.view(batch.shape[0], batch.shape[1] * batch.shape[2])
            w1 = model.decoder1(model.encoder(batch))
            w2 = model.decoder2(model.encoder(w1))
            results.append(alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1))

        if len(results) == 1:
            y_pred = results[-1].flatten().detach().cpu().numpy()
        else:
            y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                     results[-1].flatten().detach().cpu().numpy()])

        auc_roc = roc_auc_score(Target[:, 0], y_pred)
        ap = average_precision_score(Target[:, 0], y_pred)
        #print('epoch', epoch, 'auc_roc:', auc_roc, 'auc_pr:', ap, 'time:', str(convert_time(time_end - time_start)))

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, y_pred)

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)

if __name__=='__main__':
    dataname='HAR'
    A=np.load('/tmp/FDAD/DesampleData/'+dataname+'.npz',allow_pickle=True)
    A_data,A_label=A['data'],A['label']
    l_list=len(np.unique(A_label[0]))
    print('dataname: {}, number of categories: {}'.format(dataname,l_list))
    for i in range(l_list):
        for j in range(5):

            p=j*0.1+0.1
            print("Normal_label:--{} P--{}".format(i,p))
            data,label=Load_Polluted_Data(A_data,A_label,i,p=p)

            Input,Target=np.concatenate(data,axis=0),np.concatenate(label,axis=0)
            USAD_Perf(Input,Target,dataname,i,p)