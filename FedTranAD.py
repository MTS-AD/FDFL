import abc
import os
from torch import nn
import torch
from torch.autograd import Variable
from DataTransform import Load_Polluted_Data
import random
import numpy as np
import time
from torch.utils.data import TensorDataset,DataLoader,Dataset
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from model.TranAD import  TranAD,convert_time
from client_mix import get_init_grad_correct,config
import torch.optim as optim
import torch.nn.functional as F
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='3'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def average_weights(state_dicts, fed_avg_freqs):
    # init
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = state_dicts[0][key] * fed_avg_freqs[0]

    state_dicts = state_dicts[1:]
    fed_avg_freqs = fed_avg_freqs[1:]
    for state_dict, freq in zip(state_dicts, fed_avg_freqs):
        for key in state_dict.keys():
            avg_state_dict[key] += state_dict[key] * freq
    return avg_state_dict

def local_train(global_state_dict,client_loader,global_round,state_dict_prev,local_c,global_correct):

    cos_sim = torch.nn.CosineSimilarity(dim=-1).to(device)
    if dataname=='HAR':
        n_window=500
    else:
        n_window=300
    feats=A_data[0].shape[-1]
    model_global=TranAD(feats,n_window)
    model_global.load_state_dict(global_state_dict)
    model_global.requires_grad_(False)
    model_global.eval()
    model_global.to(device)

    model_current=TranAD(feats,n_window)
    model_current.load_state_dict(global_state_dict)
    model_current.requires_grad_(True)
    model_current.train()
    model_current.to(device)
    optimizer=optim.Adam(model_current.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    l = nn.MSELoss(reduction='none')
    l1s, l2s = [], []
    for local_epoch in range(5):
        n = local_epoch + 1
        for i,(x,y) in enumerate(client_loader):
            x = x.to(device)
            optimizer.zero_grad()
            local_bs = x.shape[0]
            feats = x.shape[-1]
            window = x.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feats)
            feature, logits, others = model_current(window, elem)

            z = (others['x1'], others['x2'])
            l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)

            #Fedprox and moon will change the loss
            if args['alg']=='fedprox':
                proximal_term = 0
                for w, w_0 in zip(model_current.parameters(), model_global.parameters()):
                    proximal_term += (w - w_0).norm(2)
                loss += proximal_term * 0.01 / 2

            if args['alg'] == 'moon' and state_dict_prev is not None:
                model_prev = TranAD(feats,n_window)
                model_prev.load_state_dict(state_dict_prev)
                model_prev.requires_grad_(False)
                model_prev.eval()
                model_prev.to(device)

                feature_prev, _, _ = model_prev(window, elem)
                feature_global, _, _ = model_global(window, elem)
                feature_flatten = torch.flatten(feature, start_dim=1)
                feature_global_flatten = torch.flatten(feature_global, start_dim=1)
                feature_prev_flatten = torch.flatten(feature_prev, start_dim=1)
                posi = cos_sim(feature_flatten, feature_global_flatten)
                logits_moon = posi.reshape(-1, 1)
                nega = cos_sim(feature_flatten, feature_prev_flatten)
                logits_moon = torch.cat((logits_moon, nega.reshape(-1, 1)), dim=1)
                logits_moon /= 0.5
                loss_con = F.cross_entropy(
                    logits_moon,
                    torch.zeros(x.size(0), device=device, dtype=torch.long)
                )
                loss = loss + 1 * loss_con



#update local model
            loss.backward()
            optimizer.step()


            #scaffold add global c and local c to the model.
            if args['alg']=='scaffold':
                state_dict_current=model_current.state_dict()
                lr = optimizer.state_dict()['param_groups'][-1]['lr']
                for key in state_dict_current:
                    if key == 'pos_encoder.pe':
                        continue
                    c_global = global_correct[key].to(device)
                    c_local = local_c[key].to(device)
                    state_dict_current[key] -= lr * (c_global - c_local)
                model_current.load_state_dict(state_dict_current)
            scheduler.step()
    model_current.cpu()
    return model_current.state_dict()

def array2loader(data,label):
    train_data = torch.tensor(data)
    train_target = torch.tensor(label)
    train_dataset = TensorDataset(train_data, train_target)
    return DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

def test_inference(global_state_dict,test_loader):

    if dataname=='HAR':
        n_window=500
    else:
        n_window=300
    feats=A_data[0].shape[-1]
    model=TranAD(feats,n_window)
    model.load_state_dict(global_state_dict)
    model=model.to(device)
    model.eval()
    scores = []
    ys = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            window = x.permute(1, 0, 2).to(device)
            elem = window[-1, :, :].view(1, x.shape[0], feats).to(device)
            features, logits, others = model(window, elem)
            z = (others['x1'], others['x2'])
            if isinstance(z, tuple): z = z[1]
            scores.append(z[0].detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    scores = np.concatenate(scores, axis=0)
    scores = np.mean(scores, axis=1)
    ys = np.concatenate(ys, axis=0)
    # print(scores.shape, ys.shape)
    if len(scores.shape) == 2:
        scores = np.squeeze(scores, axis=1)
    if len(ys.shape) == 2:
        ys = np.squeeze(ys, axis=1)
    auc_roc = roc_auc_score(ys, scores)
    ap = average_precision_score(ys, scores)
    return auc_roc,ap,scores

def FedTranAD_Perf(A_data, A_label, dataname, i, p):
    if dataname=='HAR':
        n_window=500
    else:
        n_window=300
    feats=A_data[0].shape[-1]

    random_seed =args['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)


    data, label = Load_Polluted_Data(A_data, A_label, i, p)

    test_loader=array2loader(np.concatenate(data, axis=0), np.concatenate(label, axis=0))

    model = TranAD(feats,n_window)
    global_state_dict=model.state_dict()

    #define c of scaffold
    global_correct=get_init_grad_correct(model)
    local_corrects=[get_init_grad_correct(model) for i in range(len(data))]
    state_dict_prevs=[None for _ in range(len(data))]

    best_auc_roc=0
    best_ap=0
    model_save_path = os.path.abspath(os.getcwd()) + '/pths/{}_{}_{}_{}_{}.pth'.format(args['alg'] ,args['tsadalg'] , dataname,i,p)
    score_save_path = os.path.abspath( os.getcwd()) + '/scores/{}_{}_{}_{}_{}.npy'.format(args['alg'] ,args['tsadalg'] , dataname,i,p)

    client_loaders=[array2loader(data[i],label[i]) for i in range(len(data))]
    time_start=time.time()
    print('Begin training')
    for global_round in range(config['epochs']):
        t_start=time.time()

        num_active_client=int((len(data)*args['client_rate']))
        ind_active_clients=np.random.choice(range(len(data)),num_active_client,replace=False)

        active_clients=[client_loaders[i] for i in ind_active_clients]


        data_nums = [len(data[i]) for i in ind_active_clients]

        active_state_dict = []

        for i in range(len(active_clients)):
            client_loader=client_loaders[i]

            local_c = local_corrects[ind_active_clients[i]]
            state_dict_prev=state_dict_prevs[ind_active_clients[i]]
            current_state_dict=local_train(global_state_dict,client_loader,global_round,state_dict_prev,local_c,global_correct)
            state_dict_prevs[ind_active_clients[i]]=current_state_dict
            active_state_dict.append(current_state_dict)

            #update  local c of scaffold
            if args['alg']=='scaffold':
                for key in local_c.keys():
                    old_c=local_c[key].to(device)
                    new_c=old_c-global_correct[key].to(device)+ \
                          (global_state_dict[key].to(device) - current_state_dict[key].to(device)) / (len(client_loader) * 5 * 0.001)
                    local_c[key]=new_c.cpu()
                local_corrects[ind_active_clients[i]]=local_c
        #update global c of scaffold
        if args['alg'] == 'scaffold':
            global_correct=average_weights([local_corrects[xx] for xx in ind_active_clients],[1/len(active_clients)]*len(active_clients))


        fed_freq=torch.tensor(data_nums,dtype=torch.float)/sum(data_nums)
        global_state_dict=average_weights(active_state_dict,fed_freq)


        t_end=time.time()
        if (global_round + 1) % 1 == 0:
            auc_roc,ap,scores=test_inference(global_state_dict,test_loader)
            #print('Global_round:{}  AUC:{}  AP:{} Time:{}'.format(global_round,auc_roc,ap,convert_time(t_end - t_start)))
            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_ap = ap
                torch.save(global_state_dict, model_save_path)
                np.save(score_save_path, scores)
    time_end=time.time()
    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))
    return best_auc_roc,best_ap,convert_time(time_end - time_start)

def save_result(result_file_path,auc,ap,total_time,fedalg,adalg,dataname,i,p):
    s='fedalg:--{} adalg:--{} dataname:--{} Normal_label:--{} P--{} bestauc:--{} bestap:--{} total_time:{}'.format(fedalg,adalg,dataname,i,p,auc,ap,total_time)
    with open(result_file_path, 'a') as f:
        f.write(s + '\n')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--dataname', type=str, default='HAR')
    parser.add_argument('--alg', type=str, default='moon')
    argsx= parser.parse_args()
    args = {'epochs': 30, 'batch_size': 128, 'lr': 0.001, 'hidden_size': 128, 'n_layers': (1, 1),
            'use_bias': (True, True), 'dropout': (0, 0), 'criterion': nn.MSELoss(), 'random_seed': 42,
            'alg': argsx.alg, 'tsadalg': 'tranad', 'client_rate': 1}

    dataname=argsx.dataname
    A = np.load('/tmp/FDAD/DesampleData/' + dataname + '.npz', allow_pickle=True)
    A_data, A_label = A['data'], A['label']
    l_list = len(np.unique(A_label[0]))
    print('fedalg:{} adalg:{} dataname: {}, number of categories: {}'.format(args['alg'],args['tsadalg'],dataname, l_list))
    result_file_path='Fed'+args['tsadalg']+'_result.txt'
    for i in range(l_list):
        for j in range(1,5):
            p = 0.1+j*0.1
            print("Normal_label:--{} P--{}".format(i, p))
            auc,ap,total_time=FedTranAD_Perf(A_data,A_label, dataname, i, p)
            save_result(result_file_path, auc, ap, total_time, args['alg'], args['tsadalg'], dataname,i,p)







