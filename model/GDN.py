# -*- coding: utf-8 -*-
import sys
from DataTransform import Load_Polluted_Data
import os
import random
import math
import time
import pandas as pd
import numpy as np
from datetime import datetime
from torch import nn
import torch
from pathlib import Path
from torch.utils.data import TensorDataset,DataLoader,Dataset,Subset
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot, zeros
from model.USAD import convert_time

def get_fc_graph_struc(columns):
    struc_map={}
    feature_list=columns
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)

    return struc_map

def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res

def build_loc_net(struc, all_features, feature_map=[]):
    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes

class TimeDataset(Dataset):
    def __init__(self, Input,Target, edge_index):



        self.edge_index = edge_index.long()
        x_data = Input
        labels = Target

        # to tensor
        self.x = torch.tensor(x_data).double()
        self.y = torch.tensor(x_data).double()
        self.labels = torch.tensor(labels).double()
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feature = self.x[idx]
        y = self.y[idx]
        edge_index = self.edge_index

        label = self.labels[idx]

        return feature, y, label, edge_index

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()

class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num,input_dim ,inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, input_dim))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out

class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)


class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index,x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)

        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        # print(inputs.shape)
        # print(index.shape)吗，
        return scatter(inputs, index, dim=0, dim_size=dim_size,
                       reduce=self.aggr)
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1,
                 topk=20):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num,input_dim, inter_num=out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index, labels):

        x = data.clone().detach()  # (批量大小, 维度, 窗口长度)
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.reshape(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings)
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        out_features = torch.mul(x, self.embedding(indexes))

        out = out_features.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.permute(0,2,1)

        #out = out.float().to(device)

        loss = loss_func(out, labels)

        return out_features, out, loss  # out_features: (批量大小, 维度数量, dims), out: 预测结果 (批量大小, 维度数量)

def GDN_Perf(Input,Target,dataname,normal_label,p):
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # construct training data
    train_data = torch.tensor(Input)
    train_target = torch.tensor(Target)
    # train_dataset = TensorDataset(train_data, train_target)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    #test_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)

    columns=[str(i) for i in range(Input.shape[-1])]

    feature_map=columns
    fc_struc=get_fc_graph_struc(columns)
    fc_edge_index = build_loc_net(fc_struc, list(columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    train_dataset = TimeDataset(Input,Target,fc_edge_index)

    train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=False)


    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/pths/gdn_{}_{}_{}.pth'.format(dataname, normal_label,p)
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/scores/gdn_{}_{}_{}.npy'.format(dataname,normal_label,p)

    edge_index_sets=[]
    edge_index_sets.append(fc_edge_index)

    model=GDN(edge_index_sets,len(feature_map),input_dim=Input.shape[1],topk=Input.shape[-1]).to(device)
    #print(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.99))
    best_auc_roc = 0
    best_ap = 0
    epochs=100
    #Train the model

    for i_epoch in range(epochs):
        time_start=time.time()
        acu_loss=0

        model.train()
        for x,labels,attack_labes,edge_index in train_dataloader:
            x=x.permute(0,2,1)
            x,labels,edge_index=[item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()

            _,out,loss=model(x,edge_index,labels)
            loss.backward()
            optimizer.step()
            acu_loss+=loss.item()
        # print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
        #                 i_epoch, epochs,
        #                 acu_loss/len(train_dataloader), acu_loss))
    #Test model
        results = []
        model.eval()
        for x,labels,attack_labes,edge_index in test_dataloader:
            x = x.permute(0, 2, 1)
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            _, out, loss = model(x, edge_index, labels)
            results.append(torch.mean((out-labels) ** 2, axis=(1,2)))

        if len(results) == 1:
            y_pred = results[-1].flatten().detach().cpu().numpy()
        else:
            y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                     results[-1].flatten().detach().cpu().numpy()])

        auc_roc = roc_auc_score(Target[:, 0], y_pred)
        ap = average_precision_score(Target[:, 0], y_pred)
        time_end=time.time()
        #print('epoch', i_epoch, 'auc_roc:', auc_roc, 'auc_pr:', ap, 'time:', str(convert_time(time_end - time_start)))

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, y_pred)

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)


if __name__ == "__main__":
    device=torch.device('cuda:7')
    dataname = 'SEDFx'
    A = np.load('/tmp/FDAD/DesampleData/' + dataname + '.npz', allow_pickle=True)
    A_data, A_label = A['data'], A['label']
    l_list = len(np.unique(A_label[0]))
    print('dataname: {}, number of categories: {},device: {}'.format(dataname, l_list,device))
    for i in range(l_list):
            p =  0.1
            print("Normal_label:--{} P--{}".format(i, p))
            data, label = Load_Polluted_Data(A_data, A_label, i, p=p)

            Input, Target = np.concatenate(data, axis=0), np.concatenate(label, axis=0)
            GDN_Perf(Input, Target, dataname, i, p)





