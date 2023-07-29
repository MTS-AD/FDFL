import warnings
warnings.filterwarnings("ignore")
import json
import math
import time

import torch
from torch import nn, optim
from abc import ABC, abstractmethod
import torch.nn.functional as F
import logging
from torch.autograd import Variable
from DataTransform import Load_Polluted_Data
import os
import random
import numpy as np
from torch.utils.data import TensorDataset,DataLoader,Dataset
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES']='3'
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

class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__


def build_network(net_name):
    """Builds the neural network."""

    # implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'smd_mlp', 'smap_mlp', 'psm_mlp')
    # assert net_name in implemented_networks
    #
    # net = None

    if net_name == 'HAR':
        net = MTS_MLP(500*6)
    elif net_name == 'CAP':
        net = MTS_MLP(300*19)
    elif net_name == 'SEDFx':
        net = MTS_MLP(300*4)
    return net


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

class MTS_MLP(BaseNet):
    def __init__(self,in_size):
        super(MTS_MLP, self).__init__()
        self.rep_dim=32
        self.in_size=in_size
        self.linear1=nn.Linear(self.in_size,128)
        self.linear2=nn.Linear(128,64)
        self.linear3=nn.Linear(64,self.rep_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x

class MTS_Autoencoder(BaseNet):
    def __init__(self,in_size):
        super(MTS_Autoencoder, self).__init__()
        self.in_size=in_size
        self.rep_dim=32
        #Encoder
        self.linear1 = nn.Linear(self.in_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, self.rep_dim)

        # Decoder
        self.linear4 = nn.Linear(self.rep_dim, 64)
        self.linear5 = nn.Linear(64, 128)
        self.linear6 = nn.Linear(128,in_size)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        h = self.linear3(x)
        x = self.linear4(h)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        return h, x


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    # implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'smd_mlp', 'smap_mlp', 'psm_mlp')
    # assert net_name in implemented_networks

    ae_net = None
    if net_name == 'HAR':
        ae_net = MTS_Autoencoder(500*6)
    if net_name == 'CAP':
        ae_net = MTS_Autoencoder(300*19)
    if net_name == 'SEDFx':
        ae_net = MTS_Autoencoder(300*4)

    return ae_net


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def train(self, dataset: Dataset, ae_net: BaseNet):
        loss_min = math.inf
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        # train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            ae_net.train()
            for data in train_loader:
                # inputs, _, _ = data
                inputs, _ = data
                inputs=inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                outputs = outputs[1]
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            ae_net.eval()



            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time


            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
           # print('epoch', epoch + 1, 'time:', epoch_train_time, 'loss:', loss_epoch / n_batches)
        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')


        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                inputs= inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
                outputs = ae_net(inputs)
                scores = torch.sum((outputs[1] - inputs) ** 2, dim=tuple(range(1, outputs[0].dim())))
                loss = torch.mean(scores)

                if (i + 1) * inputs.shape[1] < len(dataset):
                    idx = [j for j in range(i * inputs.shape[0], (i + 1) * inputs.shape[0])]
                else:
                    idx = [j for j in range(i * inputs.shape[0], len(dataset))]

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx,
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * auc))
        # print('auc_roc:', auc, 'auc_pr:', ap)

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet, model_save_path: str, test_data, score_save_path, rc_save_path):

        best_auc_roc = 0
        best_ap = 0

        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()

        for epoch in range(self.n_epochs):

            net.train()

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _ = data
                inputs=inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

            net.eval()
            #print('epoch', epoch + 1, end=' ')
            auc_roc, ap, scores = self.test(test_data, net, score_save_path)

            if auc_roc > best_auc_roc:

                best_auc_roc = auc_roc
                best_ap = ap
                torch.save(net.state_dict(), model_save_path)
                np.save(score_save_path, scores)

                rc = np.asarray([self.R.detach().cpu().numpy(), self.c.detach().cpu().numpy()])
                np.save(rc_save_path, rc)



        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net, best_auc_roc, best_ap

    def test(self, dataset, net: BaseNet, score_save_path):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2])
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                #outputs = outputs[:, -1, :]
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                if (i + 1) * inputs.shape[1] < len(dataset):
                    idx = [j for j in range(i * inputs.shape[0], (i + 1) * inputs.shape[0])]
                else:
                    idx = [j for j in range(i * inputs.shape[0], len(dataset))]

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx,
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        self.test_ap = average_precision_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test set AP: {:.2f}%'.format(100. * self.test_ap))
       # print('auc_roc:', self.test_auc, 'auc_pr:', self.test_ap)

        logger.info('Finished testing.')

        # np.save(score_save_path, scores)
        return self.test_auc, self.test_ap, scores

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs=inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2])
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                outputs_squeezed = torch.sum(outputs, dim=0)
                outputs_squeezed = torch.squeeze(outputs_squeezed, dim=0)
                c += outputs_squeezed

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, net_name):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, train_data, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, model_save_path: str='', test_data=None, score_save_path='', rc_save_path=''):
        """Trains the Deep SVDD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net, best_auc_roc, best_ap = self.trainer.train(train_data, self.net, model_save_path, test_data, score_save_path, rc_save_path)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time
        return best_auc_roc, best_ap

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0, score_save_path: str='', rc_save_path=''):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net, score_save_path)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        return self.results['test_auc']

    def pretrain(self, dataset: Dataset, test_data: Dataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SVDD network \phi via autoencoder."""

        self.ae_net = build_autoencoder(self.net_name)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(test_data, self.ae_net)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

def DeepSVDD_Perf(Input,Target,dataname,normal_label,p):
    #n_epochs = 50
    #hidden_size = 100
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
    train_dataset = TensorDataset(train_data, train_target)

    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/pths/deepsvdd_{}_{}_{}.pth'.format(dataname, normal_label,p)
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/scores/deepsvdd_{}_{}_{}'.format(dataname,normal_label,p)
    rc_save_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + '/pths/deepsvdd_rc_{}_{}_{}'.format(dataname,normal_label,p)

    cfg={}
    cfg['ae_optimizer_name']='adam'
    cfg['ae_lr']=0.001
    cfg['ae_n_epochs']=30
    cfg['ae_lr_milestone']=(50,)
    cfg['ae_batch_size']=256
    cfg['ae_weight_decay']=0.0005

    cfg['optimizer_name']='adam'
    cfg['lr']=0.001
    cfg['n_epochs']=100
    cfg['lr_milestone']=(50,)
    cfg['batch_size']=128
    cfg['weight_decay']=5e-07

    deep_SVDD=DeepSVDD(objective='one-class',nu=0.1)
    deep_SVDD.set_network(dataname)
    time_start = time.time()
    #Pretrain model on dataset
    deep_SVDD.pretrain(dataset=train_dataset,test_data=train_dataset,
                       optimizer_name=cfg['ae_optimizer_name'],
                       lr=cfg['ae_lr'],
                       n_epochs=cfg['ae_n_epochs'],
                       lr_milestones=cfg['ae_lr_milestone'],
                       batch_size=cfg['ae_batch_size'],
                       weight_decay=cfg['ae_weight_decay'],
                       device=device,
                       n_jobs_dataloader=0)

    #Train model on dataset
    best_auc_roc, best_ap = deep_SVDD.train(train_dataset,
                                            optimizer_name=cfg['optimizer_name'],
                                            lr=cfg['lr'],
                                            n_epochs=cfg['n_epochs'],
                                            lr_milestones=cfg['lr_milestone'],
                                            batch_size=cfg['batch_size'],
                                            weight_decay=cfg['weight_decay'],
                                            device=device,
                                            n_jobs_dataloader=0,
                                            model_save_path=model_save_path,
                                            test_data=train_dataset,
                                            score_save_path=score_save_path,
                                            rc_save_path=rc_save_path)

    time_end = time.time()

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))



if __name__=='__main__':
    dataname='HAR'
    A=np.load('/tmp/FDAD/DesampleData/'+dataname+'.npz',allow_pickle=True)
    A_data,A_label=A['data'],A['label']
    l_list = len(np.unique(A_label[0]))
    print('dataname: {}, number of categories: {}'.format(dataname, l_list))
    for i in range(l_list):
       # for j in range(5):

            p=0.1
            print("Normal_label:--{} P--{}".format(i,p))
            data,label=Load_Polluted_Data(A_data,A_label,i,p=p)

            Input,Target=np.concatenate(data,axis=0),np.concatenate(label,axis=0)
            DeepSVDD_Perf(Input,Target,dataname,i,p)
