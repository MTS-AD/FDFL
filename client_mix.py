import torch
import torch.nn as nn
config = {
    "epochs": 20,
    "frac": 1,
    "local_ep": 10,
    # "local_bs": 64,
    "local_bs": 32,
    "prox_mu": 0.01,
    "tau": 0.5,
    "moon_mu": 1,
    "optimizer_fun": lambda parameters: torch.optim.Adam(parameters, lr=0.001)
}


def get_init_grad_correct(model: nn.Module):
    correct = {}
    for name, _ in model.named_parameters():
        correct[name] = torch.tensor(0, dtype=torch.float, device="cpu")
    return correct