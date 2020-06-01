import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nan_parameters(model):
    return sum(torch.isnan(p).sum() for p in model.parameters() if p.requires_grad)
