import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def CT_loss(P0, P1,N0, N1, margin = 0.3):
    P0 = F.normalize(P0)
    P1 = F.normalize(P1)
    N0 = F.normalize(N0)
    N1 = F.normalize(N1)
    sim_P = torch.clamp(torch.einsum('ik, ik->i ', P0, P1), min = 0)
    sim_N = torch.clamp(torch.einsum('ij,ij->i', N0, N1), min=0)
    loss = torch.einsum('i->', 1-sim_P) + torch.einsum('i->', torch.clamp(sim_N-margin, min=0))
    return loss
    
def CT_loss_no_norm(P0,P1, N0, N1, margin = 0.85):
    sim_P = torch.clamp(torch.einsum('ik, ik->i ', P0, P1), min = 0)
    sim_N = torch.clamp(torch.einsum('ij,ij->i', N0, N1), min=0)
    loss = torch.einsum('i->', 1-sim_P) + torch.einsum('i->', torch.clamp(sim_N-margin, min=0))
    return loss