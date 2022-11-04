import torch
import torch.nn.functional as F

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)
  

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
    
def change_accuracy(P0, P1, N0, N1):
    with torch.no_grad():
        P0 = F.normalize(P0)
        P1 = F.normalize(P1)
        N0 = F.normalize(N0)
        N1 = F.normalize(N1)
        sim_P = torch.clamp(torch.einsum('ik, ik->i ', P0, P1), min = 0)
        sim_N = torch.clamp(torch.einsum('ij,ij->i', N0, N1), min=0)
        accuracy = torch.einsum('i->', torch.clamp(sim_P,max =1)) + torch.einsum('i->', 1-torch.clamp(sim_N, min=0))
        accuracy = accuracy / P0.shape[0]
    return accuracy
    
def change_accuracy_no_norm(P0, P1, N0, N1):
    with torch.no_grad():
        sim_P = torch.clamp(torch.einsum('ik, ik->i ', P0, P1), min = 0)
        sim_N = torch.clamp(torch.einsum('ij,ij->i', N0, N1), min=0)
        accuracy = torch.einsum('i->', torch.clamp(sim_P,max =1)) + torch.einsum('i->', 1-torch.clamp(sim_N, min=0))
        accuracy = accuracy / P0.shape[0]
    return accuracy
        
