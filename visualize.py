from cirtorch.networks.imageretrievalnet import init_network
import torch
from torchvision.io import read_image
import torchvision
from utils.util import visualize_activation
import os

def visual(img_path, resume_path):
    device = torch.device('cuda')
    img = read_image(img_path).to(device)
    img_load = [img]
    model_params = {}
    model_params['architecture'] = 'resnet101'
    model_params['pooling'] = 'gem'
    model_params['local_whitening'] = True
    model_params['regional'] = True
    model_params['whitening'] = True
    model_params['pretrained'] = True
    model = init_network(model_params)
    
    
    resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    activation = visualize_activation(model, img.unsqueeze(0).float())

    torchvision.utils.save_image(activation, 'test4.jpg')
    
    

if __name__ == '__main__':
    img_path = "semi-dataset/K-004.jpg"
    resume_path = "saved/models/DIT_resize/0823_184149/model_best.pth"
    
    visual(img_path,resume_path )