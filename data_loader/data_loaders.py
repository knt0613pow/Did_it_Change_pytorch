from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
import pickle
import os
import numpy as np
from torchvision.io import read_image
import torch.nn.functional as F
from PIL import Image

class PairDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, key_list, valid_score, first_session, second_session, resize):
    self.data_dir = data_dir
    self.key_list = key_list
    self.valid_score=  valid_score
    self.FS = first_session
    self.SS = second_session
    self.CSR_valid_score = self.valid_score.tocsr()
    self.resize = resize

  def __len__(self):
    return len(self.valid_score.data)

  def __getitem__(self, idx):
    P_t0 = self.valid_score.row[idx]
    P_t1 = self.valid_score.col[idx]  
    #print(self.FS[P_t0])
    #print(self.SS[P_t1])
    P_t0 = os.path.join(self.data_dir, self.key_list[self.FS[P_t0]])
    P_t1 = os.path.join(self.data_dir, self.key_list[self.SS[P_t1]])
    
    t0 = read_image(P_t0)
    t1 = read_image(P_t1)

    P_pair = {'t0' : t0, 't1': t1}
    while True:
      row = np.random.randint(len(self.FS))
      col = np.random.randint(len(self.SS))
      if(self.CSR_valid_score[row, col]==0): break
    #print(self.FS[row])
    #print(self.SS[col])
    N_t0 = os.path.join(self.data_dir, self.key_list[self.FS[row]])
    N_t1 = os.path.join(self.data_dir, self.key_list[self.SS[col]])
    
    N_t0 = read_image(N_t0)
    N_t1 = read_image(N_t1)
    N_pair = {'t0' : N_t0, 't1': N_t1}
    
    return P_pair, N_pair



class PairDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir,eval_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, resize = 1):
        self.data_dir = data_dir
        score_path = os.path.join(eval_dir, 'valid_score')
        KEY_path = os.path.join(eval_dir, 'key_list.npy')
        session_idx_path = os.path.join(eval_dir, 'session_indices.npy')
        with open(score_path, 'rb') as fr:
            valid_score = pickle.load(fr)
        key_list = np.load(KEY_path)
        session_index = np.load(session_idx_path)
        FS = [i for i,s in enumerate(session_index) if s==0]
        SS = [i for i,s in enumerate(session_index) if s!=0]
        self.dataset = PairDataset(self.data_dir, key_list, valid_score, FS, SS, resize)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


