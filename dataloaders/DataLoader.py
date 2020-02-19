import os
import pdb
import pickle
import numpy as np

from dataloaders import data_loader_review

class DataLoader:
    def __init__(self, opt):
        
        self.dpath = opt.dataset_path + '/'
        self.udict = np.load(self.dpath+'user_dict', allow_pickle=True).item()
        self.idict = np.load(self.dpath+'item_dict', allow_pickle=True).item()
        self.unkid = -1
        self.batch_size = opt.batch_size
        self.input_dim = 10
        opt.num_user, opt.num_item = len(self.udict)+1, len(self.idict)+1 # +1 for handling padding

        self.dl = data_loader_review
        
        rfpath = '/'.join(self.dpath.split('/')[:-2])+'/rf/'
        rf = np.load(rfpath+'rf_{}.npy'.format(self.input_dim))

        self.input_embedding = np.vstack([np.zeros(self.input_dim), rf])   
        
        self.trn_loader, self.vld_loader, self.tst_loader = self.get_loaders_()
        
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("==================================================================================")

    def get_loaders_(self):
        print("Loading data...")
        trn_loader = self.dl.get_loader(self.dpath+'trn', self.udict, self.idict, self.unkid, self.batch_size)
        print('\tTraining data loaded')
        
        vld_loader = self.dl.get_loader(self.dpath+'vld', self.udict, self.idict, self.unkid, self.batch_size, shuffle=False)
        print('\tValidation data loaded')
        
        tst_loader = self.dl.get_loader(self.dpath+'tst', self.udict, self.idict, self.unkid, self.batch_size, shuffle=False)
        print('\tTest data loaded')
        
        return trn_loader, vld_loader, tst_loader
        
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
        
        
        
        
        
        
        
        
        
        
        
        
            
