import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from numpy import average as AVG
from numpy import std as STD

from metric import cal_rating_measures
from dataloaders.DataLoader import DataLoader

from models import TIMAS

random.seed(2018)
np.random.seed(2018)
torch.manual_seed(2018)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
        
def get_cs_mask(model_name, loader, udict):
    # Get masks for the cold-start (new) user
    
    masks = []
    for u in loader.dataset.userids:

        if model_name in ['rebert']: u = u-1 # user index is added by 1 due to a padding token
            
        if u not in udict: # Users who appear only in test time 
            inst = True
        else:
            inst = False
        masks.append(inst)
        
    return np.array(masks)

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        # Load data loaders
        self.data_loader = DataLoader(self.opt)
        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()
        self.input_embedding = self.data_loader.get_embedding()
        
        self.vld_cs_mask = get_cs_mask(opt.model_name, self.vld_loader, self.data_loader.udict)
        self.tst_cs_mask = get_cs_mask(opt.model_name, self.tst_loader, self.data_loader.udict)

        print('VLD cs instances: {}/{} ({:.3})'.format(sum(self.vld_cs_mask), len(self.vld_cs_mask), sum(self.vld_cs_mask)/len(self.vld_cs_mask)))
        print('TST cs instances: {}/{} ({:.3})'.format(sum(self.tst_cs_mask), len(self.tst_cs_mask), sum(self.tst_cs_mask)/len(self.tst_cs_mask)))
        
        self.model = self.opt.model_class(self.input_embedding, self.opt).cuda()

    def train(self):
        newtime = round(time.time())
        
        criterion = nn.MSELoss()
            
        sparse, dense = [], []
        statedict = self.model.named_parameters()

        for name, param in statedict:
            if 'embed' in name or 'mybias' in name: 
                sparse.append(param)                
            else: 
                dense.append(param)

        sparse_optimizer = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, sparse),
                                                  lr=self.opt.learning_rate)
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dense),
                                     lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            
        vperf = 9999
        tperf = 9999
        best_vperf = 9999
        best_csvperf = 9999
        
        batch_loss = 0
        c = 0        
        for epoch in range(self.opt.num_epoch):
            
            st = time.time()
            for i, batch_data in enumerate(self.trn_loader):

                batch_only_data = batch_data[:-1]
                labels = batch_data[-1].float().cuda()
                
                optimizer.zero_grad()
                sparse_optimizer.zero_grad()  # zero the gradient buffer
                
                outputs, outputs_full = self.model(batch_only_data)

                loss = criterion(outputs, labels)
                loss_full = criterion(outputs_full, labels)

                # Joint learning
                loss = loss * opt.lamb + loss_full * (1-opt.lamb)
                
                loss.backward()
        
                max_grad_norm = 1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                sparse_optimizer.step()
                
                batch_loss += loss.data.item()
                
            elapsed = time.time() - st

            evalt = time.time()
            
            with torch.no_grad():
                self.model.eval()
                
                vperf  = cal_rating_measures(self.vld_loader, self.model)
                vld_csperf = cal_rating_measures(self.vld_loader, self.model, self.vld_cs_mask)


                if vperf < best_vperf:
                    best_vperf = vperf                    
                    best_csvperf = vld_csperf
                    
                    tperf = cal_rating_measures(self.tst_loader, self.model)
                    tcs_perf = cal_rating_measures(self.tst_loader, self.model, self.tst_cs_mask)
                    c=0
                    
                self.model.train()
                
                evalt = time.time() - evalt 
                
                print(('(%.1fs, %.1fs)\tEpoch [%d/%d], trn_e : %.4f, vld_e : %5.4f, tst_e : %5.4f  tst_NU_e : %5.4f'% (elapsed, evalt, epoch+1, self.opt.num_epoch, batch_loss/len(self.trn_loader), vperf,  tperf, tcs_perf)))
    
            
            batch_loss = 0
            
            c += 1
            if c > 5: break
        
        print('TST MSE and NU_E:\t{}\t{}'.format(tperf, tcs_perf))
        print('VLD MSE and NU_E:\t{}\t{}'.format(best_vperf, best_csvperf))
            
        return [best_vperf, best_csvperf, tperf, tcs_perf]
        

    def run(self, repeats):
        results = []
        for i in range(repeats):
            print('\nrepeat: {}/{}'.format(i+1, repeats))
            torch.manual_seed(i)
            self._reset_params()
            
            results.append(ins.train())
            
        results = np.array(results)
        
        vmse, vnue = results[:,0], results[:,1]
        tmse, tnue = results[:,2], results[:,3]
        
        print('\n\nSummary')
        print('TST MSE and NU_E:\t{}\t({})\t{}\t({})'.
              format(AVG(tmse),STD(tmse), AVG(tnue),STD(tnue)))
        print('VLD MSE and NU_E:\t{}\t({})\t{}\t({})'.
              format(AVG(vmse),STD(vmse), AVG(vnue),STD(vnue)))
            
    def _reset_params(self):
        self.model = self.opt.model_class(self.input_embedding, self.opt).cuda()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='timas', type=str)
    parser.add_argument('--dataset', default='auto', type=str)    
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)    
    parser.add_argument('--num_layer', default=3, type=int)
    parser.add_argument('--lamb', default=0.9, type=float)
    opt = parser.parse_args()
    
    if opt.dataset in ['auto', 'aiv', 'office']:
        opt.batch_size = 32
    
    model_classes = {     
        'timas': TIMAS
    }

    dataset_path = './realdata/{}/extension'.format(opt.dataset)

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_path = dataset_path

    ins = Instructor(opt)
        
    ins.run(5)
    
    
    
    
    
    
    
    
    
    
    
    
    
     

        
