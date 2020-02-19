import os
import pdb
import time
import math
import copy
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from numpy import average as AVG
from numpy import std as STD

from metric import cal_measures, cal_measures_cs
from torch.autograd import Variable
from dataloaders.DataLoader import DataLoader
from collections import Counter

# from models import MINE, PARL, DATTN, DEEPCONN, SENTIREC, MPCN, NARRE, RMR, HMN, MF, DMN, RCS, EDMN, DSSA, DSSALN, FSA, REBERT, DSHW, SSSA, DSAT, SSSAMS, SSSAMS_ABL, RIGHTMASK, TRANSNET, DAML, SA_ONE, SA_TWO, MF, NCF, FM
from models import REBERT

# torch.set_num_threads(4)

random.seed(2018)
np.random.seed(2018)
torch.manual_seed(2018)

# Arguments
# Model type
# Task type
# Hyperparameters for all models

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
        
# def get_cs_users(model_name, trn_loader, cnt):
#     trn_users = trn_loader.dataset.userids
    
#     # TODO validate they contain equal CS users
#     # RCS and PARL contains same number ov CS users!
    
#     if model_name in ['rcs']:
#         cs_mask = np.array([bool(-1 in ur) for ur in trn_loader.dataset.user_reviews])
#     elif model_name in ['sentirec']: # Not found any cs users
#         cs_mask = np.array([bool(-1 in urs[0]) for urs in trn_loader.dataset.user_reviews])
#     elif model_name in ['parl']:
#         cs_mask = np.array([bool(len(set(ur))==1) for ur in trn_loader.dataset.user_reviews])
    
#     cs_user = np.array(trn_users)[cs_mask]
    
#     print('\nCS (new) user ratio: {}/{}\n'.format(len(cs_user), len(set(trn_users))))
#     return cs_user    
    
#     usercnt = Counter(trn_users)
#     cs_user = [u for u in usercnt if usercnt[u] <= cnt]
# #     cs5_user = [u for u in usercnt if usercnt[u] <= 5]
#     print('\nCS {} user ratio: {}/{}\n'.format(cnt, len(cs_user), len(set(trn_users))))
#     return cs_user
# #     return cs3_user, cs5_user

def get_cs_mask(model_name, loader, udict):
#     # for real-setting data
#     if model_name in ['rcs']:
#         cs_mask = np.array([bool(-1 in ur) for ur in loader.dataset.user_reviews])
#     elif model_name in ['sentirec']: # Not found any cs users
#         cs_mask = np.array([bool(-1 in urs[0]) for urs in loader.dataset.user_reviews])
#     elif model_name in ['parl']:
#         cs_mask = np.array([bool(len(set(ur))==1) for ur in loader.dataset.user_reviews])
#     ucnt = Counter(loader.dataset.userids)
    
    
    masks = []
    for u in loader.dataset.userids:
#         if u in udict:
#             inst = bool(len(udict[u]) <= 0) 
#         else:
#             pdb.set_trace()
#             inst = True

        # 1 to handle padding for REBERT
        if model_name in ['rebert']: u = u-1 # user index is added due to a padding token

        if u not in udict: 
            inst = True
        else:
            inst = False
        masks.append(inst)
        
    cs_mask = np.array(masks)
    
    print('New user index: {}'.format(max(loader.dataset.userids)))
    
#     vldcnt = [len(udict[u]) for u in loader.dataset.userids]
    
    
    
#     cs_mask = np.array([bool(len(udict[u]) <= 3) for u in loader.dataset.userids])
    
#     if model_name in ['parl', 'deepconn', 'dattn', 'narre', 'mpcn']:
#         cs_mask = []
#         for rv in loader.dataset.user_reviews:
#             rvnum = sum([bool(len(set(i))!=1) for i in rv.reshape(20, 30)])
#             cs_mask.append(bool(rvnum <= 5))
#     elif model_name in ['dmn', 'sentirec']:
#         cs_mask = [len(i) <= 5 for i in loader.dataset.user_reviews]
    
#     cs_user = np.array(trn_users)[cs_mask]
    
#     print('CS (new) user ratio: {}/{}'.format(sum(cs_mask), len(cs_mask)))
    return cs_mask

# def get_cs_mask(loader, userset):
#     userids = np.array(loader.dataset.userids)    
#     return np.in1d(userids, userset)

# def get_pseudo_label(self):
#     criterion = nn.MSELoss()
    
#     fixed_trn_loader = self.data_loader.get_fixed_trn_loader()
    
#     vbest = 100000 # a large number
#     best_alloutputs = None
#     batch_loss = 0
    
#     model = MF(self.input_embedding, self.opt).cuda()
    
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
    
#     c = 0
#     for epoch in range(self.opt.num_epoch):
#         st = time.time()
#         for i, batch_data in enumerate(self.trn_loader):
#             batch_only_data = batch_data[:-1] # cuda will be called in models
#             labels = batch_data[-1].float().cuda()

#             optimizer.zero_grad()  # zero the gradient buffer

#             outputs = model(batch_only_data)

#             loss = criterion(outputs, labels)

#             loss.backward()

#             optimizer.step()

#             batch_loss += loss.data.item()
            
#         c+=1
        
#         with torch.no_grad():
#             vperfs  = cal_measures(self.vld_loader, model, self.opt.task)            

#             if vperfs[0] < vbest:
#                 vbest = vperfs[0]
                
#                 # Extract labels for training data
#                 alloutputs = []
#                 for batch_data in fixed_trn_loader:
#                     batch_only_data = batch_data[:-1]
#                     model.eval()
#                     output = model(batch_only_data)
#                     model.train()
#                     alloutputs.append(output)
                
#                 best_alloutputs = alloutputs
                
#             c = 0

#             ept = time.time() - st
            
#             if (epoch+1) % 5 == 0 or epoch == 0: 
#                 print(('(%.1fs)\tEpoch [%d/%d], trn_e : %.4f, vperf0 : %5.4f'% (ept, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), vperfs[0])))
            
#             batch_loss =0
        
#         if c > 5: break
            
#     best_alloutputs = torch.cat(best_alloutputs)
            
#     return best_alloutputs

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        # load data_loaders from a data file
        self.data_loader = DataLoader(self.opt)
        self.trn_loader, self.vld_loader, self.tst_loader = self.data_loader.get_loaders()
        self.input_embedding = self.data_loader.get_embedding()
        
#         pdb.set_trace()
        
#         self.cs_userset = get_cs_users(opt.model_name, self.trn_loader, 3)
        self.trn_cs_mask = get_cs_mask(opt.model_name, self.trn_loader, self.data_loader.udict)
        self.vld_cs_mask = get_cs_mask(opt.model_name, self.vld_loader, self.data_loader.udict)
        self.tst_cs_mask = get_cs_mask(opt.model_name, self.tst_loader, self.data_loader.udict)
        
#         self.trn_cs_mask = get_cs_mask(self.trn_loader, self.cs_userset)
#         self.vld_cs_mask = get_cs_mask(self.vld_loader, self.cs_userset)
#         self.tst_cs_mask = get_cs_mask(self.tst_loader, self.cs_userset)
        
#         print('TRN cs instances: {}/{} ({:.3})'.format(sum(self.trn_cs_mask), len(self.trn_cs_mask), sum(self.trn_cs_mask)/len(self.trn_cs_mask)))
        print('VLD cs instances: {}/{} ({:.3})'.format(sum(self.vld_cs_mask), len(self.vld_cs_mask), sum(self.vld_cs_mask)/len(self.vld_cs_mask)))
        print('TST cs instances: {}/{} ({:.3})'.format(sum(self.tst_cs_mask), len(self.tst_cs_mask), sum(self.tst_cs_mask)/len(self.tst_cs_mask)))
        
#         self.vld_cs3_mask = get_cs_mask(self.vld_loader, self.cs3_userset)

#         self.tst_cs5_mask = get_cs_mask(self.tst_loader, self.cs5_userset)        
        
        # Declare a model
        self.model = self.opt.model_class(self.input_embedding, self.opt).cuda()
        self.initial_weight = copy.deepcopy(self.model.state_dict())
        
#         self._print_args()
        
#         # Generate pseudo labels
#         if opt.model_name == 'distill':
#             pseudo_label = get_pseudo_label(self)
#             pseudo_label = pseudo_label.cpu().data.numpy()
            
#             # load new trn_loader having pseudo labels!
#             trn_loader_with_pseudo = self.data_loader.get_trn_loader_with_pseudo(pseudo_label)
#             self.trn_loader = trn_loader_with_pseudo
        
    def train(self):
        newtime = round(time.time())
        
        criterion = nn.MSELoss()
#         criterion = nn.BCELoss() if self.opt.task =='ranking' else nn.MSELoss()
        
#         if self.opt.model_name == 'transnet':
#             l1loss = nn.L1Loss()
#             l2loss = nn.MSELoss()
            
        
#         if opt.xentropy == True:
#             criterion = nn.CrossEntropyLoss()
            
        if opt.sparse_grad == True:
            # TODO make sure the first one is always input embedding!
            sparse = []
            dense = []
            statedict = self.model.named_parameters()
            
            for name, param in statedict:
                if 'ebd' in name or 'embed' in name or 'mybias' in name: 
                    sparse.append(param)
                    print(name)
                else: 
                    dense.append(param)
                    
#             allparam = [i for i in self.model.parameters()]
#             sparse = [allparam[0]]
#             dense = allparam[1:]
            
            sparse_optimizer = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, sparse),
                                                      lr=self.opt.learning_rate)
        else:
            dense = self.model.parameters() # set all the parameters as dense one
        
        
        
#         if self.opt.optimizer =='adam' and dense != False:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dense),
                                     lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            
#         elif self.opt.optimizer =='radam' and dense != False:
#             optimizer = RAdam(filter(lambda p: p.requires_grad, dense), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            
#             optimizer = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opt.learning_rate)
#         elif self.opt.optimizer =='sgd' and dense != False:
#             optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, dense), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            
        
#         if self.opt.task == 'ranking':
#             vperfs = [0 ,0]
#             tperfs = [0 ,0]
#             best_vperfs = [0, 0]
#         elif self.opt.task == 'rating':
        vperfs = [9999 , 9999]
        tperfs = [9999 ,9999]
        best_vperfs = [9999, 9999]
        best_csvperf = 9999
        
        batch_loss = 0
        
        c = 0
        
        for epoch in range(self.opt.num_epoch):
            
#             if opt.model_name == 'rightmask':
#                 pdb.set_trace()
# #                 user, item, sa = self.model.get_att_from_last_sa(batch_data)
            
            st = time.time()
            for i, batch_data in enumerate(self.trn_loader):

#                 if opt.model_name != 'distill':
                batch_only_data = batch_data[:-1] # cuda will be called in models
                labels = batch_data[-1].float().cuda()
#                 else:
#                 batch_only_data = batch_data[:-2] # cuda will be called in models
#                 labels = batch_data[-2].float().cuda()
#                 plabels = batch_data[-1].float().cuda()
    
                # Forward + Backward + Optimize
                if dense != False: # For MF, FM, NCF
                    optimizer.zero_grad()  # zero the gradient buffer
                
                if opt.sparse_grad == True: 
                    sparse_optimizer.zero_grad()  # zero the gradient buffer
                
                    
#                     pdb.set_trace()

#                     rv_user = batch_only_data[0].cuda()
#                     rv_item = batch_only_data[1].cuda()
#                     batch_size = rv_user.shape[0]
#                     if ((rv_user>0).sum(1) > 0).sum() != batch_size:
#                         pdb.set_trace()
#                     elif ((rv_item>0).sum(1) > 0).sum() != batch_size:
#                         pdb.set_trace()

                outputs, outputs_full = self.model(batch_only_data)

#                     if opt.xentropy == True:
#                         labels = labels.long() - 1
#                         loss = criterion(outputs, labels.long()-1)
#                         loss_full = criterion(outputs_full, labels.long()-1)
#                     else:
                loss = criterion(outputs, labels)
                loss_full = criterion(outputs_full, labels)

                loss = loss * opt.lamb + loss_full * (1-opt.lamb)
#                 else:
# #                     rv_user = batch_only_data[0].cuda()
# #                     rv_item = batch_only_data[1].cuda()
                        
# #                     batch_size = rv_user.shape[0]
# #                     if ((rv_user>0).sum(1) > 0).sum() != batch_size:
# #                         pdb.set_trace()
# #                     elif ((rv_item>0).sum(1) > 0).sum() != batch_size:
# #                         pdb.set_trace()

# #                     pdb.set_trace()
                        
#                     outputs = self.model(batch_only_data)
                    
# #                     if opt.xentropy == True:
# #                         loss = criterion(outputs, labels.long()-1) # CE loss includes softmax
# #                     else:
#                     loss = criterion(outputs, labels) 
# #                     y = torch.eye(5)
# #                     ys = y[labels.long()-1]
# # #                     loss = criterion(outputs, labels)
# #                     ps = y[outputs.max(dim=1)[1]]
                    
# #                     loss = criterion(ps, labels)
                loss.backward()
        
                # Gradient clipping
#                 if opt.model_name in ['rebert']:                    
                max_grad_norm = 1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                if dense != False:
                    optimizer.step()
                
                if opt.sparse_grad == True: 
                    sparse_optimizer.step()
    
                batch_loss += loss.data.item()
        
#                 print('minibatch_loss:\t{}'.format(round(loss.data.item(),3)))
#                 batch_loss += loss.item()
                
            elapsed = time.time() - st

            evalt = time.time()
            
#             pdb.set_trace()
            
            with torch.no_grad():
                self.model.eval()
                
                vperfs  = cal_measures(self.vld_loader, self.model, self.opt.task)
                vld_csperf = cal_measures_cs(self.vld_loader, self.model, 
                                             self.opt.task, self.vld_cs_mask)


                if vperfs[0] < best_vperfs[0]:
                    for k in range(len(vperfs)): best_vperfs[k] = vperfs[k]
                    best_csvperf = vld_csperf
                    
                    tperfs = cal_measures(self.tst_loader, self.model, self.opt.task)
                    tcs_perf = cal_measures_cs(self.tst_loader, self.model, 
                                               self.opt.task, self.tst_cs_mask)
                    c=0
                    
                self.model.train()
                
                evalt = time.time() - evalt 
                
                print(('(%.1fs, %.1fs)\tEpoch [%d/%d], trn_e : %.4f, vperf0 : %5.4f, tperf0 : %5.4f  tcs_p0 : %5.4f'% (elapsed, evalt, epoch, self.opt.num_epoch, batch_loss/len(self.trn_loader), vperfs[0],  tperfs[0], tcs_perf[0])))
    
            
            batch_loss =0
            
            c += 1
            if c > 5: break
        
        print('TST MSE and MAE:\t{}\t{}'.format(tperfs[0],tcs_perf[0]))
        print('VLD MSE and MAE:\t{}\t{}'.format(best_vperfs[0], best_csvperf[0]))
            
        return [best_vperfs[0], best_csvperf[0], tperfs[0], tcs_perf[0]]
        

#     def _print_args(self):
#         n_trainable_params, n_nontrainable_params = 0, 0
#         for p in self.model.parameters():
#             n_params = torch.prod(torch.tensor(p.shape))
#             if p.requires_grad:
# #                 print(n_params)
#                 n_trainable_params += n_params
#             else:
#                 n_nontrainable_params += n_params
#         print('\nn_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
#         print('> training arguments:')
#         for arg in vars(self.opt):
#             print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
#         print('')

    def run(self, repeats):
        results = []
        
        for i in range(repeats):
            print('\nrepeat: {}/{}'.format(i+1, repeats))
            torch.manual_seed(i)
            self.model = self.opt.model_class(self.input_embedding, self.opt).cuda()
#             self._reset_params()
            
            results.append(ins.train())
            
        results = np.array(results)
        
        vmse, vmae = results[:,0], results[:,1]
        tmse, tmae = results[:,2], results[:,3]
        
        print('\n\nSummary')
        print('TST MSE and MAE:\t{}\t{}\t{}\t{}'.format(AVG(tmse),STD(tmse), AVG(tmae),STD(tmae)))
        print('VLD MSE and MAE:\t{}\t{}\t{}\t{}'.format(AVG(vmse),STD(vmse), AVG(vmae),STD(vmae)))
            
#     def _reset_params(self):
#         self.model = self.opt.model_class(self.input_embedding, self.opt).cuda()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mine', type=str)
    parser.add_argument('--dataset', default='office', type=str)    
    parser.add_argument('--datatype', default='real', type=str)
    parser.add_argument('--task', default='rating', type=str)
#     parser.add_argument('--input_types', default='review', type=str)
    # General hyperparameters
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
#     parser.add_argument('--dropout_mask', default=0.5, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float) # previous default: 1e-4 
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)    
#     parser.add_argument('--save', default=False, type=str2bool)

    # Architecture-specific Hyperparameters
#     parser.add_argument('--reviewtype', default='cs', type=str)
#     parser.add_argument('--K', default=1, type=int)
#     parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--num_layer', default=0, type=int)
#     parser.add_argument('--isw', default=True, type=str2bool)
#     parser.add_argument('--isor', default=True, type=str2bool)
#     parser.add_argument('--isbiasdrop', default=False, type=str2bool)
#     parser.add_argument('--ismoreaux', default=True, type=str2bool)
#     parser.add_argument('--mdtype', default='sa', type=str)
#     parser.add_argument('--abltype', default='', type=str)
#     parser.add_argument('--issm', default=True, type=str2bool)
#     parser.add_argument('--isparl', default=False, type=str2bool)
#     parser.add_argument('--isgate', default=True, type=str2bool)
#     parser.add_argument('--gatetype', default='residual', type=str)
#     parser.add_argument('--projtype', default='avg', type=str)
#     parser.add_argument('--inputdrp', default=True, type=str2bool)
#     parser.add_argument('--lastdrp', default=True, type=str2bool)
#     parser.add_argument('--ismask', default=True, type=str2bool)
#     parser.add_argument('--masktype', default='v', type=str) # 'v' or 'm' 
    parser.add_argument('--outtype', default='p', type=str) # 'f' or 'p' 
#     parser.add_argument('--iscnn', default=False, type=str2bool)
#     parser.add_argument('--attdrp', default=False, type=str2bool)
#     parser.add_argument('--penudrp', default=False, type=str2bool)
#     parser.add_argument('--isbias', default=True, type=str2bool)
#     parser.add_argument('--isbn', default=False, type=str2bool) # False for rating prediction
    parser.add_argument('--pool', default='mask_prj', type=str)
#     parser.add_argument('--iscoatt', default=False, type=str2bool)
#     parser.add_argument('--isrelu', default=True, type=str2bool)
#     parser.add_argument('--ismssc', default=False, type=str2bool)
#     parser.add_argument('--isresidual', default=False, type=str2bool)
#     parser.add_argument('--isshare', default=False, type=str2bool)
#     parser.add_argument('--isdotprd', default=False, type=str2bool)
#     parser.add_argument('--input_augment', default=False, type=str2bool)
#     parser.add_argument('--isemb', default=False, type=str2bool)
#     parser.add_argument('--isfm', default=False, type=str2bool)
#     parser.add_argument('--nonlinear', default=False, type=str2bool)
#     parser.add_argument('--freeze', default=False, type=str2bool)
#     parser.add_argument('--savegate', default=False, type=str2bool)
#     parser.add_argument('--auxidx', default='mine', type=str)
#     parser.add_argument('--atttype', default='dot', type=str)
#     parser.add_argument('--embed_dim', default=10, type=int)
#     parser.add_argument('--hidden_dim', default=500, type=int)
#     parser.add_argument('--max_seq_len', default=600, type=int)
#     parser.add_argument('--filter_size', default=3, type=int)
#     parser.add_argument('--filter_num', default=50, type=int)
#     parser.add_argument('--filter_num_global', default=100, type=int)
#     parser.add_argument('--output_dim', default=50, type=int)
#     parser.add_argument('--fm_dim', default=50, type=int)
    parser.add_argument('--lamb', default=1, type=float)
#     parser.add_argument('--num_pointer', default=2, type=int)
    parser.add_argument('--num_head', default=2, type=int)
#     parser.add_argument('--polarities_dim', default=3, type=int)
#     parser.add_argument('--hops', default=3, type=int)
#     parser.add_argument('--device', default=None, type=str)    
#     parser.add_argument('--initializer', default='xavier_uniform_', type=str)
#     parser.add_argument('--optimizer', default='adam', type=str)    
#     parser.add_argument('--xentropy', default=False, type=str2bool)
    parser.add_argument('--biasswap', default=True, type=str2bool)
#     parser.add_argument('--timemask', default=True, type=str2bool)
#     parser.add_argument('--ffratio', default=1, type=float)
#     parser.add_argument('--encdepth', default=1, type=int)    
#     parser.add_argument('--sample_prob', default=0.5, type=float)
#     parser.add_argument('--activation', default='sigmoid', type=str)
    parser.add_argument('--sparse_grad', default=True, type=str2bool)    
    
#     parser.add_argument('--log_step', default=5, type=int)
#     parser.add_argument('--logdir', default='log', type=str)
    opt = parser.parse_args()
    
    #if opt.task == 'rating': opt.batch_size = 128
    #elif opt.task == 'ranking': opt.batch_size = 256
    
    if opt.dataset in ['auto', 'aiv', 'office']:
        opt.batch_size = 32
    
#     if opt.model_name == 'parl': opt.lamb = 0.01
#     elif opt.model_name == 'daml': opt.learning_rate = 1e-5

    if opt.model_name == 'rebert' and opt.lamb == 0: 
        print('Output type is set to FULL')
        opt.outtype = 'f'
        opt.num_layer = 3
        
#     if 'parl' in opt.abltype:
#         opt.isparl = True
        
    model_classes = {     
        'rebert': REBERT
    }
#     dataset_paths = {
#         'office': './data/office/extension',
#         'aiv': './data/office/extension',
#         'twitter': './datasets/acl-14-short-data/train.raw'        
#     }

    dataset_path = './realdata/{}/extension'.format(opt.dataset)

#     opt.csratio = None
#     if opt.datatype == 'ratio':
#         dataset_path = './ratiodata/{}/extension'.format(opt.dataset)
#     elif opt.datatype == 'l1o':
#         dataset_path = './l1odata/{}/extension'.format(opt.dataset)
#     elif opt.datatype == 'real':
#         dataset_path = './realdata/{}/extension'.format(opt.dataset)
#     elif 'cs' in opt.datatype:
#         opt.csratio = opt.datatype[2:]
#         dataset_path = './csrealdata/{}{}/extension'.format(opt.dataset, opt.csratio)
#         dataset_path = './csratiodata/{}{}/extension'.format(opt.dataset, opt.csratio)
        
        
#     dataset_path = './data/{}/extension'.format(opt.dataset)
    
    input_types = { # This one is not fully implemented in this time        
        'rebert': ['review'],
    }
    
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_path = dataset_path
    opt.input_types = input_types[opt.model_name]

    ins = Instructor(opt)
#     ins.train()
    
#     if opt.savegate ==True:
#         print('Running for saving gate weights')
#         ins.run(1)
#     else:
#     if opt.save == True or opt.abltype != '':
# #     opt.dataset in ['health', 'home', 'app', 'kindle', 'elec', 'movie', 'yelp19']: # check trend
#         ins.run(1)
#     else:
        
    ins.run(5)
    
    
    
    
    
    
    
    
    
    
    
    
    
     

        
