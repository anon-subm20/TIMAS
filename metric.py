import pdb
import torch
import random
import numpy as np

from sklearn.metrics import mean_squared_error as MSE

def cal_rating_measures(loader, model, mask=None):
    all_output, all_label = [], []
    
    for i, batch_data in enumerate(loader):
        batch_only_data = batch_data[:-1]
        labels = batch_data[-1].cuda()

        outputs = model(batch_only_data)[0]
        
        outputs = outputs.cpu().data.numpy()
        labels = labels.cpu().data.numpy()

        all_output.append(outputs)
        all_label.append(labels)
    
    all_output = np.concatenate(all_output)
    all_label = np.concatenate(all_label)
    
    # Computing performance for new users
    if mask is not None: 
        all_label = all_label[mask]
        all_output = all_output[mask]
    
    mse = MSE(all_label, all_output)

    return mse
        
        
        
        
        
