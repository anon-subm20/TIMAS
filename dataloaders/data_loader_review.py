import time
import random
import pickle
import numpy as np

from torch.utils import data
from torch.utils.data.dataloader import default_collate

random.seed(2019)

class ReviewDataset(data.Dataset):
    
    def __init__(self, path, udict, idict, unkid):
        st = time.time()
        
        dpath = '/'.join(path.split('/')[:-1]) 
        if dpath[-1] != '/': dpath += '/'
        dtype = path.split('/')[-1]
        self.dpath = dpath
        
        data = pickle.load(open(dpath+'{}_timas'.format(dtype), 'rb'))
        
        self.user_reviews, self.item_reviews, self.ur_aux, self.ir_aux, self.ur_mask, self.ir_mask, self.userids, self.itemids, self.ratings = data
        
        print('Data loading time : %.1fs' % (time.time()-st))
        
    def __getitem__(self, index):
        
        return self.user_reviews[index], self.item_reviews[index], self.ur_aux[index], self.ir_aux[index], self.ur_mask[index], self.ir_mask[index], self.userids[index], self.itemids[index], self.ratings[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        return len(self.userids)
    
def my_collate(batch):
    batch = [i for i in filter(lambda x:x is not None, batch)]
    
    return default_collate(batch) 

def get_loader(data_path, udict, idict, unkid, batch_size, shuffle=True, num_workers=0):
    """Builds and returns Dataloader."""

    dataset = ReviewDataset(data_path, udict, idict, unkid)

    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate)

    return data_loader
