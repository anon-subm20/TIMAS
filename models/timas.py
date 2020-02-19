import pdb
import numpy as np

import torch
import torch.nn as nn

from .utils.transformer import TransformerBlock

class TIMAS(nn.Module):
    def __init__(self, review_emb, opt):
        super().__init__()
        self.hidden = 10
        self.attn_heads = 1 
        
        self.n_layers = opt.num_layer        
        self.dropout = opt.dropout
        
        self.new_user_id = opt.num_user 
        self.new_item_id = opt.num_item
        
        self.num_reviews, self.ebd_size = review_emb.shape 
        
        self.review_embed = nn.Embedding.from_pretrained(torch.FloatTensor(review_emb), padding_idx=0, freeze=False, sparse=True).cuda()
        self.review_embed.weight.data[0] = 0
        
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, 
                              self.hidden, self.dropout) 
             for _ in range(self.n_layers)])
        
        self.review_projector = nn.Linear((20 + 1 + 30)*self.hidden, self.hidden)
        self.proj = nn.Linear(self.hidden, 1)
        
        # Initialize user and item biases as zero 
        self.user_mybias = nn.Embedding(opt.num_user+1, 1, sparse=True)      
        torch.nn.init.normal_(self.user_mybias.weight, mean=0.0, std=.02)
        self.item_mybias = nn.Embedding(opt.num_item+1, 1, sparse=True)
        torch.nn.init.normal_(self.item_mybias.weight, mean=0.0, std=.02)

        # Separation token
        self.SEP_idx = torch.zeros(1).long().cuda()
        self.SEP_embed = nn.Embedding(1, self.hidden, sparse=True)
        self.mask_token = torch.zeros(1).float().cuda()
    
        self.drplayer = nn.Dropout(self.dropout)
        
    def forward(self, batch_data):        
        # Prepare inputs in the form of ID
        rv_user, rv_item = batch_data[0].cuda(), batch_data[1].cuda()
        # Time-variant masks
        rm_user, rm_item = batch_data[4].cuda().float(), batch_data[5].cuda().float() 
        uid, iid = batch_data[6].cuda(), batch_data[7].cuda()
        uid_full, iid_full = uid.clone(), iid.clone()
        
        # Time-invariant masks (Conventional approach)
        fullmask_user = (rv_user!=0).float() 
        fullmask_item = (rv_item!=0).float()
        
        x_user = self.drplayer(self.review_embed(rv_user.long()))
        x_item = self.drplayer(self.review_embed(rv_item.long()))
        
        batch_size = batch_data[0].shape[0]                
        SEPs = self.SEP_embed(self.SEP_idx)
        SEPs = SEPs[None, :, :].repeat(batch_size,1, 1) # BN x 1 x ebd_size
        
        x = torch.cat([x_user, SEPs, x_item], dim=1)
        x_full = x 
        
        m_token = self.mask_token[None, :].repeat(batch_size,1)
        
        mask = torch.cat([rm_user, m_token, rm_item], dim=-1)
        fullmask = torch.cat([fullmask_user, m_token, fullmask_item], dim=-1)

        # Reformat the masks to being applied to the self-attention map
        mask_ext = mask[:, None, None, :].repeat(1,1, mask.size(1),1)  
        fullmask_ext = fullmask[:, None, None, :].repeat(1,1, fullmask.size(1),1)  
        
        # Transformer layers
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask_ext)
            
        for transformer in self.transformer_blocks:
            x_full = transformer.forward(x_full, fullmask_ext)

        maskout_x = x * mask[:,:,None]
        maskout_x_full = x_full * fullmask[:,:,None]

        feature = self.review_projector(maskout_x.reshape(batch_size, -1))
        feature_full = self.review_projector(maskout_x_full.reshape(batch_size, -1))   
        
        # Bias Switching
        new_user_idx = rm_user.sum(dim=1) == 0
        new_item_idx = rm_item.sum(dim=1) == 0

        uid[new_user_idx] = self.new_user_id
        iid[new_item_idx] = self.new_item_id
            
        user_bias = self.user_mybias(uid.long()).view(uid.shape[0], -1)
        item_bias = self.item_mybias(iid.long()).view(iid.shape[0], -1)
        user_bias_full = self.user_mybias(uid_full.long()).view(uid_full.shape[0], -1)
        item_bias_full = self.item_mybias(iid_full.long()).view(iid_full.shape[0], -1)

        # Projection
        score = self.proj(feature) + user_bias + item_bias
        score_full = self.proj(feature_full) + user_bias_full + item_bias_full
        
        return score.view(score.shape[0]), score_full.view(score_full.shape[0])
            
    
    
    
    
    
    
    
    
    
    

    
    