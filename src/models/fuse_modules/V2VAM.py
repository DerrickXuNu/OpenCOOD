import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding



class V2V_AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(V2V_AttFusion, self).__init__()
        

        self.cov_att = nn.Sequential(
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )

        self.CCNet = CrissCrossAttention(feature_dim)
    def forward(self, x, record_len):

        split_x = self.regroup(x, record_len)  #x =[5, 64, 100, 352], record_len=[3,2]

        out = []
        att = []
        for xx in split_x:#split_x[0] [num_car, C, W, H]

            ''' CCNet: Criss-Cross Attention Module: attention for ego vehicle feature + cav feature '''

            ego_q, ego_k, ego_v = xx[0:1], xx[0:1], xx[0:1] 
            for i in range(len(xx[:,0,0,0])):
                att_vehicle = self.CCNet(ego_q, xx[i:i+1], xx[i:i+1])
                att.append(att_vehicle)

            pooling_max = torch.max(torch.cat(att, dim=0), dim=0, keepdim=True)[0]
            pooling_ave = torch.mean(torch.cat(att, dim=0), dim=0, keepdim=True)[0]

            fuse_fea = pooling_max + pooling_ave

            fuse_att = fuse_fea
            fuse_att = self.cov_att(fuse_att)

            out.append(fuse_att) #[[1, 64, 100, 352], [1, 64, 100, 352]]
            # torch.cuda.empty_cache()

        return torch.cat(out, dim=0) #[2, 64, 100, 352]


    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())

        return split_x



def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module

    reference: https://github.com/speedinghzl/CCNet
    
    """
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()


        self.query_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )
        self.key_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )
        self.value_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )


        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, query, key, value):
        m_batchsize, _, height, width = query.size()

        
        proj_query = self.query_conv(query)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        
        proj_key = self.key_conv(key)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        
        proj_value = self.value_conv(value)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)


        return self.gamma*(out_H + out_W) + value

