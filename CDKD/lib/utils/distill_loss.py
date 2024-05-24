import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import fvcore.nn.weight_init as weight_init
from config import cfg




class DistillLoss(nn.Module):
    def __init__(self,image_size,joints, distill_attn_param):
        super(DistillLoss, self).__init__()
        # self.criterion_l1 = L1()
        # self.criterion_l2 = L1()
        self.distill_attn_param =distill_attn_param

        self.joints = int(joints)
        
        self.low_size = int(image_size/4.0)
        
        self.high_size = int(256/4.0)
        self.scale = int(self.high_size /self.low_size)
        self.linear_1 = nn.Linear(self.joints*self.low_size*self.low_size, self.joints*self.high_size*self.high_size)
        init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(self.joints*self.low_size*self.low_size, self.joints*self.high_size*self.high_size)
        init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(self.joints*self.low_size*self.low_size, self.joints*self.high_size*self.high_size)
        init.xavier_uniform_(self.linear_3.weight)
        self.relu_1 = torch.nn.ReLU()
        self.relu_2 = torch.nn.ReLU()
        self.relu_3 = torch.nn.ReLU()
        self.relu = torch.nn.ReLU()




        self.convs = nn.ModuleList([])
       
        
        for i in range(3):
            self.convs.append(nn.Sequential(
                nn.Conv2d(self.joints,self.joints, kernel_size=3 + i * 2, stride=1, padding=1 + i),
                nn.BatchNorm2d(self.joints),
                nn.ReLU(inplace=False)
            ))

        self.gap = nn.AvgPool2d(int(self.low_size / 1))
        self.fc = nn.Linear(self.joints, 32)
        
        self.fcs = nn.ModuleList([])
        
        for i in range(4):
            self.fcs.append(
                nn.Linear(32, self.joints)
            )
        self.softmax = nn.Softmax(dim=1)



    
    def forward(self, low_feature, high_feature,temp):
        l = low_feature[0].detach()
        h = high_feature[0].detach()
        

        batch_size,num_joint,hight,width = l.shape

        for i, conv in enumerate(self.convs):
            if i < 3 :
                fea = conv(l.cuda()).unsqueeze(dim=1)

            if i == 0:
                feas = fea
            else:
                
                feas = torch.cat([feas, fea], dim=1)

    
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        fea_z = self.fc(fea_s)

  
        for i, fc in enumerate(self.fcs):
           
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
      
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        l  = l.view(batch_size,-1).cuda()
        l1 = self.relu_1(self.linear_1(l))
        l2 = self.relu_2(self.linear_2(l))
        l3 = self.relu_3(self.linear_3(l))

        l1 = l1.reshape(batch_size,1,self.joints,64,64)
        l2 = l2.reshape(batch_size,1,self.joints,64,64)
        l3 = l3.reshape(batch_size,1,self.joints,64,64)
        l_avg = (l1+l2+l3)/3.0

        ls= torch.cat([l1.cuda(),l2.cuda(),l3.cuda(),l_avg.cuda()], dim=1)
        l_avg = (ls * attention_vectors).sum(dim=1)

        l_avg = l_avg.reshape(batch_size,-1)


        normft = l_avg.pow(2).sum(1, keepdim=True).pow(1. / 2)
        outft = l_avg.div(normft) 
        h =h.view(batch_size,-1)           
        normfs = h.pow(2).sum(1, keepdim=True).pow(1. / 2)
        outfs = h.div(normfs)
            
        cos_theta = (outft.cuda() * outfs.cuda()).sum(1, keepdim=True)
        G_diff = 1 - cos_theta
        loss_line = (G_diff).sum() / batch_size 


        low_feature = low_feature[1:]
        high_feature = high_feature[1:]

        loss = 0
        batch__size = high_feature[0].shape[0]
        joints__number = high_feature[0].shape[1]
        class__number = high_feature[0].shape[2]
        
        T = temp
        for index, (l, h) in enumerate(zip(low_feature, high_feature)):              
            h = nn.Softmax(dim=2)(h/T)
            
            h = h.view(batch__size,joints__number,int(class__number/self.scale),self.scale)
            h = h.sum(dim=3)
            h = h.view(-1,int(class__number/self.scale))
            l = l.view(-1,int(class__number/self.scale))
            
            loss +=  torch.nn.KLDivLoss(reduction='none')(nn.LogSoftmax(dim=1)(l/T), h.detach()).mean(1) * (T * T)
        loss =loss.sum()    
        loss = self.distill_attn_param * loss
    

        loss_line = loss_line * self.distill_attn_param

        return   loss_line, loss




