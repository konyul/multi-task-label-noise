import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn.init as init
from random import *
__all__ = ['original','rotation_4','pair','symmetry']

def pair(batch_data,batch_target,major,location):
    n = batch_data.shape[0]
    rotated_images=[]
    if major == 'dirichlet':
        dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.tensor([major,major]))
        dirichlet_sample = dirichlet_distribution.sample()
        major,small = float(max(dirichlet_distribution)),float(min(dirichlet_distribution))
    targets_r = torch.randint(0,4,(n,))
    targets_r_zero = torch.zeros(n,4).scatter(1,torch.zeros(n,).view(-1,1).long(),1)
    targets_rot = torch.zeros(n,4).scatter(1,targets_r.view(-1,1).long(),major)              # major를  define
    
    targets_r_sub = torch.zeros(targets_r.size())
    
    if location == 'next':
        for i in range(len(targets_r)):
            if targets_r[i] == 3:
                targets_r_sub[i] = 0
            else:
                targets_r_sub[i] = targets_r[i]+1
        targets_rot_sub = torch.zeros(n,4).scatter(1,targets_r_sub.view(-1,1).long(),1-major)   #minor define

    elif location == 'previous':
        for i in range(len(targets_r)):
            if targets_r[i] == 0:
                targets_r_sub[i] = 3
            else:
                targets_r_sub[i] = targets_r[i]-1
        targets_rot_sub = torch.zeros(n,4).scatter(1,targets_r_sub.view(-1,1).long(),1-major)       

    elif location == 'neither':
        for i in range(len(targets_r)):
            if targets_r[i] == 0 or targets_r[i] == 1:
                targets_r_sub[i] = targets_r[i]+2
            else:
                targets_r_sub[i] = targets_r[i]-2
        targets_rot_sub = torch.zeros(n,4).scatter(1,targets_r_sub.view(-1,1).long(),1-major) 
    
    elif location == 'stochastic':
        for i in range(len(targets_r)):
            while True:
                a = randint(0,3)
                if targets_r[i] != a:
                    targets_r_sub[i] = a
                    break
        targets_rot_sub = torch.zeros(n,4).scatter(1,targets_r_sub.view(-1,1).long(),1-major)    
        


    target_all = targets_rot+targets_rot_sub

    for i in range(n):
        inputs_rot = torch.rot90(batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
        rotated_images.append(inputs_rot)
    inputs_r = torch.cat(rotated_images,0)
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
    targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
    targets_cls = targets_cls.cuda()
    targets_rot = torch.stack([targets_r_zero,target_all],dim=1).view(-1,4)        # 0.7 0.3 0 0
    targets_rot = targets_rot.cuda()
    return result_input,targets_rot,targets_cls

def symmetry(batch_data,batch_target,major,location):                   #batch_data,batch_target,major
    n = batch_data.shape[0]
    rotated_images=[]
    targets_r = torch.randint(0,4,(n,))
    targets_r_zero = torch.zeros(n,4).scatter(1,torch.zeros(n,).view(-1,1).long(),1)
    minor =(1-major)/3
    targets_rot = torch.zeros(n,4).scatter(1,targets_r.view(-1,1).long(),major-minor)+minor
    for i in range(n):
        inputs_rot = torch.rot90(batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
        rotated_images.append(inputs_rot)
    inputs_r = torch.cat(rotated_images,0)
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
    targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
    targets_cls = targets_cls.cuda()    
    targets_rot = torch.stack([targets_r_zero,targets_rot],dim=1).view(-1,4)           #0.7 0.1 0.1 0.1
    targets_rot = targets_rot.cuda()
    return result_input,targets_rot,targets_cls

def original(batch_data,batch_target,major,location):
    n = batch_data.shape[0]
    rotated_images=[]
    targets_r = torch.randint(0,4,(n,))
    targets_r_zero = torch.zeros(n,)
    targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)
    for i in range(n):
        inputs_rot = torch.rot90(batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
        rotated_images.append(inputs_rot)
    inputs_r = torch.cat(rotated_images,0)
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
    targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
    targets_cls = targets_cls.cuda()
    targets_rot = torch.zeros(2*n,4).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
    targets_rot = targets_rot.cuda()
    return result_input,targets_rot,targets_cls

def rotation_4(batch_data,batch_target,major,location):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([torch.rot90(batch_data, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
    target_cls = torch.stack([batch_target for i in range(4)], 1).view(-1)
    target_rot = torch.stack([torch.tensor([0,1,2,3]) for i in range(n)], 0).view(-1)
    target_rot = target_rot.cuda()

    return result_input,target_rot,target_cls 





