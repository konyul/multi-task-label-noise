import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.nn.init as init
from random import *
__all__ = ['pair','symmetry','color_without_noise','rotation_without_noise','rotation_4','color_3','color_6','joint_24','joint_without_noise']


class pair_:
    def __init__(self,batch_data,batch_target,major,location,auxiliary):
        self.batch_data = batch_data
        self.batch_target = batch_target
        self.major = major
        self.location = location
        self.auxiliary = auxiliary
        self.batch = batch_data.shape[0]
        if auxiliary == 'rotation':
            self.output = 4
        elif auxiliary == 'color':
            self.output = 6
        elif auxiliary == 'color_rotation' or auxiliary == 'rotation_color':
            self.output = 24    
    def rotation(self):
        targets_r = torch.randint(0,self.output,(self.batch,))
        targets_r_zero = torch.zeros(self.batch,self.output).scatter(1,torch.zeros(self.batch,).view(-1,1).long(),1)
        targets_rot = torch.zeros(self.batch,self.output).scatter(1,targets_r.view(-1,1).long(),self.major)              # major를  define
        
        targets_r_sub = torch.zeros(targets_r.size())
        
        if self.location == 'next':
            for i in range(len(targets_r)):
                if targets_r[i] == self.output-1:
                    targets_r_sub[i] = 0
                else:
                    targets_r_sub[i] = targets_r[i]+1
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major)   #minor define

        elif self.location == 'previous':
            for i in range(len(targets_r)):
                if targets_r[i] == 0:
                    targets_r_sub[i] = self.output-1
                else:
                    targets_r_sub[i] = targets_r[i]-1
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major)       

        elif self.location == 'neither':
            for i in range(len(targets_r)):
                if targets_r[i] == 0 or targets_r[i] == 1:
                    targets_r_sub[i] = targets_r[i]+2
                else:
                    targets_r_sub[i] = targets_r[i]-2
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major) 
        
        elif self.location == 'stochastic':
            for i in range(len(targets_r)):
                while True:
                    a = randint(0,self.output-1)
                    if targets_r[i] != a:
                        targets_r_sub[i] = a
                        break
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major)    
            


        target_all = targets_rot+targets_rot_sub

        for i in range(self.batch):
            inputs_rot = torch.rot90(self.batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
            augmented_images.append(inputs_rot)
        inputs_r = torch.stack(augmented_images,0)
        size = self.batch_data.shape[1:]
        result_input = torch.stack([self.batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([self.batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()
        targets_rot = torch.stack([targets_r_zero,target_all],dim=1).view(-1,self.output)        # 0.7 0.3 0 0
        targets_rot = targets_rot.cuda()
        return result_input,targets_rot,targets_cls


def pair(batch_data,batch_target,major,location,auxiliary):
    n = batch_data.shape[0]
    augmented_images=[]
    if auxiliary == 'rotation':
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
            augmented_images.append(inputs_rot)
        inputs_r = torch.cat(augmented_images,0)
        size = batch_data.shape[1:]
        result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()
        targets_rot = torch.stack([targets_r_zero,target_all],dim=1).view(-1,4)        # 0.7 0.3 0 0
        targets_rot = targets_rot.cuda()
        return result_input,targets_rot,targets_cls

    elif auxiliary == 'color':
        targets_r = torch.randint(0,6,(n,))
        targets_r_zero = torch.zeros(n,6).scatter(1,torch.zeros(n,).view(-1,1).long(),1)
        targets_col = torch.zeros(n,6).scatter(1,targets_r.view(-1,1).long(),major)              # major를  define
        
        targets_r_sub = torch.zeros(targets_r.size())
        
        if location == 'next':
            for i in range(len(targets_r)):
                if targets_r[i] == 5:
                    targets_r_sub[i] = 0
                else:
                    targets_r_sub[i] = targets_r[i]+1
            targets_col_sub = torch.zeros(n,6).scatter(1,targets_r_sub.view(-1,1).long(),1-major)   #minor define

        elif location == 'previous':
            for i in range(len(targets_r)):
                if targets_r[i] == 0:
                    targets_r_sub[i] = 5
                else:
                    targets_r_sub[i] = targets_r[i]-1
            targets_col_sub = torch.zeros(n,6).scatter(1,targets_r_sub.view(-1,1).long(),1-major)       
        
        elif location == 'stochastic':
            for i in range(len(targets_r)):
                while True:
                    a = randint(0,5)
                    if targets_r[i] != a:
                        targets_r_sub[i] = a
                        break
            targets_col_sub = torch.zeros(n,6).scatter(1,targets_r_sub.view(-1,1).long(),1-major)    
            


        target_all = targets_col+targets_col_sub

        for i in range(n):
            sample_input = torch.stack([batch_data[i],
                              torch.stack([batch_data[i, 0, :, :], batch_data[i, 2, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 0, :, :], batch_data[i, 2, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 2, :, :], batch_data[i, 0, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 0, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 1, :, :], batch_data[i, 0, :, :]], 0)], 0).view(-1, *size)
        
            inputs_col = sample_input(targets_r[i])
            augmented_images.append(inputs_col)
        inputs_r = torch.stack(augmented_images,0)
        size = batch_data.shape[1:]
        result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()
        targets_col = torch.stack([targets_r_zero,target_all],dim=1).view(-1,6)       
        targets_col = targets_col.cuda()
        return result_input,targets_col,targets_cls

    elif auxiliary == 'rotation_color' or auxiliary == 'color_rotation':
        targets_r = torch.randint(0,24,(n,))
        targets_r_zero = torch.zeros(n,24).scatter(1,torch.zeros(n,).view(-1,1).long(),1)
        targets_col = torch.zeros(n,24).scatter(1,targets_r.view(-1,1).long(),major)              # major를  define
        auxiliary_images = []
        size = batch_data.shape[1:]
        targets_r_sub = torch.zeros(targets_r.size())
        if location == 'stochastic':
            for i in range(len(targets_r)):
                while True:
                    a = randint(0,23)
                    if targets_r[i] != a:
                        targets_r_sub[i] = a
                        break
            targets_col_sub = torch.zeros(n,24).scatter(1,targets_r_sub.view(-1,1).long(),1-major)    
        
        target_all = targets_col+targets_col_sub

        for i in range(n):
            augmented_images=[]
            for k in range(4):
                x = torch.rot90(batch_data,k,(2,3))
                augmented_images.append(x[i])
                augmented_images.append(torch.stack([x[i, 0, :, :], x[i, 2, :, :], x[i, 1, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 0, :, :], x[i, 2, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 2, :, :], x[i, 0, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 0, :, :], x[i, 1, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 1, :, :], x[i, 0, :, :]], 0))
            augmented_images = torch.stack(augmented_images, 0).view(-1, *size).contiguous()
            auxiliary_images.append(augmented_images[targets_r[i]])
        inputs_a = torch.cat(auxiliary_images,0)
        
        result_input = torch.stack([batch_data,inputs_a],1).view(-1,*size)
        targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()
        targets_aux = torch.stack([targets_r_zero,target_all],dim=1).view(-1,24)       
        targets_aux = targets_aux.cuda()
        return result_input,targets_aux,targets_cls

def symmetry(batch_data,batch_target,major,location,auxiliary):                   #batch_data,batch_target,major
    n = batch_data.shape[0]
    augmented_images=[]
    if auxiliary == 'rotation':
        targets_r = torch.randint(0,4,(n,))
        targets_r_zero = torch.zeros(n,4).scatter(1,torch.zeros(n,).view(-1,1).long(),1)
        minor =(1-major)/3
        targets_rot = torch.zeros(n,4).scatter(1,targets_r.view(-1,1).long(),major-minor)+minor
        for i in range(n):
            inputs_rot = torch.rot90(batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
            augmented_images.append(inputs_rot)
        inputs_r = torch.cat(augmented_images,0)
        size = batch_data.shape[1:]
        result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()    
        targets_rot = torch.stack([targets_r_zero,targets_rot],dim=1).view(-1,4)           #0.7 0.1 0.1 0.1
        targets_rot = targets_rot.cuda()
        return result_input,targets_rot,targets_cls

    elif auxiliary == 'color':
        targets_r = torch.randint(0,6,(n,))
        targets_r_zero = torch.zeros(n,6).scatter(1,torch.zeros(n,).view(-1,1).long(),1)
        minor =(1-major)/5
        targets_col = torch.zeros(n,6).scatter(1,targets_r.view(-1,1).long(),major-minor)+minor
        for i in range(n):
            sample_input = torch.stack([batch_data[i],
                              torch.stack([batch_data[i, 0, :, :], batch_data[i, 2, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 0, :, :], batch_data[i, 2, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 2, :, :], batch_data[i, 0, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 0, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 1, :, :], batch_data[i, 0, :, :]], 0)], 0).view(-1, *size)
        
            inputs_col = sample_input(targets_r[i])
            augmented_images.append(inputs_col)
        inputs_r = torch.cat(augmented_images,0)
        size = batch_data.shape[1:] 
        result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()    
        targets_col = torch.stack([targets_r_zero,targets_col],dim=1).view(-1,6)           #0.7 0.1 0.1 0.1
        targets_col = targets_col.cuda()
        return result_input,targets_col,targets_cls

    elif auxiliary == 'rotation_color' or auxiliary == 'color_rotation':
        targets_r = torch.randint(0,24,(n,))
        targets_r_zero = torch.zeros(n,24).scatter(1,torch.zeros(n,).view(-1,1).long(),1)
        minor =(1-major)/23
        targets_col = torch.zeros(n,24).scatter(1,targets_r.view(-1,1).long(),major-minor)+minor
        for i in range(n):
            for k in range(4):
                x = torch.rot90(batch_data,k,(2,3))
                augmented_images.append(x[i])
                augmented_images.append(torch.stack([x[i, 0, :, :], x[i, 2, :, :], x[i, 1, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 0, :, :], x[i, 2, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 2, :, :], x[i, 0, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 0, :, :], x[i, 1, :, :]], 0))
                augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 1, :, :], x[i, 0, :, :]], 0))
            augmented_images = torch.stack(augmented_images, 0).view(-1, *size).contiguous()
            auxiliary_images.append(augmented_images[target_r[i]])
        inputs_a = torch.cat(auxiliary_images,0)
        size = batch_data.shape[1:] 
        result_input = torch.stack([batch_data,inputs_a],1).view(-1,*size)
        targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()    
        targets_col = torch.stack([targets_r_zero,targets_col],dim=1).view(-1,24)           #0.7 0.1 0.1 0.1
        targets_col = targets_col.cuda()
        return result_input,targets_col,targets_cls

def color_without_noise(batch_data,batch_target,major,location):
    n = batch_data.shape[0]
    colored_images=[]
    targets_r = torch.randint(0,6,(n,))
    targets_r_zero = torch.zeros(n,)
    targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)
    for i in range(n):
        sample_input = torch.stack([batch_data[i],
                              torch.stack([batch_data[i, 0, :, :], batch_data[i, 2, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 0, :, :], batch_data[i, 2, :, :]], 0),
                              torch.stack([batch_data[i, 1, :, :], batch_data[i, 2, :, :], batch_data[i, 0, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 0, :, :], batch_data[i, 1, :, :]], 0),
                              torch.stack([batch_data[i, 2, :, :], batch_data[i, 1, :, :], batch_data[i, 0, :, :]], 0)], 0).view(-1, *size)
        
        inputs_col = sample_input[targets_r[i]]
        colored_images.append(inputs_col)
    inputs_r = torch.stack(colored_images,0)
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,inputs_r],1).view(-1,*size)
    targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
    targets_cls = targets_cls.cuda()
    targets_col = torch.zeros(2*n,6).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
    targets_col = targets_col.cuda()
    return result_input,targets_col,targets_cls

def rotation_without_noise(batch_data,batch_target,major,location):
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

def joint_without_noise(batch_data,batch_target,major,location,auxiliary):
    
    n = batch_data.shape[0]
    auxiliary_images = []
    augmented_images=[]
    target_r = torch.randint(0,24,(n,))
    targets_r_zero = torch.zeros(n,)
    size = batch_data.shape[1:]
    targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)
    for i in range(n):
        for k in range(4):
            x = torch.rot90(batch_data,k,(2,3))
            augmented_images.append(x[i])
            augmented_images.append(torch.stack([x[i, 0, :, :], x[i, 2, :, :], x[i, 1, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 0, :, :], x[i, 2, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 1, :, :], x[i, 2, :, :], x[i, 0, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 0, :, :], x[i, 1, :, :]], 0))
            augmented_images.append(torch.stack([x[i, 2, :, :], x[i, 1, :, :], x[i, 0, :, :]], 0))
        augmented_images = torch.stack(augmented_images, 0).view(-1, *size).contiguous()
        auxiliary_images.append(augmented_images[target_r[i]])
    inputs_a = torch.stack(auxiliary_images,0)
    result_input = torch.stack([batch_data,inputs_a],1).view(-1,*size)
    targets_cls = torch.stack([batch_target for i in range(2)], 1).view(-1)
    targets_cls = targets_cls.cuda()
    targets_aux = torch.zeros(2*n,24).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
    targets_aux = targets_aux.cuda()
    return result_input,targets_aux,targets_cls


def rotation_4(batch_data,batch_target,major,location):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([torch.rot90(batch_data, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
    target_cls = torch.stack([batch_target for i in range(4)], 1).view(-1)
    target_rot = torch.stack([torch.tensor([0,1,2,3]) for i in range(n)], 0).view(-1)
    target_rot = target_rot.cuda()

    #만약에 Softmax가 더 잘나오면


    return result_input,target_rot,target_cls 

def color_3(batch_data,batch_target,major,location):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,
                              torch.stack([batch_data[:, 1, :, :], batch_data[:, 2, :, :], batch_data[:, 0, :, :]], 1),
                              torch.stack([batch_data[:, 2, :, :], batch_data[:, 0, :, :], batch_data[:, 1, :, :]], 1)], 1).view(-1, *size)
    target_cls = torch.stack([batch_target for i in range(3)], 1).view(-1)
    target_col = torch.stack([torch.tensor([0,1,2]) for i in range(n)], 0).view(-1)
    target_col = target_col.cuda()

    return result_input,target_col,target_cls

def color_6(batch_data,batch_target,major,location):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    result_input = torch.stack([batch_data,
                              torch.stack([batch_data[:, 0, :, :], batch_data[:, 2, :, :], batch_data[:, 1, :, :]], 1),
                              torch.stack([batch_data[:, 1, :, :], batch_data[:, 0, :, :], batch_data[:, 2, :, :]], 1),
                              torch.stack([batch_data[:, 1, :, :], batch_data[:, 2, :, :], batch_data[:, 0, :, :]], 1),
                              torch.stack([batch_data[:, 2, :, :], batch_data[:, 0, :, :], batch_data[:, 1, :, :]], 1),
                              torch.stack([batch_data[:, 2, :, :], batch_data[:, 1, :, :], batch_data[:, 0, :, :]], 1)], 1).view(-1, *size)
    target_cls = torch.stack([batch_target for i in range(6)], 1).view(-1)
    target_col = torch.stack([torch.tensor([0,1,2,3,4,5]) for i in range(n)], 0).view(-1)
    target_col = target_col.cuda()

    return result_input,target_col,target_cls

def joint_24(batch_data,batch_target,major,location,auxiliary):
    n = batch_data.shape[0]
    size = batch_data.shape[1:]
    augmented_images = []
    auxiliary_list = auxiliary.split('_')
    if 'rotation' in auxiliary_list and 'color' in auxiliary_list:
        for k in range(4):
            x = torch.rot90(batch_data,k,(2,3))
            augmented_images.append(x)
            augmented_images.append(torch.stack([x[:, 0, :, :], x[:, 2, :, :], x[:, 1, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 1, :, :], x[:, 0, :, :], x[:, 2, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
            augmented_images.append(torch.stack([x[:, 2, :, :], x[:, 1, :, :], x[:, 0, :, :]], 1))
        result_input = torch.stack(augmented_images, 1).view(-1, *size).contiguous()
        target_cls = torch.stack([batch_target for i in range(24)], 1).view(-1)
        target_auxiliary = torch.stack([torch.tensor(range(24)) for i in range(n)], 0).view(-1)
        target_auxiliary = target_auxiliary.cuda()
    return result_input,target_auxiliary,target_cls

class rotation_:
    def __init__(self,batch_data,batch_target,epoch,major,location,auxiliary,major_function):
        self.batch_data = batch_data
        self.batch_target = batch_target
        if major_function == 'default':
            self.major = major

        elif major_function[location_seperated+1:] == 'increasing':
            if major_function[:location_seperated] == 'linearly':
                self.major = major + (1-major)/epochs*epoch
            elif major_function[:location_seperated] == 'exponentially':
                self.major = math.exp(math.log(major)/epochs*(epochs-epoch))

        elif major_function[location_seperated+1:] == 'decreasing':
            if major_function[:location_seperated] == 'linearly':
                self.major = 1 - (1-major)/epochs*epoch
            elif major_function[:location_seperated] == 'exponentially':
                self.major = 1+major - math.exp(math.log(major)/epochs*(epochs-epoch))
        elif major_function == 'dirichlet':
            dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.tensor([major,major]))
            dirichlet_sample = dirichlet_distribution.sample()
            self.major,small = float(max(dirichlet_sample)),float(min(dirichlet_sample))
        self.location = location
        self.auxiliary = auxiliary
        self.batch = batch_data.shape[0]
        self.output = 4
    def rotation_without_noise(self):
        rotated_images=[]
        targets_r = torch.randint(0,self.output,(self.batch,))
        targets_r_zero = torch.zeros(self.batch,)
        targets_a = torch.stack([targets_r_zero,targets_r],dim=1).view(-1)
        for i in range(self.batch):
            inputs_rot = torch.rot90(self.batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
            rotated_images.append(inputs_rot)
        inputs_r = torch.cat(rotated_images,0)
        size = self.batch_data.shape[1:]
        result_input = torch.stack([self.batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([self.batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()
        targets_rot = torch.zeros(2*self.batch,self.output).scatter(1,targets_a.view(-1,1).long(),1)   #1 0 0 0
        targets_rot = targets_rot.cuda()
        return result_input,targets_rot,targets_cls
    def pair(self):
        augmented_images=[]
        targets_r = torch.randint(0,self.output,(self.batch,))
        targets_r_zero = torch.zeros(self.batch,self.output).scatter(1,torch.zeros(self.batch,).view(-1,1).long(),1)
        targets_rot = torch.zeros(self.batch,self.output).scatter(1,targets_r.view(-1,1).long(),self.major)              # self.major를  define
        
        targets_r_sub = torch.zeros(targets_r.size())
        
        if self.location == 'next':
            for i in range(len(targets_r)):
                if targets_r[i] == self.output-1:
                    targets_r_sub[i] = 0
                else:
                    targets_r_sub[i] = targets_r[i]+1
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major)   #minor define

        elif self.location == 'previous':
            for i in range(len(targets_r)):
                if targets_r[i] == 0:
                    targets_r_sub[i] = self.output-1
                else:
                    targets_r_sub[i] = targets_r[i]-1
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major)       

        elif self.location == 'neither':
            for i in range(len(targets_r)):
                if targets_r[i] == 0 or targets_r[i] == 1:
                    targets_r_sub[i] = targets_r[i]+2
                else:
                    targets_r_sub[i] = targets_r[i]-2
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major) 
        
        elif self.location == 'stochastic':
            for i in range(len(targets_r)):
                while True:
                    a = randint(0,self.output-1)
                    if targets_r[i] != a:
                        targets_r_sub[i] = a
                        break
            targets_rot_sub = torch.zeros(self.batch,self.output).scatter(1,targets_r_sub.view(-1,1).long(),1-self.major)    
            


        target_all = targets_rot+targets_rot_sub

        for i in range(self.batch):
            inputs_rot = torch.rot90(self.batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
            augmented_images.append(inputs_rot)
        inputs_r = torch.cat(augmented_images,0)
        size = self.batch_data.shape[1:]
        result_input = torch.stack([self.batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([self.batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()
        targets_rot = torch.stack([targets_r_zero,target_all],dim=1).view(-1,self.output)        # 0.7 0.3 0 0
        targets_rot = targets_rot.cuda()
        return result_input,targets_rot,targets_cls

    def symmetry(self):
        augmented_images=[]
        targets_r = torch.randint(0,self.output,(self.batch,))
        targets_r_zero = torch.zeros(self.batch,self.output).scatter(1,torch.zeros(self.batch,).view(-1,1).long(),1)
        minor =(1-self.major)/(self.output-1)
        targets_rot = torch.zeros(self.batch,self.output).scatter(1,targets_r.view(-1,1).long(),self.major-minor)+minor
        for i in range(self.batch):
            inputs_rot = torch.rot90(self.batch_data[i],targets_r[i],[1,2]).reshape(1,3,32,32)
            augmented_images.append(inputs_rot)
        inputs_r = torch.cat(augmented_images,0)
        size = self.batch_data.shape[1:]
        result_input = torch.stack([self.batch_data,inputs_r],1).view(-1,*size)
        targets_cls = torch.stack([self.batch_target for i in range(2)], 1).view(-1)
        targets_cls = targets_cls.cuda()    
        targets_rot = torch.stack([targets_r_zero,targets_rot],dim=1).view(-1,self.output)           #0.7 0.1 0.1 0.1
        targets_rot = targets_rot.cuda()
        return result_input,targets_rot,targets_cls
