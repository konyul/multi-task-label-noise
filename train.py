import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import prepare_for_dataset as pf
import resnet
import argparse
import os
import time
import rotation_loss
import torch.nn.functional as F
#from pdb import set_trace as bp
import math

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")                  #소문자,resnet으로 시작 하는것으로 model_names에 넣음
                     and callable(resnet.__dict__[name]))
  #CUDA_VISIBLE_DEVICES=0,1,2,3 python self_rotation.py --arch resnetself --save-dir ./save_dir/
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')           
parser.add_argument('--save-dir', dest='save_dir',                                       #dest : argument가 저장되는 곳을 지정 args.save_dir에 a가 저장됨 (--save -dir a라고 썻을때)
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--range-of-lr', nargs='+', type=int,help='1234 1234 1234',default=[100,150])       #--range-of-lr 1234 1234 1234
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')        #체크포인트에서 load할때 필요 



parser.add_argument('--dataset',required=True,default=False)           #cifar10,cifar100
parser.add_argument('--arch', '-a',required=True, metavar='ARCH', default='resnetself',           #metavar : description
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnetself)')
parser.add_argument('--auxiliary',required=True,default=False) #rotation color exemplar      if joint, rotation_color
parser.add_argument('--augmentation',required=True,type=int,default=False) # 4 or 2 in rotation, 3 6 2 in color permutation
parser.add_argument('--noise',required=True,action='store_true') # store_true

parser.add_argument('--noise-type',required=True,default=False) # pair symmetry 
parser.add_argument('--major-function',required=True,default=False) # default linearly_increasing linearly_decreasing exponentially_increasing exponentially_decreasing dirichlet
parser.add_argument('--location',required=True,default=False)   #previous,next,neither stochastic   pair에만 해당
parser.add_argument('--major',required=True,default=False) # float     in linear or exponential case, where to start where to end
parser.add_argument('--loss',default='mseloss') # softmax or mseloss




best_prec1 = 0
args = parser.parse_args()
def main():
    
    global args,best_prec1
    
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):     #./save_dir/
        os.makedirs(args.save_dir)
    try:
        args.major = float(args.major)
    except:
        pass

    
    log_directory = [args.dataset,args.arch,args.auxiliary]
    while False in log_directory:
        log_directory.remove(False)
    for i in range(len(log_directory)):
        log_directory[i] = str(log_directory[i])
    log_directory = '/'.join(log_directory)    

    if args.auxiliary == 'rotation':
        log_rotation_directory = [args.augmentation,args.noise,args.noise_type,args.location,args.major_function,args.major]
        while False in log_rotation_directory:
            log_rotation_directory.remove(False)
        for i in range(len(log_rotation_directory)):
            log_rotation_directory[i] = str(log_rotation_directory[i])
        log_rotation_directory = '/'.join(log_rotation_directory)    
    
    if args.auxiliary == 'color':
        log_rotation_directory = [args.augmentation,args.noise,args.noise_type,args.location,args.major_function,args.major]
        while False in log_rotation_directory:
            log_rotation_directory.remove(False)
        for i in range(len(log_rotation_directory)):
            log_rotation_directory[i] = str(log_rotation_directory[i])
        log_rotation_directory = '/'.join(log_rotation_directory)
    



    log_all_directory = './logs/'+log_directory +'/'+ log_rotation_directory+'/'


    start_time = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time()))
    if not os.path.exists(log_all_directory):
        os.makedirs(log_all_directory)
    file = open(log_all_directory+start_time+'.txt','w')
    
    # code_name = [args.major,args.location,args.major_type,args.dataset,args.arch,args.rotation_var]
    # while False in code_name:
    #     code_name.remove(False)
    # for i in range(len(code_name)):
    #     code_name[i] = str(code_name[i])
    # code_name = '-'.join(code_name)
    # file = open('./log_file2/'+code_name+'.txt','w')

    file.write('loss type : {0}\n'.format(args.loss)) 
    file.write('architecture: {0}\n'
                'total epochs: {1}\n'
                'batch size: {2}\n'
                'start learning rate: {3}\n'
                'range of learning rate: {4}\n'
                'dataset: {5}\n'
                'auxiliary type: {6}\n'
                'number of augmentation: {7}\n'
                'noise or not: {8}\n'
                'noise type: {9}\n'
                'major function: {10}\n'
                'location: {11}\n'
                'major: {12}\n'.format(
                    args.arch,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.range_of_lr,
                    args.dataset,
                    args.auxiliary,
                    args.augmentation,
                    args.noise,
                    args.noise_type,
                    args.major_function,
                    args.location,
                    args.major
                ))
    file.close()


    
    train_loader,val_loader = pf.__dict__[args.dataset](batch=args.batch_size)
    

    model = nn.DataParallel(resnet.__dict__[args.arch](num_classes=int(args.dataset[5:])))
    
    model.cuda()
    
    
    if args.resume:
        if os.path.isfile(args.resume):          #args.resume: 체크포인트 path
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.range_of_lr, gamma=0.1, last_epoch=-1)
    
    for epoch in range(args.start_epoch, args.epochs):

        
        
        
        train(train_loader, model, criterion, optimizer,epoch,auxiliary=args.auxiliary,augmentation=args.augmentation,noise = args.noise,noise_type = args.noise_type,location = args.location,major_function=args.major_function,major=args.major,start_time=start_time,log_all_directory=log_all_directory)
        scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, optimizer,criterion,auxiliary=args.auxiliary,start_time=start_time,log_all_directory=log_all_directory)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.pth'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.pth'))


    
    if args.evaluate:
        validate(val_loader, model, optimizer,criterion,auxiliary=args.auxiliary,start_time=start_time,log_all_directory=log_all_directory)
        return

def train(train_loader, model, criterion, optimizer,epoch,auxiliary,augmentation,noise,noise_type,location,major_function,major,start_time,log_all_directory):
    """
        Run one train epoch
    """ 
    losses = AverageMeter()
    acc = AverageMeter()
    acc_rot = AverageMeter()
    regression_loss = nn.MSELoss().cuda()
    # switch to train mode
    model.train()
    location_seperated = major_function.find("_")
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        
        if auxiliary == 'rotation':
            if augmentation == 2:
                if noise == True:
                    if noise_type == 'pair':
                        if major_function == 'default':
                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary) 

                        elif major_function[location_seperated+1:] == 'increasing':
                            if major_function[:location_seperated] == 'linearly':
                                major = args.major + (1-args.major)/args.epochs*epoch
                            elif major_function[:location_seperated] == 'exponentially':
                                major = math.exp(math.log(args.major)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)                                 

                        elif major_function[location_seperated+1:] == 'decreasing':
                            if major_function[:location_seperated] == 'linearly':
                                major = 1 - (1-args.major)/args.epochs*epoch
                            elif major_function[:location_seperated] == 'exponentially':
                                major = 1+args.major - math.exp(math.log(args.major)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)     
                        
                        elif major_function == 'dirichlet':
                            dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.tensor([major,major]))
                            dirichlet_sample = dirichlet_distribution.sample()
                            major,small = float(max(dirichlet_sample)),float(min(dirichlet_sample))

                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)
                         
                    elif noise_type == 'symmetry':
                        if major_function == 'default':
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary) 

                        elif major_function[location_seperated+1:] == 'increasing':
                            if major_function[:location_seperated] == 'linearly':
                                major = args.major + (1-args.major)/args.epochs*epoch
                            elif major_function[:location_seperated] == 'exponentially':
                                major = math.exp(math.log(args.major)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary)                                 

                        elif major_function[location_seperated+1:] == 'decreasing':
                            if major_function[:location_seperated] == 'linearly':
                                major = 1 - (1-args.major)/args.epochs*epoch
                            elif major_function[:location_seperated] == 'exponentially':
                                major = 1+args.major - math.exp(math.log(args.major)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary)     
                        
                        
                    
                elif noise == False:
                    input_var,target_rot,target_var = rotation_loss.__dict__['rotation_without_noise'](input_var,target_var,major,location)  

                optimizer.zero_grad()
                output, output_rot = model(input_var)
                if args.loss == 'softmax':
                    soft = F.log_softmax(output_rot,dim=1)
                    loss = criterion(output,target_var) + torch.mean(-torch.sum(target_rot*soft,dim=1))
                elif args.loss == 'mseloss':
                    loss = criterion(output,target_var) + torch.sqrt(regression_loss(output_rot,target_rot))
                
                output, output_rot = output.float(), output_rot.float()
                prec1 = accuracy(output.data, target_var)[0]
                prec_rot = accuracy(output_rot.data,torch.argmax(target_rot,dim=1))[0]
                loss.backward()
                optimizer.step()
                loss = loss.float()
                acc.update(prec1.item(), input_var.size(0))    
                acc_rot.update(prec_rot.item(),input_var.size(0))
                losses.update(loss.item(), input_var.size(0))



            elif augmentation == 4:
                input_var,target_rot,target_var = rotation_loss.__dict__['rotation_4'](input_var,target_var,major,location)  
                optimizer.zero_grad()
                output, output_rot = model(input_var)
                loss = criterion(output,target_var) + criterion(output_rot,target_rot)
                output, output_rot = output.float(), output_rot.float()
                prec1 = accuracy(output.data, target_var)[0]
                prec_rot = accuracy(output_rot.data,torch.argmax(target_rot,dim=1))[0]
                acc_rot.update(prec_rot.item(),input_var.size(0))    
                loss.backward()
                optimizer.step()
                loss = loss.float()
                acc.update(prec1.item(), input_var.size(0))
                losses.update(loss.item(), input_var.size(0))                    
           
        elif auxiliary == 'color':
            if augmentation == 2:
                if noise == True:
                    if noise_type == 'pair':
                        if major_function == 'default':
                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary) 
                        elif major_function == 'increasing':
                            if args.major == 'linear':
                                major = 0.7 + 0.3/args.epochs*epoch
                            elif args.major == 'exponential':
                                major = math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)                                 

                        elif major_function == 'decreasing':
                            if args.major == 'linear':
                                major = 1 - 0.3/args.epochs*epoch
                            elif args.major == 'exponential':
                                major = 1.7 - math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)     
                        
                        elif major_function == 'dirichlet':
                            dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.tensor([major,major]))
                            dirichlet_sample = dirichlet_distribution.sample()
                            major,small = float(max(dirichlet_sample)),float(min(dirichlet_sample))
                            input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)
                         
                    elif noise_type == 'symmetry':
                        if major_function == 'default':
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary) 
                        elif major_function == 'increasing':
                            if args.major == 'linear':
                                major = 0.7 + 0.3/args.epochs*epoch
                            elif args.major == 'exponential':
                                major = math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary)                                 

                        elif major_function == 'decreasing':
                            if args.major == 'linear':
                                major = 1 - 0.3/args.epochs*epoch
                            elif args.major == 'exponential':
                                major = 1.7 - math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary)    

                elif noise == False:
                    input_var,target_rot,target_var = rotation_loss.__dict__['color_without_noise'](input_var,target_var,major,location)

                optimizer.zero_grad()
                output, output_rot = model(input_var)
                if args.loss == 'softmax':
                    soft = F.log_softmax(output_rot,dim=1)
                    loss = criterion(output,target_var) + torch.mean(-torch.sum(target_rot*soft,dim=1))
                elif args.loss == 'mseloss':
                    loss = criterion(output,target_var) + torch.sqrt(regression_loss(output_rot,target_rot))
                
                output, output_rot = output.float(), output_rot.float()
                prec1 = accuracy(output.data, target_var)[0]
                prec_rot = accuracy(output_rot.data,torch.argmax(target_rot,dim=1))[0]
                loss.backward()
                optimizer.step()
                loss = loss.float()
                acc.update(prec1.item(), input_var.size(0))    
                acc_rot.update(prec_rot.item(),input_var.size(0))
                losses.update(loss.item(), input_var.size(0))

            elif augmentation == 6:
                input_var,target_rot,target_var = rotation_loss.__dict__['color_6'](input_var,target_var,major,location)  
                optimizer.zero_grad()
                output, output_rot = model(input_var)
                loss = criterion(output,target_var) + criterion(output_rot,target_rot)
                output, output_rot = output.float(), output_rot.float()
                prec1 = accuracy(output.data, target_var)[0]
                prec_rot = accuracy(output_rot.data,torch.argmax(target_rot,dim=1))[0]
                acc_rot.update(prec_rot.item(),input_var.size(0))    
                loss.backward()
                optimizer.step()
                loss = loss.float()
                acc.update(prec1.item(), input_var.size(0))
                losses.update(loss.item(), input_var.size(0)) 
            elif augmentation == 3:
                input_var,target_rot,target_var = rotation_loss.__dict__['color_3'](input_var,target_var,major,location)  
                optimizer.zero_grad()
                output, output_rot = model(input_var)
                loss = criterion(output,target_var) + criterion(output_rot,target_rot)
                output, output_rot = output.float(), output_rot.float()
                prec1 = accuracy(output.data, target_var)[0]
                prec_rot = accuracy(output_rot.data,torch.argmax(target_rot,dim=1))[0]
                acc_rot.update(prec_rot.item(),input_var.size(0))    
                loss.backward()
                optimizer.step()
                loss = loss.float()
                acc.update(prec1.item(), input_var.size(0))
                losses.update(loss.item(), input_var.size(0)) 

        elif auxiliary == 'rotation_color' or auxiliary == 'color_rotation':
            if augmentation == 2:
                if noise == True:
                    if noise_type == 'pair':
                        if location == 'stochastic':
                            if major_function == 'default':
                                input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary) 
                            elif major_function == 'increasing':
                                if args.major == 'linear':
                                    major = 0.7 + 0.3/args.epochs*epoch
                                elif args.major == 'exponential':
                                    major = math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                                input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)                                 

                            elif major_function == 'decreasing':
                                if args.major == 'linear':
                                    major = 1 - 0.3/args.epochs*epoch
                                elif args.major == 'exponential':
                                    major = 1.7 - math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                                input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)     
                            
                            elif major_function == 'dirichlet':
                                dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.tensor([major,major]))
                                dirichlet_sample = dirichlet_distribution.sample()
                                major,small = float(max(dirichlet_sample)),float(min(dirichlet_sample))
                                input_var,target_rot,target_var = rotation_loss.__dict__['pair'](input_var,target_var,major,location,auxiliary)
                        else:
                            raise Exception('location : only stochastic')
                    elif noise_type == 'symmetry':
                        if major_function == 'default':
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary) 
                        elif major_function == 'increasing':
                            if args.major == 'linear':
                                major = 0.7 + 0.3/args.epochs*epoch
                            elif args.major == 'exponential':
                                major = math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary)                                 

                        elif major_function == 'decreasing':
                            if args.major == 'linear':
                                major = 1 - 0.3/args.epochs*epoch
                            elif args.major == 'exponential':
                                major = 1.7 - math.exp(math.log(0.7)/args.epochs*(args.epochs-epoch))
                            input_var,target_rot,target_var = rotation_loss.__dict__['symmetry'](input_var,target_var,major,location,auxiliary) 

                elif noise == False:
                    input_var,target_rot,target_var = rotation_loss.__dict__['joint_without_noise'](input_var,target_var,major,location)
                optimizer.zero_grad()
                output, output_rot = model(input_var)
                if args.loss == 'softmax':
                    soft = F.log_softmax(output_rot,dim=1)
                    loss = criterion(output,target_var) + torch.mean(-torch.sum(target_rot*soft,dim=1))
                elif args.loss == 'mseloss':
                    loss = criterion(output,target_var) + torch.sqrt(regression_loss(output_rot,target_rot))
                
                output, output_rot = output.float(), output_rot.float()
                prec1 = accuracy(output.data, target_var)[0]
                prec_rot = accuracy(output_rot.data,torch.argmax(target_rot,dim=1))[0]
                loss.backward()
                optimizer.step()
                loss = loss.float()
                acc.update(prec1.item(), input_var.size(0))    
                acc_rot.update(prec_rot.item(),input_var.size(0))
                losses.update(loss.item(), input_var.size(0))

            elif augmentation == 24:
                if noise == False:
                    input_var,target_rot,target_var = rotation_loss.__dict__['joint_24'](input_var,target_var,major,location)
                    optimizer.zero_grad()
                    output, output_rot = model(input_var)
                    loss = criterion(output,target_var) + criterion(output_rot,target_rot)
                    output, output_rot = output.float(), output_rot.float()
                    prec1 = accuracy(output.data, target_var)[0]
                    prec_rot = accuracy(output_rot.data,torch.argmax(target_rot,dim=1))[0]
                    acc_rot.update(prec_rot.item(),input_var.size(0))    
                    loss.backward()
                    optimizer.step()
                    loss = loss.float()
                    acc.update(prec1.item(), input_var.size(0))
                    losses.update(loss.item(), input_var.size(0)) 




        elif auxiliary == False: 

            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer.zero_grad()
            output = output.float()
            prec1 = accuracy(output.data, target)[0]   
            loss.backward()
            optimizer.step()
            loss = loss.float()
            acc.update(prec1.item(), input_var.size(0))    
            losses.update(loss.item(), input_var.size(0))

        # measure elapsed time
        tm = time.localtime(time.time())
        string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
        file = open(log_all_directory+start_time+'.txt','a')
        if i % 50 == 0:
            print('{0} -'
                  ' Epoch: [{1}][{2}/{3}] -'
                  ' learning rate: {4:0.5e} -'
                  ' Loss: {5:0.4f} -'
                  ' main acc: {6:0.2f} %'
                  ' auxiliary acc: {7:0.2f} %'.format(string,
                      epoch, i, len(train_loader),optimizer.param_groups[0]['lr'],
                       losses.val,acc.val,acc_rot.val))
            
            file.write('{0} -'
                  ' Epoch: [{1}][{2}/{3}] -'
                  ' learning rate: {4:0.5e} -'
                  ' Loss: {5:0.4f} -'
                  ' acc: {6:0.2f} %'
                  ' auxiliary acc: {7:0.2f} %\n'.format(string,
                      epoch, i, len(train_loader),optimizer.param_groups[0]['lr'],
                       losses.val,acc.val,acc_rot.val))           
    print('average training accuracy: {acc.avg:.3f}'
          .format(acc=acc))                   
    file.write('average training accuracy: {acc.avg:.3f}\n'
          .format(acc=acc))
    file.close()

def validate(val_loader, model,optimizer,criterion,auxiliary,start_time,log_all_directory):
    """
    Run evaluation
    """
    losses = AverageMeter()
    acc = AverageMeter()
    acc_rot = AverageMeter()
    regression_loss = nn.MSELoss().cuda()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            
            if auxiliary == 'rotation':
                output, output_rot = model(input_var)
                
            else:    
                # compute output
                output = model(input_var)
                
            loss = criterion(output,target_var)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            acc.update(prec1.item(), input.size(0))
                  

            # measure elapsed time
            tm = time.localtime(time.time())
            string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
            file = open(log_all_directory+start_time+'.txt','a')
            if i % 50 == 0:
                print('{0} -'
                    ' Epoch: [{1}/{2}] -'
                    ' learning rate: {3:0.5e} -'
                    ' Loss: {4:0.4f} -'
                    ' acc: {5:0.2f} %'.format(string,
                        i, len(val_loader),optimizer.param_groups[0]['lr'],
                        losses.val,acc.val))
                file.write('{0} -'
                    ' Epoch: [{1}/{2}] -'
                    ' learning rate: {3:0.5e} -'
                    ' Loss: {4:0.4f} -'
                    ' acc: {5:0.2f} %\n'.format(string,
                        i, len(val_loader),optimizer.param_groups[0]['lr'],
                        losses.val,acc.val))        
    print('average validation accuracy: {acc.avg:.3f}'
          .format(acc=acc))
    file.write('---------------------------------------------\n'
               '|      average validation accuracy          |\n'
               '|             {acc.avg:.3f}                        |\n'
               '---------------------------------------------\n'
          .format(acc=acc))     
    file.close()
    return acc.avg
class AverageMeter(object):   #average sum 등등 만들어주는 클래스
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)  # 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res







if __name__ == '__main__':              #이 파일을 직접실행했을때만 main() 함수를 실행시켜라
    main()                          
