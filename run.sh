#!/bin/sh

# for num in 0.9 0.8 0.7 0.6
#     do
#         python3 self_rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --major-function default --location stochastic --major $num
#         python3 self_rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --major-function default --location stochastic --major $num --loss softmax 
#         python3 self_rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --major-function default --location False --major $num

#     done

#cifar10 뺏음 b 1번째
for b in cifar100
    do
        for a in next stochastic
            do
                python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location $a --major-function increasing --major exponential
                python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location $a --major-function decreasing --major linear --loss softmax
                python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location $a --major-function decreasing --major exponential --loss softmax
                python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location $a --major-function increasing --major linear --loss softmax
                python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location $a --major-function increasing --major exponential --loss softmax
            done
        
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function decreasing --major linear
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function decreasing --major exponential
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function increasing --major linear
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function increasing --major exponential
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function decreasing --major linear --loss softmax
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function decreasing --major exponential --loss softmax
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function increasing --major linear --loss softmax
        python3 self_rotation.py --dataset $b --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type symmetry --location False --major-function increasing --major exponential --loss softmax

    done
python3 self_rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location stochastic --major-function decreasing --major linear
python3 self_rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location stochastic --major-function decreasing --major exponential
python3 self_rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location stochastic --major-function increasing --major linear
for number in 0.9 0.8 0.7 0.6
    do
        for dataset in cifar10 cifar100
            do
                python3 self_rotation.py --dataset $dataset --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location next --major-function dirichlet --major $number
                python3 self_rotation.py --dataset $dataset --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location stochastic --major-function dirichlet --major $number
                python3 self_rotation.py --dataset $dataset --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type stochastic --location False --major-function dirichlet --major $number
                python3 self_rotation.py --dataset $dataset --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location next --major-function dirichlet --major $number --loss softmax
                python3 self_rotation.py --dataset $dataset --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location stochastic --major-function dirichlet --major $number --loss softmax
                python3 self_rotation.py --dataset $dataset --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type stochastic --location False --major-function dirichlet --major $number --loss softmax
            done
    done

python3 self_rotation.py --dataset cifar10 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location next --major-function default --major 0.7 --loss softmax



python3 self_Rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation False --noise --noise-type False --location False --major-function False --major False 

# stochastic pair cifar 100 9,8,7,6
# symmetry cifar 100 9 8 7 6
#cifar10,cifar100 -> pair,symmetry -> next ,stochastic -> decreasing(linear,exponential),increasing(linear,exponential),dirichlet
#m = torch.distributions.dirichlet.Dirichlet(torch.tensor([0.5,0.5])) m.sample() to pair noise or symmetry noise)
#cross entropy -> 바꿔서
#visdom
#auxiliary acc만


#colorwithout noise,color3,color6,joint training with color_rotation with noise, resnet output 바꾸야함


python3 alarm.py



