#!/bin/sh


python3 self_rotation.py --dataset cifar10 --location previous --major-type linear --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar10 --location previous --major-type exponential --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar10 --location neither --major-type linear --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar10 --location neither --major-type linear --rotation-var 
python3 self_rotation.py --dataset cifar10 --location next --major-type exponential --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar10 --location previous --major 0.9 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar10 --location previous --major 0.8 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar10 --location previous --major 0.6 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar10 --location neither --major 0.9 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar10 --location neither --major 0.8 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar10 --location neither --major 0.6 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar10 --location neither --major 0.9 --rotation-var rotation_2_stochastic_pair_noise
python3 self_rotation.py --dataset cifar10 --location neither --major 0.8 --rotation-var rotation_2_stochastic_pair_noise
python3 self_rotation.py --dataset cifar10 --major 0.9 --rotation-var rotation_2_symmetry_noise
python3 self_rotation.py --dataset cifar10 --major 0.8 --rotation-var rotation_2_symmetry_noise
python3 self_rotation.py --dataset cifar10 --major 0.6 --rotation-var rotation_2_symmetry_noise

python3 self_rotation.py --dataset cifar100 --location previous --major-type linear --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar100 --location previous --major-type exponential --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar100 --location neither --major-type linear --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar100 --location neither --major-type linear --rotation-var 
python3 self_rotation.py --dataset cifar100 --location next --major-type exponential --rotation-var rotation_2_pair_major_different_with_epoch
python3 self_rotation.py --dataset cifar100 --location previous --major 0.9 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar100 --location previous --major 0.8 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar100 --location previous --major 0.6 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar100 --location neither --major 0.9 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar100 --location neither --major 0.8 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar100 --location neither --major 0.6 --rotation-var rotation_2_pair_noise
python3 self_rotation.py --dataset cifar100 --location neither --major 0.9 --rotation-var rotation_2_stochastic_pair_noise
python3 self_rotation.py --dataset cifar100 --location neither --major 0.8 --rotation-var rotation_2_stochastic_pair_noise
python3 self_rotation.py --dataset cifar100 --major 0.9 --rotation-var rotation_2_symmetry_noise
python3 self_rotation.py --dataset cifar100 --major 0.8 --rotation-var rotation_2_symmetry_noise
python3 self_rotation.py --dataset cifar100 --major 0.6 --rotation-var rotation_2_symmetry_noise




python3 alarm.py




# cifar100도 둘다 돌아가도록 해보자
#previous neither linear exp pair noise, next exponential pair noise,previous neither 9 8 6 pair noise,stochastic 9 8 pair noise, 
#9 8 6 original symmetry noise




#m = torch.distributions.dirichlet.Dirichlet(torch.tensor([0.5,0.5])) m.sample() to pair noise or symmetric noise)




