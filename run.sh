#!/bin/sh
python3 self_Rotation.py --dataset cifar100 --arch resnetself --auxiliary rotation --augmentation False --noise --noise-type False --location False --major-function False --major False 
python3 self_rotation.py --dataset cifar10 --arch resnetself --auxiliary rotation --augmentation 2 --noise --noise-type pair --location next --major-function default --major 0.7 --loss softmax
 

python3 alarm.py



