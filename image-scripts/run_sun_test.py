#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.system('''OMP_NUM_THREADS=8  python sun_test.py --seed 255 --gammaD 10 --gammaG 10 --pretrained --is_fix --workers 8 --lat_bias 0.4 --lat_num 1000 \
--gzsl --encoded_noise --preprocessing --cuda --class_embedding att --contrastive_lambda 0.001 --rank_lambda 0.00001 --tem 0.5 --tem2 0.5 \
--nepoch 1000 --ngh 4096 --ndh 4096 --lr 0.00005 --classifier_lr 0.0005 --lambda1 10 --critic_iter 5 --dataroot data --dataset SUN \
--nclass_all 717 --batch_size 10 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --syn_num 100 \
--a1 1 --a2 1 --dec_lr 0.00005 ''')