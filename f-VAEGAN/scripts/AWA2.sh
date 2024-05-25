
OMP_NUM_THREADS=4 python train.py --gammaD 10 --gammaG 10 \
    --encoded_noise --seed 9182 --preprocessing --cuda --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 \
    --lambda1 10 --critic_iter 5 \
    --nclass_all 50 --matRoot datasets --dataset AWA2 \
    --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
    --lr 0.00001 --classifier_lr 0.001 --loop 2
