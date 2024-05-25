
OMP_NUM_THREADS=4 python train.py --gammaD 10 --gammaG 10 \
    --seed 3483 --encoded_noise --preprocessing --cuda --nepoch 300 --ngh 4096 --ndh 4096 \
    --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
    --matRoot datasets --dataset CUB \
    --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
    --loop 2

