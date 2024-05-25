
OMP_NUM_THREADS=4 python train.py --gammaD 1 --gammaG 1 \
    --seed 4115 --encoded_noise --preprocessing --cuda --nepoch 400 --ngh 4096 \
    --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 \
    --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 \
    --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --matRoot datasets --loop 2

