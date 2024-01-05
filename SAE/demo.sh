# There must be some mistake!

# [1] zsl accuracy for APY dataset [F >>> S]: 5.50%
# [2] zsl accuracy for APY dataset [S >>> F]: 7.72%
python SAE.py --dataset APY --ld 50000

# [1] zsl accuracy for AWA1 dataset [F >>> S]: 45.96%
# [2] zsl accuracy for AWA1 dataset [S >>> F]: 51.64%
python SAE.py --dataset AWA1 --ld 50000

# [1] zsl accuracy for AWA2 dataset [F >>> S]: 39.97%
# [2] zsl accuracy for AWA2 dataset [S >>> F]: 49.93%
python SAE.py --dataset AWA2 --ld 50000

# [1] zsl accuracy for CUB dataset [F >>> S]: 33.50%
# [2] zsl accuracy for CUB dataset [S >>> F]: 29.90%
python SAE.py --dataset CUB --ld 50000

# [1] zsl accuracy for SUN dataset [F >>> S]: 40.21%
# [2] zsl accuracy for SUN dataset [S >>> F]: 30.56%
python SAE.py --dataset SUN --ld 50000