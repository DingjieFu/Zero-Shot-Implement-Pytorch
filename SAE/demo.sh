# There must be some mistake!

# [1] zsl accuracy for APY dataset [F >>> S]: 5.50%
# [2] zsl accuracy for APY dataset [S >>> F]: 5.50%
python SAE.py --dataset APY --ld 1000000
# [1] zsl accuracy for AWA1 dataset [F >>> S]: 45.93%
# [2] zsl accuracy for AWA1 dataset [S >>> F]: 50.40%
python SAE.py --dataset AWA1 --ld 1000000
# [1] zsl accuracy for AWA2 dataset [F >>> S]: 39.72%
# [2] zsl accuracy for AWA2 dataset [S >>> F]: 43.43%
python SAE.py --dataset AWA2 --ld 1000000
# [1] zsl accuracy for CUB dataset [F >>> S]: 30.57%
# [2] zsl accuracy for CUB dataset [S >>> F]: 17.90%
python SAE.py --dataset CUB --ld 1000000
# [1] zsl accuracy for SUN dataset [F >>> S]: 38.61%
# [2] zsl accuracy for SUN dataset [S >>> F]: 33.19%
python SAE.py --dataset SUN --ld 1000000