# 验证集上的最佳参数 (在测试集上未必最佳)

# APY --> Top-1 acc: 0.608414471617502(val) | 0.349980418781888(test)
python EsZSL.py --dataset APY --Gamma 2 --Lambda 0
# AWA1 --> Top-1 acc: 0.5752166125604982(val) | 0.561923169812470(test)
python EsZSL.py --dataset AWA1 --Gamma 3 --Lambda 0
# AWA2 --> Top-1 acc: 0.6266412137155378(val) | 0.5450310779325793(test)
python EsZSL.py --dataset AWA2 --Gamma 3 --Lambda 0
# CUB --> Top-1 acc: 0.5062013087372061(val) | 0.519153390696949(test)
python EsZSL.py --dataset CUB --Gamma 3 --Lambda 0
# SUN --> Top-1 acc: 0.5557692307692308(val) | 0.55625(test)
python EsZSL.py --dataset SUN --Gamma 3 --Lambda 2