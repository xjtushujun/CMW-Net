
# cifar10 type1 35% unif 30%
nohup python -u mwn.py --noise_label_file cifar10-1-0.35.npy --dataset cifar10 --corruption_type unif --corruption_prob 0.3 >> cifar10_type1_unif.log &

# cifar10 type1 35% flip 30%
nohup python -u mwn.py --noise_label_file cifar10-1-0.35.npy --dataset cifar10 --corruption_type flip --corruption_prob 0.3 >> cifar10_type1_flip.log &


# cifar10 type2 35% unif 30%
nohup python -u mwn.py --noise_label_file cifar10-2-0.35.npy --dataset cifar10 --corruption_type unif --corruption_prob 0.3 >> cifar10_type2_unif.log &

# cifar10 type2 35% flip 30%
nohup python -u mwn.py --noise_label_file cifar10-2-0.35.npy --dataset cifar10 --corruption_type flip --corruption_prob 0.3 >> cifar10_type2_flip.log &


# cifar10 type3 35% unif 30%
nohup python -u mwn.py --noise_label_file cifar10-3-0.35.npy --dataset cifar10 --corruption_type unif --corruption_prob 0.3 >> cifar10_type3_unif.log &

# cifar10 type3 35% flip 30%
nohup python -u mwn.py --noise_label_file cifar10-3-0.35.npy --dataset cifar10 --corruption_type flip --corruption_prob 0.3 >> cifar10_type3_flip.log &




# cifar100 type1 35% unif 30%
nohup python -u mwn.py --noise_label_file cifar100-1-0.35.npy --dataset cifar100 --corruption_type unif --corruption_prob 0.3 >> cifar100_type1_unif.log &

# cifar100 type1 35% flip 30%
nohup python -u mwn.py --noise_label_file cifar100-1-0.35.npy --dataset cifar100 --corruption_type flip --corruption_prob 0.3 >> cifar100_type1_flip.log &


# cifar100 type2 35% unif 30%
nohup python -u mwn.py --noise_label_file cifar100-2-0.35.npy --dataset cifar100 --corruption_type unif --corruption_prob 0.3 >> cifar100_type2_unif.log &

# cifar100 type2 35% flip 30%
nohup python -u mwn.py --noise_label_file cifar100-2-0.35.npy --dataset cifar100 --corruption_type flip --corruption_prob 0.3 >> cifar100_type2_flip.log &


# cifar100 type3 35% unif 30%
nohup python -u mwn.py --noise_label_file cifar100-3-0.35.npy --dataset cifar100 --corruption_type unif --corruption_prob 0.3 >> cifar100_type3_unif.log &

# cifar100 type3 35% flip 30%
nohup python -u mwn.py --noise_label_file cifar100-3-0.35.npy --dataset cifar100 --corruption_type flip --corruption_prob 0.3 >> cifar100_type3_flip.log &










