
# cifar10 type1 35%
nohup python -u mwn.py --noise_label_file cifar10-1-0.35.npy --dataset cifar10 >> cifar10_type1_35.log &

# cifar10 type1 70%
nohup python -u mwn.py --noise_label_file cifar10-1-0.70.npy --dataset cifar10 >> cifar10_type1_70.log &


# cifar10 type2 35%
nohup python -u mwn.py --noise_label_file cifar10-2-0.35.npy --dataset cifar10 >> cifar10_type2_35.log &

# cifar10 type2 70%
nohup python -u mwn.py --noise_label_file cifar10-2-0.70.npy --dataset cifar10 >> cifar10_type2_70.log &


# cifar10 type3 35%
nohup python -u mwn.py --noise_label_file cifar10-3-0.35.npy --dataset cifar10 >> cifar10_type3_35.log &

# cifar10 type3 70%
nohup python -u mwn.py --noise_label_file cifar10-3-0.70.npy --dataset cifar10 >> cifar10_type3_70.log &




# cifar100 type1 35%
nohup python -u mwn.py --noise_label_file cifar100-1-0.35.npy --dataset cifar100 >> cifar100_type1_35.log &

# cifar100 type1 70%
nohup python -u mwn.py --noise_label_file cifar100-1-0.70.npy --dataset cifar100 >> cifar100_type1_70.log &


# cifar100 type2 35%
nohup python -u mwn.py --noise_label_file cifar100-2-0.35.npy --dataset cifar100 >> cifar100_type2_35.log &

# cifar100 type2 70%
nohup python -u mwn.py --noise_label_file cifar100-2-0.70.npy --dataset cifar100 >> cifar100_type2_70.log &


# cifar100 type3 35%
nohup python -u mwn.py --noise_label_file cifar100-3-0.35.npy --dataset cifar100 >> cifar100_type3_35.log &

# cifar100 type3 70%
nohup python -u mwn.py --noise_label_file cifar100-3-0.70.npy --dataset cifar100 >> cifar100_type3_70.log &














