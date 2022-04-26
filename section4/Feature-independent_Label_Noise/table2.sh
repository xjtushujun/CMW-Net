
# cifar10 sym 20%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode sym --r 0.2 >> cifar10_sym_20.log &

# cifar10 sym 40%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode sym --r 0.4 >> cifar10_sym_40.log &

# cifar10 sym 60%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode sym --r 0.6 >> cifar10_sym_60.log &

# cifar10 sym 80%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode sym --r 0.8 >> cifar10_sym_80.log &


# cifar10 asym 20%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode asym --r 0.2 >> cifar10_asym_20.log &

# cifar10 asym 40%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode asym --r 0.4 >> cifar10_asym_40.log &

# cifar10 asym 60%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode asym --r 0.6 >> cifar10_asym_60.log &

# cifar10 asym 80%
nohup python -u cmwn.py --dataset cifar10 --data_path cifar10-python --noise_mode asym --r 0.8 >> cifar10_asym_80.log &



# cifar100 sym 20%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode sym --r 0.2 >> cifar100_sym_20.log &

# cifar100 sym 40%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode sym --r 0.4 >> cifar100_sym_40.log &

# cifar100 sym 60%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode sym --r 0.6 >> cifar100_sym_60.log &

# cifar100 sym 80%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode sym --r 0.8 --alpha 0.1 >> cifar100_sym_80.log &

# cifar10 asym 20%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode asym --r 0.2 --alpha 0.4 >> cifar100_asym_20.log &

# cifar10 asym 40%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode asym --r 0.4 --alpha 0.4 >> cifar100_asym_40.log &

# cifar10 asym 60%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode asym --r 0.6 --alpha 4. >> cifar100_asym_60.log &

# cifar10 asym 80%
nohup python -u cmwn.py --dataset cifar100 --data_path cifar100-python --noise_mode asym --r 0.8 >> cifar100_asym_80.log &


