
### CE loss

# cifar10 IF=1
nohup python -u LT1.py --dataset cifar10 --imb_factor 1 >> cifar10_1.log &

# cifar10 IF=10
nohup python -u LT1.py --dataset cifar10 --imb_factor 0.1 >> cifar10_10.log &

# cifar10 IF=20
nohup python -u LT1.py --dataset cifar10 --imb_factor 0.05 >> cifar10_20.log &

# cifar10 IF=50
nohup python -u LT1.py --dataset cifar10 --imb_factor 0.02 >> cifar10_50.log &

# cifar10 IF=100
nohup python -u LT1.py --dataset cifar10 --imb_factor 0.01 >> cifar10_100.log &

# cifar10 IF=200
nohup python -u LT1.py --dataset cifar10 --imb_factor 0.005 >> cifar10_200.log &


# cifar100 IF=1
nohup python -u LT1.py --dataset cifar100 --imb_factor 1 >> cifar100_1.log &

# cifar100 IF=10
nohup python -u LT1.py --dataset cifar100 --imb_factor 0.1 >> cifar100_10.log &

# cifar100 IF=20
nohup python -u LT1.py --dataset cifar100 --imb_factor 0.05 >> cifar100_20.log &

# cifar100 IF=50
nohup python -u LT1.py --dataset cifar100 --imb_factor 0.02 >> cifar100_50.log &

# cifar100 IF=100
nohup python -u LT1.py --dataset cifar100 --imb_factor 0.01 >> cifar100_100.log &

# cifar100 IF=200
nohup python -u LT1.py --dataset cifar100 --imb_factor 0.005 >> cifar100_200.log &



### LDAM loss

nohup python -u LT2.py --dataset cifar10 --imb_factor 1 >> cifar10_1.log &

# cifar10 IF=10
nohup python -u LT2.py --dataset cifar10 --imb_factor 0.1 >> cifar10_10.log &

# cifar10 IF=20
nohup python -u LT2.py --dataset cifar10 --imb_factor 0.05 >> cifar10_20.log &

# cifar10 IF=50
nohup python -u LT2.py --dataset cifar10 --imb_factor 0.02 >> cifar10_50.log &

# cifar10 IF=100
nohup python -u LT2.py --dataset cifar10 --imb_factor 0.01 >> cifar10_100.log &

# cifar10 IF=200
nohup python -u LT2.py --dataset cifar10 --imb_factor 0.005 >> cifar10_200.log &


# cifar100 IF=1
nohup python -u LT2.py --dataset cifar100 --imb_factor 1 >> cifar100_1.log &

# cifar100 IF=10
nohup python -u LT2.py --dataset cifar100 --imb_factor 0.1 >> cifar100_10.log &

# cifar100 IF=20
nohup python -u LT2.py --dataset cifar100 --imb_factor 0.05 >> cifar100_20.log &

# cifar100 IF=50
nohup python -u LT2.py --dataset cifar100 --imb_factor 0.02 >> cifar100_50.log &

# cifar100 IF=100
nohup python -u LT2.py --dataset cifar100 --imb_factor 0.01 >> cifar100_100.log &

# cifar100 IF=200
nohup python -u LT2.py --dataset cifar100 --imb_factor 0.005 >> cifar100_200.log &
