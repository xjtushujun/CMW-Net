
### cifar10
nohup python -u main.py --dataset cifar10 --model resnet --partial_type binomial --partial_rate 0.1 >> cifar10_1.log &

nohup python -u main.py --dataset cifar10 --model resnet --partial_type binomial --partial_rate 0.3 >> cifar10_3.log &

nohup python -u main.py --dataset cifar10 --model resnet --partial_type binomial --partial_rate 0.5 >> cifar10_5.log &

nohup python -u main.py --dataset cifar10 --model resnet --partial_type binomial --partial_rate 0.7 >> cifar10_7.log &



### cifar100
nohup python -u main.py --dataset cifar100 --model resnet --partial_type binomial --partial_rate 0.03 >> cifar100_3.log &

nohup python -u main.py --dataset cifar100 --model resnet --partial_type binomial --partial_rate 0.05 >> cifar100_5.log &

nohup python -u main.py --dataset cifar100 --model resnet --partial_type binomial --partial_rate 0.07 >> cifar100_7.log &

nohup python -u main.py --dataset cifar100 --model resnet --partial_type binomial --partial_rate 0.10 >> cifar100_10.log &


