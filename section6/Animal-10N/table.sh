
nohup python -u Resnet50_MWN_webvision_v2.py --root_dir ./data/webvision/ --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:25' >> webvision_trans.log &

