
# CMW-Net-SL
nohup bash mwn_train_v13.sh >> aircraft_cmwn_sl.log &

# CMW-Net
nohup bash mwn_train_v13_v2.sh >> aircraft_cmwn.log &

# MW-Net
nohup bash mwn_train_v13_v1.sh >> aircraft_mwn.log &
