CUDA_VISIBLE_DEVICES=3 nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 dns_interspeech_2020/train.py -C dns_interspeech_2020/EinFFT_MBFF/train.toml > train.log 2>&1 &


